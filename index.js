const express = require("express");

const app = express();
app.use(express.json());

// ---- Config (set these as environment variables on Render) ----
const {
  VERIFY_TOKEN,       // any string you invent; must match Meta webhook setup
  WHATSAPP_TOKEN,     // access token from Meta (System User token, not the temporary Explorer one)
  PHONE_NUMBER_ID,    // the test number's Phone Number ID (not the phone number itself)
  GEMINI_API_KEY,     // free key from Google AI Studio
} = process.env;

// If GEMINI_MODEL is set explicitly, it's honored as a pin/override.
// If left unset, the server auto-discovers the best current model from
// Google's API and re-discovers whenever the pinned/cached model stops working.
const GEMINI_MODEL_OVERRIDE = process.env.GEMINI_MODEL || null;
const GRAPH_VERSION = process.env.GRAPH_VERSION || "v21.0";
const SYSTEM_PROMPT =
  process.env.SYSTEM_PROMPT ||
  "You are a helpful WhatsApp assistant. Keep replies short and clear.";
const PORT = process.env.PORT || 3000;

for (const [name, value] of Object.entries({
  VERIFY_TOKEN,
  WHATSAPP_TOKEN,
  PHONE_NUMBER_ID,
  GEMINI_API_KEY,
})) {
  if (!value) {
    console.error(`Missing environment variable: ${name}`);
    process.exit(1);
  }
}

const GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta";
const WA_URL = `https://graph.facebook.com/${GRAPH_VERSION}/${PHONE_NUMBER_ID}/messages`;

// ---- Automatic Gemini model resolution ----
// Rather than hardcoding a model name that Google can deprecate at any time,
// we ask the API which models currently exist and pick the best match.
// Result is cached and refreshed periodically (and on demand if a call fails).
const MODEL_CACHE_TTL_MS = 6 * 60 * 60 * 1000; // re-check every 6 hours
let modelCache = { name: null, resolvedAt: 0 };

// Preference order: newest/best "flash" family first, since that's what this
// bot was built around (fast + cheap). Adjust this list if you'd rather
// default to "pro" models instead.
const MODEL_PREFERENCE_PATTERNS = [
  /^models\/gemini-.*flash-latest$/,
  /^models\/gemini-\d+(\.\d+)*-flash$/,
  /^models\/gemini-\d+(\.\d+)*-flash-\d+$/,
  /^models\/gemini-.*flash.*/,
  /^models\/gemini-.*pro.*/,
];

async function listAvailableModels() {
  const res = await fetch(`${GEMINI_API_BASE}/models`, {
    headers: { "x-goog-api-key": GEMINI_API_KEY },
  });
  if (!res.ok) {
    throw new Error(`Failed to list Gemini models: ${res.status} ${await res.text()}`);
  }
  const data = await res.json();
  return (data.models || []).filter((m) =>
    (m.supportedGenerationMethods || []).includes("generateContent")
  );
}

function pickBestModel(models) {
  for (const pattern of MODEL_PREFERENCE_PATTERNS) {
    const match = models.find((m) => pattern.test(m.name));
    if (match) return match.name.replace(/^models\//, "");
  }
  // Fall back to whatever the API returns first, rather than failing outright.
  if (models.length > 0) return models[0].name.replace(/^models\//, "");
  throw new Error("No Gemini models with generateContent support are available");
}

async function resolveGeminiModel({ force = false } = {}) {
  if (GEMINI_MODEL_OVERRIDE) return GEMINI_MODEL_OVERRIDE;

  const isStale = Date.now() - modelCache.resolvedAt > MODEL_CACHE_TTL_MS;
  if (force || !modelCache.name || isStale) {
    const models = await listAvailableModels();
    const chosen = pickBestModel(models);
    if (chosen !== modelCache.name) {
      console.log(`Gemini model resolved: ${modelCache.name || "(none)"} -> ${chosen}`);
    }
    modelCache = { name: chosen, resolvedAt: Date.now() };
  }
  return modelCache.name;
}

// ---- Simple in-memory state (resets on restart/redeploy) ----
const MAX_MESSAGES = 10; // trimmed from 20 - shorter context, faster Gemini responses
const history = new Map();       // sender -> [{ role, parts }]
const seenIds = new Set();       // dedupe WhatsApp retries

function remember(sender, role, text) {
  const list = history.get(sender) || [];
  list.push({ role, parts: [{ text }] });
  while (list.length > MAX_MESSAGES) list.shift();
  history.set(sender, list);
}

async function callGemini(sender, modelName, { skipThinkingControl = false } = {}) {
  const url = `${GEMINI_API_BASE}/models/${modelName}:generateContent`;

  const generationConfig = { maxOutputTokens: 300 };
  if (!skipThinkingControl) {
    // Gemini 3.x models replaced the old numeric "thinkingBudget" with a
    // "thinkingLevel" enum (minimal/low/medium/high) and silently ignore
    // thinkingBudget entirely, defaulting to "medium" thinking. Since
    // thinking tokens are drawn from the same maxOutputTokens budget as the
    // actual reply, an ignored thinkingBudget can silently eat the whole
    // budget and leave nothing for the answer. "low" is the safest floor:
    // some model versions don't support "minimal" and reject it outright.
    generationConfig.thinkingConfig = { thinkingLevel: "low" };
  }

  return fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-goog-api-key": GEMINI_API_KEY,
    },
    body: JSON.stringify({
      system_instruction: { parts: [{ text: SYSTEM_PROMPT }] },
      contents: history.get(sender),
      generationConfig,
    }),
  });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function askGemini(sender, text) {
  remember(sender, "user", text);

  let modelName = await resolveGeminiModel();
  let skipThinkingControl = false;
  let res = await callGemini(sender, modelName, { skipThinkingControl });

  // If the model rejects the thinkingLevel field outright (older/newer
  // models sometimes disagree on the shape), retry once without it.
  if (!res.ok && res.status === 400) {
    const bodyText = await res.text();
    if (/thinking(Level|Config|Budget)/i.test(bodyText)) {
      console.warn("Gemini rejected thinkingConfig, retrying without it...");
      skipThinkingControl = true;
      res = await callGemini(sender, modelName, { skipThinkingControl });
    } else if (/not found|not supported|deprecated/i.test(bodyText)) {
      modelName = await resolveGeminiModel({ force: true });
      res = await callGemini(sender, modelName, { skipThinkingControl });
    } else {
      history.get(sender).pop();
      throw new Error(`Gemini error 400: ${bodyText}`);
    }
  } else if (!res.ok && res.status === 404 && !GEMINI_MODEL_OVERRIDE) {
    // Model retired/renamed - force a re-resolve and retry once.
    const bodyText = await res.text();
    console.warn(`Gemini model "${modelName}" seems unavailable, re-resolving...`);
    modelName = await resolveGeminiModel({ force: true });
    res = await callGemini(sender, modelName, { skipThinkingControl });
  }

  // 503 "overloaded" is usually a brief spike on Google's end - one quick
  // retry often succeeds without bothering the user.
  if (res.status === 503) {
    await sleep(1000);
    res = await callGemini(sender, modelName, { skipThinkingControl });
  }

  if (res.status === 429) {
    history.get(sender).pop();
    return "I'm getting too many requests right now. Please try again in a minute.";
  }
  if (res.status === 503) {
    history.get(sender).pop();
    return "Gemini is under heavy load right now. Please try again shortly.";
  }
  if (!res.ok) {
    history.get(sender).pop();
    throw new Error(`Gemini error ${res.status}: ${await res.text()}`);
  }

  const data = await res.json();
  const reply = data?.candidates?.[0]?.content?.parts?.[0]?.text?.trim();
  if (!reply) {
    history.get(sender).pop();
    console.error("Unexpected Gemini response:", JSON.stringify(data));
    return "Sorry, I couldn't generate a reply. Please try again.";
  }

  remember(sender, "model", reply);
  return reply.slice(0, 4000); // WhatsApp text limit is 4096
}

async function sendWhatsApp(to, body) {
  const res = await fetch(WA_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${WHATSAPP_TOKEN}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      messaging_product: "whatsapp",
      to,
      type: "text",
      text: { body },
    }),
  });
  if (!res.ok) {
    console.error("WhatsApp send failed:", res.status, await res.text());
  }
}

// Marks the incoming message as read AND shows the "typing..." bubble.
// The typing indicator disappears automatically once sendWhatsApp() replies,
// or after 25 seconds if no reply is sent by then.
async function showTypingIndicator(messageId) {
  const res = await fetch(WA_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${WHATSAPP_TOKEN}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      messaging_product: "whatsapp",
      status: "read",
      message_id: messageId,
      typing_indicator: { type: "text" },
    }),
  });
  if (!res.ok) {
    console.error("Typing indicator failed:", res.status, await res.text());
  }
}

async function handleMessage(msg) {
  const sender = msg.from;

  // Fire-and-forget: the typing indicator doesn't need to finish before we
  // start the (slower) Gemini call, so don't block on it.
  showTypingIndicator(msg.id).catch((err) =>
    console.error("Typing indicator failed:", err)
  );

  if (msg.type !== "text") {
    await sendWhatsApp(sender, "I can only read text messages for now.");
    return;
  }

  const text = msg.text.body;
  console.log(`Incoming from ${sender}: ${text}`);

  let reply;
  try {
    reply = await askGemini(sender, text);
  } catch (err) {
    console.error("Gemini call failed:", err);
    reply = "Something went wrong on my side. Please try again.";
  }
  await sendWhatsApp(sender, reply);
}

// Health check (also useful for an uptime pinger on Render's free tier)
app.get("/", (req, res) => res.status(200).send("ok"));

// Optional: inspect which model is currently in use
app.get("/model", async (req, res) => {
  try {
    const modelName = await resolveGeminiModel();
    res.status(200).json({ model: modelName, override: !!GEMINI_MODEL_OVERRIDE });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Webhook verification (Meta calls this once when you save the webhook)
app.get("/webhook", (req, res) => {
  const mode = req.query["hub.mode"];
  const token = req.query["hub.verify_token"];
  const challenge = req.query["hub.challenge"];

  if (mode === "subscribe" && token === VERIFY_TOKEN) {
    return res.status(200).send(challenge);
  }
  return res.sendStatus(403);
});

// Incoming messages
app.post("/webhook", (req, res) => {
  // Reply 200 immediately so Meta doesn't retry
  res.status(200).json({ status: "received" });

  try {
    for (const entry of req.body?.entry || []) {
      for (const change of entry.changes || []) {
        for (const msg of change.value?.messages || []) {
          if (seenIds.has(msg.id)) continue;
          seenIds.add(msg.id);
          if (seenIds.size > 500) seenIds.delete(seenIds.values().next().value);

          handleMessage(msg).catch((err) =>
            console.error("handleMessage failed:", err)
          );
        }
      }
    }
  } catch (err) {
    console.error("Error processing webhook:", err);
  }
});

app.listen(PORT, async () => {
  console.log(`Server listening on port ${PORT}`);
  try {
    const modelName = await resolveGeminiModel();
    console.log(`Using Gemini model: ${modelName}${GEMINI_MODEL_OVERRIDE ? " (pinned via GEMINI_MODEL)" : " (auto-resolved)"}`);
  } catch (err) {
    console.error("Could not resolve a Gemini model at startup:", err.message);
  }
});
