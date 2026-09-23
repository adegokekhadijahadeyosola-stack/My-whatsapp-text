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

const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-3.6-flash";
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

const GEMINI_URL = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent`;
const WA_URL = `https://graph.facebook.com/${GRAPH_VERSION}/${PHONE_NUMBER_ID}/messages`;

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

async function askGemini(sender, text) {
  remember(sender, "user", text);

  const res = await fetch(GEMINI_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-goog-api-key": GEMINI_API_KEY,
    },
    body: JSON.stringify({
      system_instruction: { parts: [{ text: SYSTEM_PROMPT }] },
      contents: history.get(sender),
      generationConfig: { maxOutputTokens: 200 }, // caps reply length -> faster generation
    }),
  });

  if (res.status === 429) {
    history.get(sender).pop();
    return "I'm getting too many requests right now. Please try again in a minute.";
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

  await showTypingIndicator(msg.id);

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

app.listen(PORT, () => console.log(`Server listening on port ${PORT}`));
