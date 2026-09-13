// src/memory/stmManager.js
import { Message } from "../models/messages.model.js";

const stmCache = new Map();
const MAX_STM_MESSAGES = 10;

/**
 * 🧠 Get STM context for a given conversation.
 * Cache-first; lazy-loads from MongoDB if cache empty.
 */
export const getSTMContext = async (convId) => {
  if (!convId) throw new Error("convId required");

  let stmMessages = stmCache.get(convId);

  // Cache miss → fetch last 5 messages from DB
  if (!stmMessages) {
    const dbMessages = await Message.find({ convId })
      .sort({ timestamp: -1 })
      .limit(5)
      .lean();

    if (dbMessages.length > 0) {
      stmMessages = dbMessages.reverse().map((m) => ({
        role: m.role,
        text: m.text,
      }));
      stmCache.set(convId, stmMessages);
      console.log(`🧠 STM cache initialized from DB: ${stmMessages.length} msgs`);
    } else {
      stmMessages = [];
      stmCache.set(convId, stmMessages);
      console.log(`🧩 New STM context created for convId: ${convId}`);
    }
  }

  return stmMessages;
};

/**
 * 💬 Add a message (user or assistant) to STM cache & optionally to DB.
 */
export const addToSTM = async (convId, userId, role, text, saveToDB = true) => {
  if (!convId || !userId || !text) throw new Error("Missing fields");
  if (!["user", "assistant"].includes(role))
    throw new Error(`Invalid role: ${role}`);

  if (!stmCache.has(convId)) stmCache.set(convId, []);

  const stmMessages = stmCache.get(convId);
  stmMessages.push({ role, text });

  if (stmMessages.length > MAX_STM_MESSAGES) stmMessages.shift();
  stmCache.set(convId, stmMessages);

  if (saveToDB) {
    await Message.create({
      userId,
      convId,
      role,
      text,
      timestamp: new Date(),
    }).catch((err) => console.error("⚠️ Mongo save error:", err));
  }
};

/**
 * 🧩 Get recent context messages for conversation.
 */
export const getRecentContext = (convId, limit = 5) => {
  const stmMessages = stmCache.get(convId) || [];
  return stmMessages.slice(-limit);
};

/**
 * 🔍 Check if STM is too shallow, signaling we should query LTM.
 */
export const needsLTM = (convId, threshold = 3) => {
  const stmMessages = stmCache.get(convId) || [];
  return stmMessages.length < threshold;
};

/**
 * 🧹 Clear STM for a conversation (on delete/end).
 */
export const clearSTM = (convId) => {
  stmCache.delete(convId);
  console.log(`🧹 Cleared STM cache for convId: ${convId}`);
};
