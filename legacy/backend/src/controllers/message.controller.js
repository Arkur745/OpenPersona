import mongoose from "mongoose";
import { generateResponse } from "../utils/gemini.js";
import { getSTMContext, addToSTM, getRecentContext } from "../storage/stm.js";
import { saveToLTM, retrieveFromLTM } from "../storage/ltmManager.js"; // 👈 added
import { Message } from "../models/messages.model.js";
import { Conversation } from "../models/conversation.models.js";

export const sendMessage = async (req, res) => {
  try {
    const { userId, convId, text } = req.body;
    if (!userId || !convId || !text)
      return res.status(400).json({ error: "Missing required fields" });

    // 🧠 1. Ensure STM context loaded
    await getSTMContext(convId);

    // 🧠 2. Get short-term memory (last few messages)
    const stmContext = getRecentContext(convId, 5);

    // 🧩 3. Retrieve relevant long-term memories (semantic context)
    const ltmMemories = await retrieveFromLTM(convId, text, 3);

    // Combine STM + LTM as unified context
    const fullContext = [
      ...stmContext,
      ...ltmMemories.map((mem) => ({
        role: "memory",
        text: mem,
      })),
    ];

    // 💬 4. Save user message in STM + LTM
    await addToSTM(convId, userId, "user", text);
    await saveToLTM({ userId, convId, role: "user", text });

    // 🤖 5. Generate response with both STM & LTM context
    const replyText = await generateResponse(text, fullContext);

    // 💾 6. Save AI reply in both STM + LTM
    await addToSTM(convId, userId, "assistant", replyText);
    await saveToLTM({ userId, convId, role: "assistant", text: replyText });

    // 📤 7. Send messages to frontend
    res.status(200).json({
      messages: [
        { role: "user", text },
        { role: "assistant", text: replyText },
      ],
    });
  } catch (error) {
    console.error("❌ Controller error:", error);
    res.status(500).json({ error: "Internal server error" });
  }
};
