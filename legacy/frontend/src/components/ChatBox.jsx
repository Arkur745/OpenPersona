import React from "react";
import { useState } from "react";
import { useChatStore } from "../store/chat.store.js";
import { api } from "../services/api.js";
import { useSettingsStore } from "../store/settingsStore.js";
import { fetchAndPlay } from "../utils/tts.js";

const ChatBox = () => {
  const [input, setInput] = useState("");
  const {messages, addMessage} = useChatStore();

  const { ttsEnabled, toggleTTS } = useSettingsStore();

  const sendMessage = async () => {
    if (!input.trim()) return;

    try {
      const res = await api.post("/messages", {
        userId: "652...",
        convId: "653...",
        text: input,
      });

      for (const m of res.data.messages) {
        if (m.role === "assistant" && ttsEnabled) {
          // First, wait for TTS audio to play before showing message
          await fetchAndPlay(m.text);
          addMessage({ sender: "assistant", text: m.text });
        } else {
          // For user messages or when TTS is off
          addMessage({
            sender: m.role === "assistant" ? "assistant" : "user",
            text: m.text,
          });
        }
      }

      setInput("");
    } catch (error) {
      console.error("Message send error:", error);
    }
  };
  
  return (
    <div className="flex flex-col h-screen p-4">
      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto space-y-2">
        {messages.map((m, i) => (
          <div
            key={i}
            className={`p-2 rounded-xl max-w-[70%] ${
              m.sender === "user"
                ? "bg-blue-600 text-white ml-auto"
                : "bg-gray-300 text-black mr-auto"
            }`}
          >
            {m.text}
          </div>
        ))}
      </div>

      {/* Input + Buttons */}
      <div className="flex mt-3 items-center">
        {/* Zustand toggle button */}
        <button
          onClick={toggleTTS}
          className={`px-3 py-2 mr-2 rounded-xl transition-all ${
            ttsEnabled ? "bg-green-500 text-white" : "bg-gray-300 text-black"
          }`}
        >
          {ttsEnabled ? "Voice ON 🔊" : "Voice OFF 🔇"}
        </button>

        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          className="flex-1 border p-2 rounded-l-xl"
          placeholder="Type something..."
        />
        <button
          onClick={sendMessage}
          className="cursor-pointer bg-blue-600 text-white px-4 rounded-r-xl"
        >
          Send
        </button>
      </div>
    </div>
  );
};

export default ChatBox;
