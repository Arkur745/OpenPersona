import fs from "fs";
import { ElevenLabsClient } from "@elevenlabs/elevenlabs-js";
import cloudinary from "../utils/cloudinary.js";
import { ELEVENLABS_API_KEY } from "../config/env.js";

const elevenlabs = new ElevenLabsClient({
    apiKey: ELEVENLABS_API_KEY
})

export const generateTTS = async (req,res)=>{
    try {
        const { text, voice_id } = req.body;
        if(!text) res.status(400).json({ error: "Text is required "});

        const voiceId = voice_id || "cgSgspJ2msm6clMCkdW9";

        const audioStream = await elevenlabs.textToSpeech.convert(voiceId, {
          text,
          modelId: "eleven_multilingual_v2",
          outputFormat: "mp3_44100_128",
        });

        const tempPath = "./output.mp3"
        const chunks = [];

        for await (const chunk of audioStream) {
            chunks.push(chunk);
        }
        const buffer = Buffer.concat(chunks);
        fs.writeFileSync(tempPath, buffer);

        const uploadRes = await cloudinary.uploader.upload(tempPath,{
            resource_type: "video",
            folder: "tts_audio"
        })

        fs.unlinkSync(tempPath)

        return res.json({
            success: true,
            audioUrl: uploadRes.secure_url,
        })
    } catch (error) {
        console.error("TTS Error:", error);
        return res.status(500).json({ error: "Failed to generate TTS" });
    }
}