// src/utils/gettingEmbeddings.js
import { InferenceClient } from "@huggingface/inference";
import { HUGGINGFACE_API_KEY } from "../config/env.js";

// Pass API key as string directly
const client = new InferenceClient(HUGGINGFACE_API_KEY);

export async function generateEmbeddings(text) {
  try {
    // Call feature extraction endpoint
    const result = await client.featureExtraction({
      model: "sentence-transformers/all-MiniLM-L6-v2",
      inputs: text,
    });

    // Some models return nested arrays → flatten if needed
    const embedding = Array.isArray(result[0]) ? result[0] : result;

    return embedding;
  } catch (err) {
    console.error("Embedding generation error:", err);
    throw err;
  }
}
