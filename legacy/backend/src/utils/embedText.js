import { InferenceClient } from "@huggingface/inference";
import { HUGGINGFACE_API_KEY } from "../config/env.js";

// Debug logging for API key
console.log("HF API Key available:", !!HUGGINGFACE_API_KEY);

const hfClient = new InferenceClient(HUGGINGFACE_API_KEY);

// Verify client initialization
console.log("HF Client initialized:", !!hfClient);

export const embedText = async (text) => {
  try {
    const response = await hfClient.featureExtraction({
      model: "sentence-transformers/all-MiniLM-L6-v2",
      inputs: text, // Fixed typo: 'imputs' -> 'inputs'
    });
    const raw = response[0];
    const embedding = Array.isArray(raw) ? raw : Object.values(raw).flat();
    return embedding;
  } catch (error) {
    console.error("Embedding Error: ", error);
    return [];
  }
};
