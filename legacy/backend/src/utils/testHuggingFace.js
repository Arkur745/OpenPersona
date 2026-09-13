import { HUGGINGFACE_API_KEY } from "../config/env.js";
import { InferenceClient } from "@huggingface/inference";

async function testHuggingFace() {
  console.log("Testing Hugging Face configuration...\n");

  // 1. Check API key format
  console.log("API Key Check:");
  console.log("- Key exists:", !!HUGGINGFACE_API_KEY);
  console.log("- Key type:", typeof HUGGINGFACE_API_KEY);
  console.log("- Key length:", HUGGINGFACE_API_KEY?.length);
  console.log('- Starts with "hf_":', HUGGINGFACE_API_KEY?.startsWith("hf_"));

  // 2. Test client initialization
  try {
    console.log("\nTesting client initialization...");
    const client = new InferenceClient(HUGGINGFACE_API_KEY);

    // 3. Test a simple feature extraction
    console.log("Testing feature extraction...");
    const result = await client.featureExtraction({
      model: "sentence-transformers/all-MiniLM-L6-v2",
      inputs: "Hello world",
    });

    console.log("Feature extraction successful!");
    console.log("Result type:", typeof result);
    console.log("Result length:", result?.length);
  } catch (error) {
    console.error("\nError during test:", error);
    console.log("\nTroubleshooting tips:");
    console.log(
      "1. Check if HUGGINGFACE_API_KEY in .env.development.local is:"
    );
    console.log("   - Not wrapped in quotes");
    console.log("   - Starts with hf_");
    console.log("   - No extra spaces or newlines");
    console.log("2. Try re-copying the API key from Hugging Face website");
    console.log("3. Verify the API key is active in your Hugging Face account");
  }
}

testHuggingFace();
