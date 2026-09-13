import { embedText } from "./embedText.js";

async function testEmbedding() {
  try {
    console.log("Testing text embedding...");
    const testText = "Hello, world!";

    console.log("Input text:", testText);
    const embedding = await embedText(testText);

    console.log("Embedding successful:", {
      received: !!embedding,
      length: embedding.length,
      sample: embedding.slice(0, 3), // Show first 3 values
    });
  } catch (error) {
    console.error("Embedding test failed:", error);
    console.log("\nTroubleshooting steps:");
    console.log(
      "1. Check if HUGGINGFACE_API_KEY is set in .env.development.local"
    );
    console.log("2. Verify the API key is valid");
    console.log("3. Check network connectivity to Hugging Face API");
  }
}

testEmbedding();
