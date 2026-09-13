import { generateEmbeddings } from "./gettingEmbeddings.js";

async function testPineconeFormat() {
  try {
    console.log("🧪 Testing Pinecone-compatible embeddings...\n");

    // Test 1: Single text (for upsert)
    const singleText = "This is a test message for Pinecone";
    console.log("📝 Test 1: Single text input");
    const singleEmbedding = await generateEmbeddings(singleText);

    console.log("\nSingle Embedding Analysis:");
    console.log("- Dimensions:", singleEmbedding.length);
    console.log("- Sample (first 3):", singleEmbedding.slice(0, 3));
    console.log(
      "- All numbers:",
      singleEmbedding.every((v) => typeof v === "number")
    );

    // Test 2: Multiple texts (for batch operations)
    const multipleTexts = [
      "First test message",
      "Second test message",
      "Third test message",
    ];

    console.log("\n📝 Test 2: Multiple texts input");
    const multipleEmbeddings = await generateEmbeddings(multipleTexts);

    console.log("\nMultiple Embeddings Analysis:");
    console.log("- Number of embeddings:", multipleEmbeddings.length);
    multipleEmbeddings.forEach((embedding, idx) => {
      console.log(`\nEmbedding ${idx + 1}:`);
      console.log("- Dimensions:", embedding.length);
      console.log("- Sample:", embedding.slice(0, 3));
      console.log(
        "- All numbers:",
        embedding.every((v) => typeof v === "number")
      );
    });

    // Simulate Pinecone upsert format
    const pineconeFormat = {
      vectors: [
        {
          id: "test-1",
          values: singleEmbedding,
          metadata: {
            text: singleText,
          },
        },
      ],
    };

    console.log("\n📊 Pinecone Upsert Format Example:");
    console.log(JSON.stringify(pineconeFormat, null, 2));

    console.log("\n✅ Tests completed successfully");
  } catch (error) {
    console.error("\n❌ Test failed:", error.message);
    console.error("\nDebug information:");
    console.error("Error name:", error.name);
    console.error("Stack trace:", error.stack);
  }
}

// Run the test
console.log(
  "🔑 API Key status:",
  process.env.HUGGINGFACE_API_KEY ? "Present" : "Missing"
);
testPineconeFormat();
