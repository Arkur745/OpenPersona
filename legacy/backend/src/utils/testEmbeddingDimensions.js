import { generateEmbeddings } from "./gettingEmbeddings.js";

async function testEmbeddingDimensions() {
  try {
    console.log("🧪 Starting embedding tests...\n");
    console.log(
      "API Key status:",
      process.env.HUGGINGFACE_API_KEY ? "Present" : "Missing"
    );

    // Test 1: Single text input
    console.log("\n📝 Test 1: Single text input");
    const testText = "Hello, this is a test message";
    console.log("Input:", testText);

    console.log("\n⏳ Generating single embedding...");
    const singleEmbedding = await generateEmbeddings(testText);

    // Test 2: Multiple texts
    console.log("\n📝 Test 2: Multiple texts input");
    const testTexts = [
      "First test message",
      "Second test message",
      "Third test message",
    ];
    console.log("Inputs:", testTexts);

    console.log("\n⏳ Generating multiple embeddings...");
    const multipleEmbeddings = await generateEmbeddings(testTexts);

    // Basic structure validation
    console.log("\n🔍 Response Analysis:");
    console.log("Type:", typeof embedding);
    console.log("Is Array:", Array.isArray(embedding));
    if (Array.isArray(embedding)) {
      console.log("Length:", embedding.length);
      console.log("Sample (first 3):", embedding.slice(0, 3));
    } else {
      console.log("Raw value:", embedding);
    }

    // Detailed validation
    const checks = [
      {
        name: "Is Array",
        pass: Array.isArray(embedding),
      },
      {
        name: "Correct Length (384)",
        pass: Array.isArray(embedding) && embedding.length === 384,
      },
      {
        name: "All Numbers",
        pass:
          Array.isArray(embedding) &&
          embedding.every((v) => typeof v === "number" && !isNaN(v)),
      },
    ];

    console.log("\n✅ Validation Results:");
    checks.forEach(({ name, pass }) => {
      console.log(`${pass ? "✓" : "✗"} ${name}`);
    });

    const passed = checks.every((c) => c.pass);
    console.log(
      `\n${passed ? "🎉" : "❌"} Test ${passed ? "PASSED" : "FAILED"}`
    );
  } catch (error) {
    console.error("\n❌ Test failed:", error.message);
    console.error("\nDebug Information:");
    console.error("Error name:", error.name);
    console.error("Stack trace:", error.stack);
  }
}

testEmbeddingDimensions();
