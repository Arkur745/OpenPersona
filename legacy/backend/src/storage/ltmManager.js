import initializePinecone from "../utils/pineConeClient.js";
import { generateEmbeddings } from "../utils/gettingEmbeddings.js";
import { MemoryIndex } from "../models/memoryIndex.models.js";
import { v4 as uuidv4 } from "uuid";

/**
 * Save a text as a conversation-scoped memory (Pinecone + Mongo index).
 * convId is required.
 */
export const saveToLTM = async ({ userId, convId, role, text }) => {
  if (!userId || !convId || !text) {
    throw new Error("saveToLTM requires userId, convId and text");
  }

  try {
    // Generate embeddings for the text
    const embedding = await generateEmbeddings(text);
    const embeddingId = uuidv4();

    // Initialize Pinecone
    const index = await initializePinecone();

    // Create metadata with namespace for conversation scoping
    const metadata = {
      userId: userId.toString(),
      convId: convId.toString(),
      role,
      text, // Store text in metadata for retrieval
      timestamp: new Date().toISOString(),
    };

    // Upsert vector to Pinecone (conversation-scoped via namespace)
    await index.namespace(convId).upsert([
      {
        id: embeddingId,
        values: embedding,
        metadata,
      },
    ]);

    // Create summary
    const summary = text.length > 200 ? text.substring(0, 200) + "..." : text;

    // Create an index record in Mongo
    await MemoryIndex.create({
      userId,
      convId,
      embeddingId,
      summary,
      tags: [],
      createdAt: new Date(),
      isDeleted: false,
    });

    console.log(
      "✅ LTM saved to Pinecone:",
      embeddingId,
      "namespace:",
      convId.substring(0, 20) + "..."
    );
    return embeddingId;
  } catch (err) {
    console.error("❌ LTM save error:", err);
    throw err;
  }
};

/**
 * Retrieve relevant memories from Pinecone using semantic search.
 * Returns array of document texts with proper similarity scoring.
 */
export const retrieveFromLTM = async (
  convId,
  query,
  nResults = 3,
  minSimilarity = 0.3
) => {
  if (!convId || !query) return [];

  try {
    // Generate embedding for the query
    const queryEmbedding = await generateEmbeddings(query);

    // Initialize Pinecone
    const index = await initializePinecone();

    console.log(`\n🔍 Querying Pinecone for: "${query.substring(0, 60)}..."`);

    // Query Pinecone in the conversation namespace
    const queryResponse = await index.namespace(convId).query({
      vector: queryEmbedding,
      topK: Math.max(nResults * 2, 10), // Fetch more for filtering
      includeMetadata: true,
    });

    if (!queryResponse.matches || queryResponse.matches.length === 0) {
      console.log("ℹ️ No memories found in Pinecone for this conversation");
      return [];
    }

    console.log(
      `📊 Pinecone returned ${queryResponse.matches.length} results\n`
    );

    // Log all similarity scores
    console.log("📈 Similarity Scores (highest to lowest):");
    queryResponse.matches.forEach((match, i) => {
      const text = match.metadata?.text || "(no text)";
      const role = match.metadata?.role || "unknown";
      console.log(
        `  ${i + 1}. [Score: ${match.score.toFixed(
          4
        )}] [${role}] ${text.substring(0, 80)}...`
      );
    });

    console.log(`\n🎯 Filtering with minimum similarity: ${minSimilarity}\n`);

    // Filter and deduplicate
    const seenTexts = new Set();
    const uniqueMemories = [];

    for (const match of queryResponse.matches) {
      const score = match.score;
      const text = match.metadata?.text;

      if (!text) continue;

      // Skip if below threshold
      if (score < minSimilarity) {
        console.log(
          `  ⛔ Skipped (score ${score.toFixed(
            4
          )} < ${minSimilarity}): ${text.substring(0, 50)}...`
        );
        continue;
      }

      // Skip duplicates
      if (seenTexts.has(text)) {
        console.log(`  ⛔ Skipped (duplicate): ${text.substring(0, 50)}...`);
        continue;
      }

      seenTexts.add(text);
      uniqueMemories.push(text);
      console.log(
        `  ✅ Selected (score ${score.toFixed(4)}): ${text.substring(0, 50)}...`
      );

      if (uniqueMemories.length >= nResults) break;
    }

    // Final summary
    console.log(
      `\n✅ Retrieved ${uniqueMemories.length}/${nResults} memories from Pinecone`
    );
    if (uniqueMemories.length > 0) {
      console.log("📋 Final Selected Memories:");
      queryResponse.matches
        .slice(0, uniqueMemories.length)
        .forEach((match, i) => {
          console.log(
            `  ${i + 1}. [${match.score.toFixed(
              4
            )}] ${match.metadata.text.substring(0, 80)}...`
          );
        });
    } else {
      console.log("⚠️ No memories met the similarity threshold!");
      if (queryResponse.matches.length > 0) {
        console.log(
          `💡 Highest score found: ${queryResponse.matches[0].score.toFixed(4)}`
        );
        console.log(`💡 Try lowering minSimilarity below ${minSimilarity}`);
      }
    }
    console.log("\n" + "=".repeat(80) + "\n");

    return uniqueMemories;
  } catch (err) {
    console.error("❌ LTM Retrieval Error:", err);
    return [];
  }
};
