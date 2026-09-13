import { Pinecone } from "@pinecone-database/pinecone";
import { PINECONE_API_KEY, PINECONE_INDEX_NAME } from "../config/env.js";

let pinecone;
let index;

const initializePinecone = async () => {
  if (!pinecone) {
    pinecone = new Pinecone({
      apiKey: PINECONE_API_KEY,
    });

    // Get the index
    index = pinecone.index(PINECONE_INDEX_NAME);

    console.log(`✅ Pinecone initialized with index: ${PINECONE_INDEX_NAME}`);
  }

  return index;
};

export default initializePinecone;
