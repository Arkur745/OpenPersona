import { config } from "dotenv";

config({ path: `.env.${process.env.NODE_ENV || "development"}.local` });

// Clean up environment variables, ensuring no whitespace
const cleanEnv = Object.fromEntries(
  Object.entries(process.env).map(([key, value]) => [key, value?.trim()])
);

export const {
  PORT,
  MONGO_URI,
  NODE_ENV,
  GEMINI_API_KEY,
  CLOUDINARY_CLOUD_NAME,
  CLOUDINARY_API_KEY,
  CLOUDINARY_API_SECRET,
  CLOUDINARY_URL,
  ELEVENLABS_API_KEY,
  HUGGINGFACE_API_KEY,
  PINECONE_API_KEY,
  PINECONE_INDEX_NAME,
} = cleanEnv;
