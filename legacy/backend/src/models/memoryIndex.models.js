import mongoose from "mongoose";
const { Schema } = mongoose;

const memoryIndexSchema = new Schema({
  userId: {
    type: Schema.Types.Mixed,
    ref: "User",
    required: true,
    index: true,
  },
  convId: {
    type: mongoose.Schema.Types.Mixed,
    ref: "Conversation",
    default: null,
  },
  embeddingId: { type: String }, // Link to ChromaDB
  summary: { type: String },
  createdAt: { type: Date, default: Date.now },
  tags: [{ type: String }],
  isDeleted: { type: Boolean, default: false },
});

memoryIndexSchema.methods.softDelete = async function () {
  this.isDeleted = true;
  await this.save();
};

export const MemoryIndex = mongoose.model("MemoryIndex", memoryIndexSchema);
