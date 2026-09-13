import mongoose from "mongoose";
const { Schema } = mongoose;

const messageSchema = new Schema({
  // convId: {
  //   type: Schema.Types.ObjectId,
  //   ref: "Conversation",
  //   required: true,
  //   index: true,
  // },
  // userId: {
  //   type: Schema.Types.ObjectId,
  //   ref: "User",
  //   required: true,
  //   index: true,
  // },
  userId: { type: String, required: true }, // changed from ObjectId → String
  convId: { type: String, required: true },
  role: { type: String, enum: ["user", "assistant", "system"], required: true },
  text: { type: String },
  audioUrl: { type: String },
  embeddingId: { type: String }, // Link to ChromaDB
  sentiment_polarity: { type: Number },
  sentiment_label: { type: String },
  timestamp: { type: Date, default: Date.now },
  isDeleted: { type: Boolean, default: false },
});

messageSchema.methods.softDelete = async function () {
  this.isDeleted = true;
  await this.save();
};

export const Message = mongoose.model("Message", messageSchema);
