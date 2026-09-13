import mongoose from "mongoose";
const { Schema } = mongoose;

const conversationSchema = new Schema({
  userId: {
    type: Schema.Types.ObjectId,
    ref: "User",
    required: true,
    index: true,
  },
  startedAt: { type: Date, default: Date.now },
  lastActiveAt: { type: Date, default: Date.now },
  title: { type: String },
  isActive: { type: Boolean, default: true },
  isDeleted: { type: Boolean, default: false },
});

conversationSchema.methods.softDelete = async function() {
    this.isDeleted = true;
    this.isActive = false;
    await this.isActive();

    const Message = mongoose.model("Message");
    await Message.updateMany({ convId: this._id }, { isDeleted:true });

};

export const Conversation = mongoose.model("Conversation",conversationSchema);