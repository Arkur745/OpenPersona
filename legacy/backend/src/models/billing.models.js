import mongoose from "mongoose";
const { Schema } = mongoose;

const billingSchema = new Schema({
  userId: {
    type: Schema.Types.ObjectId,
    ref: "User",
    required: true,
    index: true,
  },
  plan: { type: String, enum: ["free", "pro"], required: true },
  status: {
    type: String,
    enum: ["active", "expired", "cancelled"],
    default: "active",
  },
  startDate: { type: Date, default: Date.now },
  endDate: { type: Date },
  isDeleted: { type: Boolean, default: false },
});

billingSchema.methods.softDelete = async function () {
  this.isDeleted = true;
  await this.save();
};

export const Billing = mongoose.model("Billing", billingSchema);
