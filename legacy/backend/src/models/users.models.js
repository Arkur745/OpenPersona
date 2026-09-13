import mongoose, { Schema } from 'mongoose';
const {schema} = mongoose;

const userSchema = new Schema({
    email: {type: String, required: true, unique:true, index: true},
    name: {type: String},
    createdAt: {type: Date, default: Date.now},
    plan: {type: String, enum: ["free", "pro"], default: "free"},
    preferences_voice: {type: String},
    preferences_mode: {type: String},
    preferences_memory_depth: {type: Number, default: 5},
    isDeleted: {type: Boolean, default: false},  
});

userSchema.methods.softDelete = async function(){
    this.isDeleted = true;
    await this.save();

    const Conversation = mongoose.model("Conversation");
    const Message = mongoose.model("Message");
    const MemoryIndex = mongoose.model("MemoryIndex");
    const Billing = mongoose.model("Billing");

    await Promise.all([
      Conversation.updateMany({ userId: this._id }, { isDeleted: true }),
      Message.updateMany({ userId: this._id }, { isDeleted: true }),
      MemoryIndex.updateMany({ userId: this._id }, { isDeleted: true }),
      Billing.updateMany({ userId: this._id }, { isDeleted: true }),
    ]);
};

export const User = mongoose.model('User', userSchema);