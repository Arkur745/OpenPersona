import mongoose from "mongoose";
import { MONGO_URI } from "./env.js";

export const connectDB = async () => {
    try{
        await mongoose.connect(MONGO_URI);
        console.log('MongoDB connected');
    } catch (error){
        console.error('DB Error:',error);
        process.exit(1);
    }
}