import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { connectDB } from './config/db.js';
import messageRoute from './routes/messageRoute.js'
import { PORT } from './config/env.js';
import ttsRoutes from "./routes/ttsRoutes.js";
const app = express();


console.log("Loaded PORT:", process.env.PORT);

app.use(cors());
app.use(express.json());


await connectDB();


app.use("/api/messages", messageRoute)

app.get('/',(req,res)=>{
    res.send('Hello world');
})
app.use("/api/tts", ttsRoutes);

app.listen(PORT,()=>{
    console.log(`Server started on port ${PORT}`);
})