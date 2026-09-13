import express from "express";
import { generateTTS } from "../controllers/ttsController.js";

const router = express.Router();
router.post("/generate", generateTTS);
export default router;
