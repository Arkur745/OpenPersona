import axios from "axios";

export const fetchAndPlay = async (text) => {
    try {
        const response = await axios.post(
            "http://localhost:5000/api/tts/generate",
            {text}
        );
        const {audioUrl} = response.data;
        if(!audioUrl) throw new Error("No audio URL returned from TTS")
        const audio = new Audio(audioUrl);
    await audio.play()
    } catch(error){
        console.error("TTS Playback error:" , error.response?.data || error.message)
    }
}