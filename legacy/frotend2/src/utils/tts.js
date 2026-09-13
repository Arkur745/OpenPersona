import api from './api';

export async function playTTS(text, voiceId = null) {
  try {
    const response = await api.post('/tts', {
      text,
      voice_id: voiceId,
    });
    
    if (response.data.success && response.data.audioUrl) {
      const audio = new Audio(response.data.audioUrl);
      audio.play();
      return true;
    }
    
    return false;
  } catch (error) {
    console.error('TTS Error:', error);
    return false;
  }
}
