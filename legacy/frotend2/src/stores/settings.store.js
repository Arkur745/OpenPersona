import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export const useSettingsStore = create(
  persist(
    (set) => ({
      ttsEnabled: false,
      voiceId: null,
      
      toggleTTS: () => set((state) => ({ ttsEnabled: !state.ttsEnabled })),
      
      setVoiceId: (voiceId) => set({ voiceId }),
    }),
    {
      name: 'chat-settings',
    }
  )
);
