import { create } from "zustand";

export const useSettingsStore = create((set) => ({
  ttsEnabled: false,
  toggleTTS: () => set((state) => ({ ttsEnabled: !state.ttsEnabled })),
}));
