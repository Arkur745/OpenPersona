import { create } from "zustand";

export const useChatStore = create((set) =>({
    messages: [],
    addMessage: (msg) => set((state) => ({ messages: [...state.messages, msg]})),
    clearChat: () => set({messages: []})
}))