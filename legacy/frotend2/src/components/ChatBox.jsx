import { useState, useRef, useEffect } from 'react';
import { useUser } from '@clerk/clerk-react';
import { Send, Volume2, VolumeX, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useChatStore } from '@/stores/chat.store';
import { useSettingsStore } from '@/stores/settings.store';
import { getConvId } from '@/utils/conv';
import { playTTS } from '@/utils/tts';
import api from '@/utils/api';

const ChatBox = () => {
  const { user } = useUser();
  const [inputText, setInputText] = useState('');
  const messagesEndRef = useRef(null);
  
  const { messages, addMessage, setMessages, isLoading, setLoading } = useChatStore();
  const { ttsEnabled, toggleTTS, voiceId } = useSettingsStore();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!inputText.trim() || !user || isLoading) return;

    const userMessage = inputText.trim();
    setInputText('');
    setLoading(true);

    try {
      const response = await api.post('/messages', {
        userId: user.id,
        convId: getConvId(),
        text: userMessage,
      });

      if (response.data.messages) {
        setMessages(response.data.messages);
        
        // Find the latest assistant message for TTS
        const assistantMessages = response.data.messages.filter(
          (msg) => msg.role === 'assistant'
        );
        const latestAssistant = assistantMessages[assistantMessages.length - 1];
        
        if (ttsEnabled && latestAssistant) {
          playTTS(latestAssistant.text, voiceId);
        }
      }
    } catch (error) {
      console.error('Error sending message:', error);
      addMessage({
        role: 'system',
        text: 'Error: Could not send message. Please try again.',
      });
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="h-full flex flex-col container mx-auto max-w-4xl">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center">
              <div className="w-16 h-16 rounded-full bg-primary/20 flex items-center justify-center mx-auto mb-4 glow">
                <Send className="w-8 h-8 text-primary" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Start a conversation</h3>
              <p className="text-muted-foreground">Send a message to begin chatting with your AI companion</p>
            </div>
          </div>
        ) : (
          messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                  message.role === 'user'
                    ? 'bg-primary text-primary-foreground'
                    : message.role === 'system'
                    ? 'bg-destructive/20 text-destructive-foreground'
                    : 'glass'
                }`}
              >
                <p className="whitespace-pre-wrap break-words">{message.text}</p>
              </div>
            </div>
          ))
        )}
        {isLoading && (
          <div className="flex justify-start">
            <div className="glass rounded-2xl px-4 py-3">
              <Loader2 className="w-5 h-5 animate-spin text-primary" />
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-border p-4 flex-shrink-0">
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="icon"
            onClick={toggleTTS}
            className={ttsEnabled ? 'glow' : ''}
          >
            {ttsEnabled ? (
              <Volume2 className="w-4 h-4" />
            ) : (
              <VolumeX className="w-4 h-4" />
            )}
          </Button>
          
          <Input
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message..."
            disabled={isLoading}
            className="flex-1"
          />
          
          <Button
            onClick={handleSend}
            disabled={!inputText.trim() || isLoading}
            className="glow-hover"
          >
            {isLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </Button>
        </div>
      </div>
    </div>
  );
};

export default ChatBox;
