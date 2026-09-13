import { Link } from 'react-router-dom';
import { UserButton } from '@clerk/clerk-react';
import { MessageSquare, Home } from 'lucide-react';
import { Button } from '@/components/ui/button';
import ChatBox from '@/components/ChatBox';

const Chat = () => {
  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Header */}
      <header className="border-b border-border flex-shrink-0">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <MessageSquare className="w-6 h-6 text-primary" />
            <span className="text-xl font-bold">AI Companion</span>
          </div>
          
          <div className="flex items-center gap-4">
            <Link to="/dashboard">
              <Button variant="ghost" size="sm">
                <Home className="w-4 h-4 mr-2" />
                Dashboard
              </Button>
            </Link>
            <UserButton />
          </div>
        </div>
      </header>

      {/* Chat Container */}
      <main className="flex-1 overflow-hidden">
        <ChatBox />
      </main>
    </div>
  );
};

export default Chat;
