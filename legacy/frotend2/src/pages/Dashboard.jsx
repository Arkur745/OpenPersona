import { Link } from 'react-router-dom';
import { UserButton } from '@clerk/clerk-react';
import { MessageSquare, Sparkles, TrendingUp, Clock } from 'lucide-react';
import { Button } from '@/components/ui/button';

const Dashboard = () => {
  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <MessageSquare className="w-6 h-6 text-primary" />
            <span className="text-xl font-bold">AI Companion</span>
          </div>
          
          <div className="flex items-center gap-4">
            <Link to="/">
              <Button variant="ghost">Home</Button>
            </Link>
            <Link to="/chat">
              <Button>Open Chat</Button>
            </Link>
            <UserButton />
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-12">
        <div className="mb-12">
          <h1 className="text-4xl font-bold mb-2">Welcome back!</h1>
          <p className="text-muted-foreground">Here's your activity overview</p>
        </div>

        {/* Stats Grid */}
        <div className="grid md:grid-cols-3 gap-6 mb-12">
          <div className="glass p-6 rounded-xl glow-hover">
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 rounded-lg bg-primary/20 flex items-center justify-center">
                <MessageSquare className="w-6 h-6 text-primary" />
              </div>
              <span className="text-2xl font-bold">24</span>
            </div>
            <h3 className="text-lg font-semibold mb-1">Total Messages</h3>
            <p className="text-sm text-muted-foreground">Conversations this week</p>
          </div>
          
          <div className="glass p-6 rounded-xl glow-hover">
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 rounded-lg bg-primary/20 flex items-center justify-center">
                <TrendingUp className="w-6 h-6 text-primary" />
              </div>
              <span className="text-2xl font-bold">92%</span>
            </div>
            <h3 className="text-lg font-semibold mb-1">Satisfaction</h3>
            <p className="text-sm text-muted-foreground">Response quality rating</p>
          </div>
          
          <div className="glass p-6 rounded-xl glow-hover">
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 rounded-lg bg-primary/20 flex items-center justify-center">
                <Clock className="w-6 h-6 text-primary" />
              </div>
              <span className="text-2xl font-bold">1.2s</span>
            </div>
            <h3 className="text-lg font-semibold mb-1">Avg Response</h3>
            <p className="text-sm text-muted-foreground">Average response time</p>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="glass p-8 rounded-xl">
          <h2 className="text-2xl font-bold mb-6">Quick Actions</h2>
          <div className="grid md:grid-cols-2 gap-4">
            <Link to="/chat">
              <Button className="w-full justify-start h-auto py-4 glow-hover" variant="outline">
                <MessageSquare className="w-5 h-5 mr-3" />
                <div className="text-left">
                  <div className="font-semibold">Start New Chat</div>
                  <div className="text-sm text-muted-foreground">Begin a fresh conversation</div>
                </div>
              </Button>
            </Link>
            
            <Button className="w-full justify-start h-auto py-4 glow-hover" variant="outline">
              <Sparkles className="w-5 h-5 mr-3" />
              <div className="text-left">
                <div className="font-semibold">Voice Settings</div>
                <div className="text-sm text-muted-foreground">Configure text-to-speech</div>
              </div>
            </Button>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Dashboard;
