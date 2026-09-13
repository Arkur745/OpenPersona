import { Link } from 'react-router-dom';
import { SignedIn, SignedOut, UserButton } from '@clerk/clerk-react';
import { MessageSquare, Sparkles, Zap } from 'lucide-react';
import { Button } from '@/components/ui/button';

const Home = () => {
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
            <SignedOut>
              <Link to="/sign-in">
                <Button variant="ghost">Sign In</Button>
              </Link>
              <Link to="/sign-up">
                <Button>Get Started</Button>
              </Link>
            </SignedOut>
            <SignedIn>
              <Link to="/dashboard">
                <Button>Dashboard</Button>
              </Link>
              <UserButton />
            </SignedIn>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <main className="container mx-auto px-4 py-20">
        <div className="text-center max-w-4xl mx-auto">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass glow mb-8">
            <Sparkles className="w-4 h-4 text-primary" />
            <span className="text-sm">Your Personal AI Assistant</span>
          </div>
          
          <h1 className="text-6xl font-bold mb-6 bg-gradient-to-r from-foreground to-primary bg-clip-text text-transparent">
            Chat with Intelligence
          </h1>
          
          <p className="text-xl text-muted-foreground mb-12 max-w-2xl mx-auto">
            Experience the future of conversation with our advanced AI companion. 
            Get instant responses, voice synthesis, and personalized interactions.
          </p>
          
          <SignedOut>
            <div className="flex gap-4 justify-center">
              <Link to="/sign-up">
                <Button size="lg" className="glow-hover">
                  Start Chatting
                  <Zap className="ml-2 w-4 h-4" />
                </Button>
              </Link>
              <Link to="/sign-in">
                <Button size="lg" variant="outline" className="glow-hover">
                  Sign In
                </Button>
              </Link>
            </div>
          </SignedOut>
          
          <SignedIn>
            <Link to="/dashboard">
              <Button size="lg" className="glow-hover">
                Go to Dashboard
                <Zap className="ml-2 w-4 h-4" />
              </Button>
            </Link>
          </SignedIn>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-3 gap-8 mt-24">
          <div className="glass p-6 rounded-xl glow-hover">
            <div className="w-12 h-12 rounded-lg bg-primary/20 flex items-center justify-center mb-4">
              <MessageSquare className="w-6 h-6 text-primary" />
            </div>
            <h3 className="text-xl font-semibold mb-2">Smart Conversations</h3>
            <p className="text-muted-foreground">
              Engage in natural, context-aware conversations that understand your needs.
            </p>
          </div>
          
          <div className="glass p-6 rounded-xl glow-hover">
            <div className="w-12 h-12 rounded-lg bg-primary/20 flex items-center justify-center mb-4">
              <Sparkles className="w-6 h-6 text-primary" />
            </div>
            <h3 className="text-xl font-semibold mb-2">Voice Synthesis</h3>
            <p className="text-muted-foreground">
              Listen to responses with high-quality text-to-speech technology.
            </p>
          </div>
          
          <div className="glass p-6 rounded-xl glow-hover">
            <div className="w-12 h-12 rounded-lg bg-primary/20 flex items-center justify-center mb-4">
              <Zap className="w-6 h-6 text-primary" />
            </div>
            <h3 className="text-xl font-semibold mb-2">Lightning Fast</h3>
            <p className="text-muted-foreground">
              Get instant responses with our optimized AI infrastructure.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Home;
