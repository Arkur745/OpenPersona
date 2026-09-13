import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'
import ChatBox from './components/ChatBox.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ChatBox/>
  </StrictMode>,
)
