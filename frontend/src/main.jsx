import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'
import { RunProvider } from './RunContext.jsx'
import './styles/global.css'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <RunProvider>
      <App />
    </RunProvider>
  </StrictMode>
)
