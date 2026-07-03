import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'
import { RunProvider } from './RunContext.jsx'
import { ToastProvider } from './context/ToastContext.jsx'
import './styles/index.css'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ToastProvider>
      <RunProvider>
        <App />
      </RunProvider>
    </ToastProvider>
  </StrictMode>
)
