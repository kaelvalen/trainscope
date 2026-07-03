import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import { cn } from '../utils.js'

const ToastContext = createContext(null)

const variantStyles = {
  info: 'border-accent/30 bg-panel',
  success: 'border-success/30 bg-panel',
  warning: 'border-warning/30 bg-panel',
  danger: 'border-danger/30 bg-panel',
}

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([])

  const addToast = useCallback((toast) => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`
    setToasts((prev) => [...prev, { ...toast, id }])
  }, [])

  const dismissToast = useCallback((id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id))
  }, [])

  return (
    <ToastContext.Provider value={{ addToast, dismissToast }}>
      <ToastContainer toasts={toasts} onDismiss={dismissToast} />
      {children}
    </ToastContext.Provider>
  )
}

export function useToast() {
  const ctx = useContext(ToastContext)
  if (!ctx) {
    throw new Error('useToast must be used within a ToastProvider')
  }
  return ctx
}

function ToastContainer({ toasts, onDismiss }) {
  return (
    <div
      className="pointer-events-none fixed right-4 top-4 z-[100] flex w-80 max-w-[calc(100vw-2rem)] flex-col gap-2"
      aria-live="polite"
      aria-atomic="true"
    >
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onDismiss={onDismiss} />
      ))}
    </div>
  )
}

function ToastItem({ toast, onDismiss }) {
  useEffect(() => {
    const timer = setTimeout(() => onDismiss(toast.id), toast.duration || 5000)
    return () => clearTimeout(timer)
  }, [toast.id, toast.duration, onDismiss])

  return (
    <div
      className={cn(
        'pointer-events-auto flex flex-col gap-1 rounded-lg border p-3 shadow-lg',
        'transition-all duration-300 ease-out',
        variantStyles[toast.variant || 'info']
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <span className="text-sm font-semibold text-foreground">{toast.title}</span>
        <button
          onClick={() => onDismiss(toast.id)}
          className="text-muted transition-colors hover:text-foreground"
          aria-label="Dismiss notification"
        >
          ×
        </button>
      </div>
      {toast.message && <p className="text-xs text-muted">{toast.message}</p>}
    </div>
  )
}
