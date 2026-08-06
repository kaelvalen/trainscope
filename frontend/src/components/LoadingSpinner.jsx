export default function LoadingSpinner({ message = 'Loading…' }) {
  return (
    <div
      className="flex flex-col items-center justify-center gap-3 rounded-xl border border-border bg-panel/60 p-12 text-muted"
      role="status"
      aria-live="polite"
    >
      <div className="h-6 w-6 animate-spin rounded-full border-2 border-border border-t-accent" />
      <span className="text-sm">{message}</span>
    </div>
  )
}
