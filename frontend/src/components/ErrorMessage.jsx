import { Button } from './ui/Button.jsx'

export default function ErrorMessage({ message, onRetry }) {
  if (!message) return null

  return (
    <div
      className="flex flex-wrap items-center justify-between gap-3 rounded-panel border border-border bg-surface px-4 py-3 text-sm text-red"
      role="alert"
    >
      <span>{message}</span>
      {onRetry && (
        <Button variant="danger" size="sm" onClick={onRetry}>
          Retry
        </Button>
      )}
    </div>
  )
}
