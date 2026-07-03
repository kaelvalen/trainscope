import { Button } from './ui/Button.jsx'

export default function ErrorMessage({ message, onRetry }) {
  if (!message) return null

  return (
    <div
      className="mb-4 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-danger/30 bg-danger-bg/20 p-4 text-sm text-danger"
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
