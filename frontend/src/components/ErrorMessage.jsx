import { Button } from './ui/Button.jsx'

export default function ErrorMessage({ message, onRetry }) {
  if (!message) return null

  return (
    <div
      className="error-message mb-5 flex flex-wrap items-center justify-between gap-3 p-4 text-sm"
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
