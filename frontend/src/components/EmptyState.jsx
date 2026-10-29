export default function EmptyState({ children }) {
  return (
    <div role="status" className="rounded-panel border border-border bg-surface px-6 py-10 text-sm">
      <div className="max-w-md text-muted">{children}</div>
    </div>
  )
}
