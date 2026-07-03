export default function EmptyState({ children, icon = '—' }) {
  return (
    <div
      className="rounded-lg border border-dashed border-border bg-panel p-10 text-center text-sm text-muted"
      role="status"
    >
      <div className="mb-2 text-2xl">{icon}</div>
      <div>{children}</div>
    </div>
  )
}
