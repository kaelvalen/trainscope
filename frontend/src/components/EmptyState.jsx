export default function EmptyState({ children, icon = '—' }) {
  return (
    <div className="empty-state" role="status">
      <div className="empty-state__icon" aria-hidden="true">
        {icon}
      </div>
      <div className="max-w-md text-sm leading-relaxed">{children}</div>
    </div>
  )
}
