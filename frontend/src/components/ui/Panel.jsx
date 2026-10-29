import { cn } from '../../utils.js'

/** A surface with a hairline border and an optional header row (title left, actions right). */
export function Panel({ title, actions, className, bodyClassName, children }) {
  return (
    <section className={cn('rounded-panel border border-border bg-surface', className)}>
      {(title || actions) && (
        <header className="flex min-h-[44px] items-center justify-between gap-3 border-b border-border px-4 py-2">
          {title && <h2 className="text-sm font-medium text-ink">{title}</h2>}
          {actions && <div className="flex items-center gap-3 text-xs text-muted">{actions}</div>}
        </header>
      )}
      <div className={bodyClassName ?? 'p-4'}>{children}</div>
    </section>
  )
}
