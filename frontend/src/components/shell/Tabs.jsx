/** Underlined tab row. `items` are the visible views; the active one gets an ink underline. */
export function Tabs({ items, active, onSelect }) {
  return (
    <nav aria-label="Views" className="border-b border-border bg-surface">
      <div className="mx-auto flex max-w-[1360px] gap-5 overflow-x-auto px-4 sm:px-8">
        {items.map((item) => {
          const isActive = item.id === active
          return (
            <a
              key={item.id}
              href={`#${item.id}`}
              aria-current={isActive ? 'page' : undefined}
              onClick={(event) => {
                event.preventDefault()
                onSelect(item.id)
              }}
              className={`-mb-px flex h-11 shrink-0 items-center border-b-2 text-sm transition-colors duration-150 ${
                isActive ? 'border-ink text-ink' : 'border-transparent text-muted hover:text-ink'
              }`}
            >
              {item.label}
            </a>
          )
        })}
      </div>
    </nav>
  )
}
