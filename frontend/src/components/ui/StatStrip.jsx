/** One segmented row of numbers: label above, value below, hairlines between. */
export function StatStrip({ items }) {
  return (
    <dl className="grid grid-cols-2 overflow-hidden rounded-panel border border-border bg-surface md:grid-cols-4">
      {items.map((item, i) => (
        <div
          key={item.label}
          className={`px-4 py-3 ${i % 2 ? 'border-l' : ''} ${i >= 2 ? 'border-t md:border-t-0' : ''} ${i === 2 ? 'md:border-l' : ''} border-border`}
        >
          <dt className="text-xs text-muted">{item.label}</dt>
          <dd className="mt-0.5 text-xl font-medium tracking-tight text-ink">{item.value}</dd>
        </div>
      ))}
    </dl>
  )
}
