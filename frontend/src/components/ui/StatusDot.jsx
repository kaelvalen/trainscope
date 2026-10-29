const COLORS = {
  diverged: 'bg-red',
  spiked: 'bg-red',
  training: 'bg-blue',
  stable: 'bg-subtle',
  waiting: 'bg-subtle',
}

/** Small status dot; always shown next to a word, never alone. */
export function StatusDot({ kind }) {
  return (
    <span
      aria-hidden="true"
      className={`inline-block h-2 w-2 shrink-0 rounded-full ${COLORS[kind] ?? 'bg-subtle'}`}
    />
  )
}
