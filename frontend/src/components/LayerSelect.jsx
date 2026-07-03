import { truncateLayerName } from '../theme.js'

export default function LayerSelect({ label = 'Layer:', layers, value, onChange, id }) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <label htmlFor={id} className="text-xs font-medium text-muted">
        {label}
      </label>
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="max-w-[24rem] flex-1 rounded-md border border-border bg-background px-3 py-1.5 text-sm text-foreground focus:border-accent"
      >
        {layers.map((name) => (
          <option key={name} value={name} title={name}>
            {truncateLayerName(name, 60)}
          </option>
        ))}
      </select>
    </div>
  )
}
