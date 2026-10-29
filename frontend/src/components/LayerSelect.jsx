import { useMemo, useState } from 'react'
import { truncateLayerName } from '../theme.js'

export default function LayerSelect({ label = 'Layer', layers, value, onChange, id }) {
  const [query, setQuery] = useState('')
  const visibleLayers = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase()
    if (!normalizedQuery) return layers

    const matches = layers.filter((name) => name.toLowerCase().includes(normalizedQuery))
    if (value && !matches.includes(value)) return [value, ...matches]
    return matches
  }, [layers, query, value])

  return (
    <div className="flex flex-wrap items-center gap-2">
      <label htmlFor={id} className="text-sm text-muted">
        {label}
      </label>
      <div className="flex flex-wrap items-center gap-2">
        <div>
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter layers"
            aria-label="Filter layers"
            className="h-8 w-44 rounded-control border border-border bg-surface px-2.5 text-sm placeholder:text-subtle"
          />
        </div>
        <select
          id={id}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="h-8 max-w-[22rem] rounded-control border border-border bg-surface px-2 text-sm"
        >
          {visibleLayers.length > 0 ? (
            visibleLayers.map((name) => (
              <option key={name} value={name} title={name}>
                {truncateLayerName(name, 60)}
              </option>
            ))
          ) : (
            <option value="" disabled>
              No matching layers
            </option>
          )}
        </select>
      </div>
    </div>
  )
}
