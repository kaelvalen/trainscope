import { useMemo, useState } from 'react'
import { Search } from 'lucide-react'
import { truncateLayerName } from '../theme.js'

export default function LayerSelect({ label = 'Layer:', layers, value, onChange, id }) {
  const [query, setQuery] = useState('')
  const visibleLayers = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase()
    if (!normalizedQuery) return layers

    const matches = layers.filter((name) => name.toLowerCase().includes(normalizedQuery))
    if (value && !matches.includes(value)) return [value, ...matches]
    return matches
  }, [layers, query, value])

  return (
    <div className="layer-select">
      <label htmlFor={id} className="control-label shrink-0">
        {label}
      </label>
      <div className="layer-select__controls">
        <div className="layer-filter">
          <Search className="layer-filter__icon" aria-hidden="true" />
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter layers"
            aria-label="Filter layers"
            className="control-input"
          />
        </div>
        <select
          id={id}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="control-select layer-select__select"
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
