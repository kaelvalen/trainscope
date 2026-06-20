import { truncateLayerName } from '../theme.js'

export default function LayerSelect({ label = 'Layer:', layers, value, onChange, id }) {
  return (
    <div className="ts-control-row" style={{ marginBottom: 0 }}>
      <label htmlFor={id} className="ts-label">
        {label}
      </label>
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="ts-select ts-layer-select"
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
