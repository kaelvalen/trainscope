import { truncateLayerName } from '../theme.js'

export default function LayerSelect({ label = 'Layer:', layers, value, onChange, id }) {
  return (
    <div className="flex min-w-0 flex-col gap-2 sm:flex-row sm:items-center sm:gap-3">
      <label htmlFor={id} className="control-label shrink-0">
        {label}
      </label>
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="control-select min-w-0 max-w-full flex-1 sm:max-w-[32rem]"
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
