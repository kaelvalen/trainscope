import { useMemo } from 'react'

export default function StepScrubber({
  steps,
  value,
  onChange,
  label = 'Step scrubber:',
  showValue = true,
  id = 'step-scrubber',
}) {
  const stepSet = useMemo(() => new Set(steps || []), [steps])

  if (!steps || steps.length === 0) return null

  const min = steps[0]
  const max = steps[steps.length - 1]

  function handleChange(e) {
    const target = Number(e.target.value)
    if (stepSet.has(target)) {
      onChange(target)
      return
    }
    const closest = steps.reduce((prev, curr) =>
      Math.abs(curr - target) < Math.abs(prev - target) ? curr : prev
    )
    onChange(closest)
  }

  return (
    <div className="scrubber">
      <div className="scrubber__meta">
        <label htmlFor={id} className="scrubber__label">
          {label}
        </label>
        <div className="flex items-center gap-3">
          <span className="scrubber__range">
            {min} → {max}
          </span>
          {showValue && <span className="scrubber__value">Step {value ?? min}</span>}
        </div>
      </div>
      <div className="scrubber__track">
        <input
          id={id}
          type="range"
          min={min}
          max={max}
          step="1"
          value={value ?? min}
          onChange={handleChange}
          list={`${id}-ticks`}
          aria-label={label}
        />
        <datalist id={`${id}-ticks`}>
          {steps.map((s) => (
            <option key={s} value={s} />
          ))}
        </datalist>
      </div>
    </div>
  )
}
