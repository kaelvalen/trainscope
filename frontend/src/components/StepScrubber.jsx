import { useMemo } from 'react'
import { COLORS } from '../theme.js'

export default function StepScrubber({
  steps,
  value,
  onChange,
  label = 'Step scrubber:',
  showValue = true,
  id = 'step-scrubber',
}) {
  const stepSet = useMemo(() => new Set(steps), [steps])

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
    <div className="ts-control-row">
      <label htmlFor={id} className="ts-label">
        {label}
      </label>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={steps.length > 1 ? steps[1] - steps[0] : 1}
        value={value ?? min}
        onChange={handleChange}
        className="ts-slider"
        style={{ flex: 1 }}
        list={`${id}-ticks`}
      />
      <datalist id={`${id}-ticks`}>
        {steps.map((s) => (
          <option key={s} value={s} />
        ))}
      </datalist>
      {showValue && (
        <span
          style={{
            fontSize: '12px',
            color: COLORS.muted,
            minWidth: '60px',
            textAlign: 'right',
          }}
        >
          Step {value ?? min}
        </span>
      )}
    </div>
  )
}
