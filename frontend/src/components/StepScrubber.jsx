import { useMemo } from 'react'

export default function StepScrubber({
  steps,
  value,
  onChange,
  label = 'Step',
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
    <div className="rounded-panel border border-border bg-surface px-4 py-3">
      <div className="flex items-center justify-between gap-3 text-sm">
        <label htmlFor={id} className="text-muted">
          {label}
        </label>
        <div className="flex items-center gap-3">
          <span className="text-xs text-muted">
            {min}–{max}
          </span>
          {showValue && <span className="font-medium">Step {value ?? min}</span>}
        </div>
      </div>
      <div className="mt-2">
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
          className="h-10 w-full cursor-pointer accent-[var(--ink)]"
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
