import { useMemo } from 'react'

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
    <div className="flex flex-wrap items-center gap-3">
      <label htmlFor={id} className="text-xs font-medium text-muted">
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
        className="h-1 flex-1 cursor-pointer appearance-none rounded bg-border accent-accent"
        list={`${id}-ticks`}
      />
      <datalist id={`${id}-ticks`}>
        {steps.map((s) => (
          <option key={s} value={s} />
        ))}
      </datalist>
      {showValue && (
        <span className="min-w-[4rem] text-right text-xs text-muted">Step {value ?? min}</span>
      )}
    </div>
  )
}
