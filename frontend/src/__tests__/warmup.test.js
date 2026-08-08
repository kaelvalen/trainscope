import { describe, it, expect } from 'vitest'
import { computeWarmupBand } from '../utils/warmup.js'

describe('computeWarmupBand', () => {
  const meta = { detector: { name: 'changepoint', min_observations: 30 } }

  function steps(n, start = 0) {
    return Array.from({ length: n }, (_, i) => ({ step: start + i }))
  }

  it('returns null when detector info is missing or unknown', () => {
    expect(computeWarmupBand(null, steps(10))).toBeNull()
    expect(computeWarmupBand({}, steps(10))).toBeNull()
    expect(computeWarmupBand({ detector: { name: 'custom' } }, steps(10))).toBeNull()
    expect(computeWarmupBand({ detector: { min_observations: 0 } }, steps(10))).toBeNull()
  })

  it('returns null when no steps are recorded yet', () => {
    expect(computeWarmupBand(meta, [])).toBeNull()
    expect(computeWarmupBand(meta)).toBeNull()
  })

  it('spans the first min_observations recorded steps', () => {
    const band = computeWarmupBand(meta, steps(100))
    expect(band).toEqual({ startStep: 0, endStep: 29 })
  })

  it('respects the first recorded step when the run does not start at 0', () => {
    const band = computeWarmupBand(meta, steps(100, 10))
    expect(band).toEqual({ startStep: 10, endStep: 39 })
  })

  it('extends to the last recorded step while still warming up', () => {
    const band = computeWarmupBand(meta, steps(12))
    expect(band).toEqual({ startStep: 0, endStep: 11 })
  })
})
