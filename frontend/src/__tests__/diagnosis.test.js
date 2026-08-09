import { describe, it, expect } from 'vitest'
import { diagnoseSpike } from '../utils/diagnosis.js'

describe('diagnoseSpike', () => {
  it('returns null for an empty window', () => {
    expect(diagnoseSpike([], 100, null)).toBeNull()
    expect(diagnoseSpike(null, 100, null)).toBeNull()
  })

  it('flags a Loss Shift from a persisted spike_score above 3.5', () => {
    const window = [
      { step: 90, loss: 1.0, grad_norm_before_clip: 1.0, spike_score: 0.5 },
      { step: 91, loss: 1.01, grad_norm_before_clip: 1.1, spike_score: 1.2 },
      { step: 92, loss: 1.05, grad_norm_before_clip: 1.2, spike_score: 4.0 },
    ]
    const diagnosis = diagnoseSpike(window, 92, null)
    expect(diagnosis.primaryTrigger).toBe('Loss Shift')
    expect(diagnosis.cascadePath).toBe('Loss Shift (Step 92)')
    expect(diagnosis.badgeVariant).toBe('accent')
  })

  it('falls back to the loss-vs-baseline heuristic when spike_score is missing', () => {
    const window = [
      { step: 90, loss: 1.0, grad_norm_before_clip: 1.0 },
      { step: 91, loss: 1.1, grad_norm_before_clip: 1.1 },
      { step: 92, loss: 1.6, grad_norm_before_clip: 1.2 },
    ]
    const diagnosis = diagnoseSpike(window, 92, null)
    expect(diagnosis.primaryTrigger).toBe('Loss Shift')
    // Loss jumped > 1.5x baseline (1.0 -> 1.6) at step 92.
    expect(diagnosis.cascadePath).toBe('Loss Shift (Step 92)')
  })

  it('detects Gradient Explosion when grad norm exceeds 5.0 even with a low baseline', () => {
    const window = [
      { step: 90, loss: 1.0, grad_norm_before_clip: 1.0 },
      { step: 91, loss: 1.0, grad_norm_before_clip: 1.2 },
      { step: 92, loss: 1.1, grad_norm_before_clip: 6.0 },
    ]
    const diagnosis = diagnoseSpike(window, 92, null)
    expect(diagnosis.primaryTrigger).toBe('Gradient Explosion')
    expect(diagnosis.badgeVariant).toBe('warning')
  })

  it('detects Gradient Explosion at 3x baseline grad norm when baseline exceeds the 5.0 floor', () => {
    const window = [
      { step: 90, loss: 1.0, grad_norm_before_clip: 2.5 },
      { step: 91, loss: 1.0, grad_norm_before_clip: 2.6 },
      { step: 92, loss: 1.1, grad_norm_before_clip: 7.7 }, // > 3 * 2.5, also > 5.0
    ]
    const diagnosis = diagnoseSpike(window, 92, null)
    expect(diagnosis.primaryTrigger).toBe('Gradient Explosion')
  })

  it('does not flag grad norm between the 5.0 floor and 3x a high baseline', () => {
    const window = [
      { step: 90, loss: 1.0, grad_norm_before_clip: 2.5 },
      { step: 91, loss: 1.0, grad_norm_before_clip: 2.6 },
      { step: 92, loss: 1.1, grad_norm_before_clip: 5.8 }, // > 5.0 but < 3 * 2.5
    ]
    const diagnosis = diagnoseSpike(window, 92, null)
    expect(diagnosis.primaryTrigger).toBe('Distributional Shift')
  })

  it('detects NaN / Inf Collapse from a non-finite loss', () => {
    const window = [
      { step: 90, loss: 1.0, grad_norm_before_clip: 1.0, grad_nan_inf_ratio: 0 },
      { step: 91, loss: 1.0, grad_norm_before_clip: 1.0, grad_nan_inf_ratio: 0.2 },
    ]
    const diagnosis = diagnoseSpike(window, 91, null)
    expect(diagnosis.primaryTrigger).toBe('NaN / Inf Collapse')
    expect(diagnosis.badgeVariant).toBe('danger')
    expect(diagnosis.cascadePath).toBe('NaN / Inf Collapse (Step 91)')
  })

  it('orders a multi-cause cascade chronologically, not by type', () => {
    const window = [
      { step: 90, loss: 1.0, grad_norm_before_clip: 1.0, spike_score: 0.5 },
      { step: 91, loss: 1.2, grad_norm_before_clip: 1.0, spike_score: 4.0 }, // drift first
      { step: 92, loss: 1.5, grad_norm_before_clip: 9.0, spike_score: 5.0 }, // grad explodes
      {
        step: 93,
        loss: 1.8,
        grad_norm_before_clip: 12.0,
        spike_score: 6.0,
        grad_nan_inf_ratio: 0.3,
      },
    ]
    const diagnosis = diagnoseSpike(window, 93, null)
    expect(diagnosis.primaryTrigger).toBe('Loss Shift')
    expect(diagnosis.cascadePath).toBe(
      'Loss Shift (Step 91) → Gradient Explosion (Step 92) → NaN / Inf Collapse (Step 93)'
    )
    // Primary trigger is the drift, so the badge stays accent.
    expect(diagnosis.badgeVariant).toBe('accent')
  })

  it('reports the early-warning window when a currentEvent is provided', () => {
    const window = [
      { step: 90, loss: 1.0, grad_norm_before_clip: 1.0, spike_score: 0.5 },
      { step: 91, loss: 1.2, grad_norm_before_clip: 1.0, spike_score: 4.0 },
    ]
    const event = {
      startStep: 80,
      endStep: 91,
      peakStep: 91,
      earlyWarningWindow: 11,
    }
    const diagnosis = diagnoseSpike(window, 91, event)
    expect(diagnosis.desc).toContain('began 11 steps before the highest observed loss at Step 91')
  })

  it('omits the warning narrative when the event has no early warning', () => {
    const window = [{ step: 90, loss: 1.0, grad_norm_before_clip: 1.0, spike_score: 4.0 }]
    const diagnosis = diagnoseSpike(window, 90, { earlyWarningWindow: 0 })
    expect(diagnosis.desc).not.toContain('anomaly window began')
  })

  it('falls back to Distributional Shift when no mechanism is identified', () => {
    const window = [
      { step: 90, loss: 1.0, grad_norm_before_clip: 1.0, spike_score: 0.5 },
      { step: 91, loss: 1.01, grad_norm_before_clip: 1.0, spike_score: 1.0 },
    ]
    const diagnosis = diagnoseSpike(window, 91, null)
    expect(diagnosis.primaryTrigger).toBe('Distributional Shift')
    expect(diagnosis.cascadePath).toBe('Step 91')
    expect(diagnosis.badgeVariant).toBe('accent')
  })
})
