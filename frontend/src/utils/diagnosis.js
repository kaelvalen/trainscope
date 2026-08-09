/**
 * Spike cascade diagnosis: translate a pre/post-spike window into a
 * chronological failure narrative. Pure logic extracted from
 * SpikeInspector so it can be unit-tested without rendering.
 */

export function diagnoseSpike(globalWindow, selectedSpike, currentEvent) {
  if (!globalWindow || globalWindow.length === 0) return null

  // Chronological scan of pre-spike window up to spike step
  let firstDriftStep = null
  let firstGradExplosionStep = null
  let firstNanStep = null

  const baselineLoss = globalWindow[0]?.loss || 1.0
  const baselineGrad = globalWindow[0]?.grad_norm_before_clip || 1.0

  for (const row of globalWindow) {
    if (firstNanStep == null && (row.grad_nan_inf_ratio > 0 || !Number.isFinite(row.loss))) {
      firstNanStep = row.step
    }
    if (
      firstGradExplosionStep == null &&
      row.grad_norm_before_clip > Math.max(5.0, baselineGrad * 3.0)
    ) {
      firstGradExplosionStep = row.step
    }
    // spike_score is the backend detector's actual anomaly score (CUSUM or
    // z-score, whichever is configured), persisted per step. Older runs
    // written before this field existed fall back to a rough heuristic.
    const zScore = row.spike_score ?? (row.loss - baselineLoss) / Math.max(1e-6, baselineLoss * 0.1)
    if (firstDriftStep == null && (zScore >= 3.5 || row.loss > baselineLoss * 1.5)) {
      firstDriftStep = row.step
    }
  }

  // Determine Chronological Sequence of Events
  const events = []
  if (firstDriftStep != null) events.push({ name: 'Loss Shift', step: firstDriftStep })
  if (firstGradExplosionStep != null)
    events.push({ name: 'Gradient Explosion', step: firstGradExplosionStep })
  if (firstNanStep != null) events.push({ name: 'NaN / Inf Collapse', step: firstNanStep })

  events.sort((a, b) => a.step - b.step)

  if (events.length === 0) {
    return {
      primaryTrigger: 'Distributional Shift',
      cascadePath: `Step ${selectedSpike}`,
      badgeVariant: 'accent',
      desc: 'Anomalous deviation detected in loss stream baseline.',
    }
  }

  const primaryTrigger = events[0].name
  const cascadePath = events.map((e) => `${e.name} (Step ${e.step})`).join(' → ')

  let badgeVariant = 'accent'
  if (primaryTrigger === 'NaN / Inf Collapse') {
    badgeVariant = 'danger'
  } else if (primaryTrigger === 'Gradient Explosion') {
    badgeVariant = 'warning'
  }

  let narrative = `Chronological Failure Cascade: ${cascadePath}.`
  if (currentEvent && currentEvent.earlyWarningWindow > 0) {
    narrative += ` The detected anomaly window began ${currentEvent.earlyWarningWindow} steps before the highest observed loss at Step ${currentEvent.peakStep}.`
  }

  return {
    primaryTrigger,
    cascadePath,
    badgeVariant,
    desc: narrative,
  }
}
