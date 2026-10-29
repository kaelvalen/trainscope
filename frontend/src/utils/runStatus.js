/**
 * One status word per run: waiting, training, stable, spiked or diverged.
 * Diverged means the last loss is non-finite or still above 10x the median
 * loss before the first spike (the whole run when there is no spike).
 */
const DIVERGE_FACTOR = 10

function median(values) {
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

export function runStatus({ globalData = [], spikeEvents = [], live = false }) {
  const n = globalData.length
  if (n === 0) return { kind: 'waiting', label: 'Waiting', detail: 'Waiting for the first step' }

  const firstSpike = spikeEvents.length ? Math.min(...spikeEvents.map((e) => e.startStep)) : null
  const baseline = globalData
    .filter((r) => (firstSpike == null || r.step < firstSpike) && Number.isFinite(r.loss))
    .map((r) => r.loss)
  const limit = baseline.length ? DIVERGE_FACTOR * median(baseline) : Infinity
  const blownUp = (loss) => !Number.isFinite(loss) || loss > limit

  if (blownUp(globalData[n - 1].loss)) {
    const first = globalData.find((r) => blownUp(r.loss))
    return {
      kind: 'diverged',
      label: 'Diverged',
      detail: `Diverged at step ${first.step}, ${n} steps recorded`,
    }
  }
  if (spikeEvents.length) {
    const k = spikeEvents.length
    return {
      kind: 'spiked',
      label: 'Spiked',
      detail: `${k} ${k === 1 ? 'spike' : 'spikes'}, recovered`,
    }
  }
  if (live) return { kind: 'training', label: 'Training', detail: `${n} steps, no spike yet` }
  return { kind: 'stable', label: 'Stable', detail: `${n} steps, no spike` }
}
