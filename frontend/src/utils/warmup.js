/**
 * Compute the detector warmup band from run metadata and recorded steps.
 *
 * The anomaly detector produces no scores until it has seen
 * `min_observations` loss values, so spikes during that window are invisible.
 * The band is expressed in recorded-step space: it spans from the first
 * recorded step to the step of the `min_observations`-th recorded row.
 *
 * @param {Object|null} meta - Run metadata from `/api/meta` (with a `detector`
 *   section carrying `min_observations`).
 * @param {Array} globalData - Array of global step rows (ascending by step).
 * @returns {{startStep: number, endStep: number}|null} Band edges, or null
 *   when the warmup window is unknown or no steps are recorded yet.
 */
export function computeWarmupBand(meta, globalData = []) {
  const minObservations = meta?.detector?.min_observations
  if (!Number.isInteger(minObservations) || minObservations <= 0) return null
  if (!Array.isArray(globalData) || globalData.length === 0) return null

  const startRow = globalData[0]
  const endRow = globalData[Math.min(minObservations, globalData.length) - 1]
  if (startRow?.step == null || endRow?.step == null) return null

  return { startStep: startRow.step, endStep: endRow.step }
}
