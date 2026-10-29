/**
 * Group raw spike detection triggers into coherent spike events / anomaly cascades.
 *
 * @param {Array} globalData - Array of global step row objects { step, loss, is_spike, spike_score, ... }
 * @param {Array} spikeList - Fallback array of spike objects { step: N } if globalData is not ready
 * @param {number} maxGap - Maximum gap between consecutive detection steps to merge into a single event (default: 5)
 * @returns {Array<SpikeEvent>} Array of clustered spike events
 */
export function groupSpikes(globalData = [], spikeList = [], maxGap = 5) {
  let items = []

  if (Array.isArray(globalData) && globalData.some((r) => r.is_spike)) {
    items = globalData
      .filter((r) => r.is_spike)
      .map((r) => ({
        step: r.step,
        loss: r.loss ?? null,
        zScore: r.spike_score ?? null,
      }))
  } else if (Array.isArray(spikeList) && spikeList.length > 0) {
    items = spikeList.map((s) => ({
      step: typeof s === 'number' ? s : s.step,
      loss: null,
      zScore: null,
    }))
  }

  if (items.length === 0) return []

  // Sort by step ascending
  items.sort((a, b) => a.step - b.step)

  const events = []
  let currentGroup = [items[0]]

  for (let i = 1; i < items.length; i++) {
    const prevStep = items[i - 1].step
    const currStep = items[i].step

    if (currStep - prevStep <= maxGap) {
      currentGroup.push(items[i])
    } else {
      events.push(createSpikeEvent(currentGroup, events.length))
      currentGroup = [items[i]]
    }
  }

  if (currentGroup.length > 0) {
    events.push(createSpikeEvent(currentGroup, events.length))
  }

  return events
}

function createSpikeEvent(groupItems, index) {
  const steps = groupItems.map((item) => item.step)
  const startStep = steps[0]
  const endStep = steps[steps.length - 1]
  const count = steps.length

  // Find peak step (highest loss, fallback to highest zScore, fallback to endStep)
  let peakItem = groupItems[0]
  for (const item of groupItems) {
    if (item.loss != null && (peakItem.loss == null || item.loss > peakItem.loss)) {
      peakItem = item
    } else if (
      item.loss == null &&
      item.zScore != null &&
      (peakItem.zScore == null || item.zScore > peakItem.zScore)
    ) {
      peakItem = item
    }
  }

  const peakStep = peakItem.step
  const peakLoss = peakItem.loss
  const earlyWarningWindow = Math.max(0, peakStep - startStep)

  let label = ''
  if (startStep === endStep) {
    label = `Step ${startStep}`
  } else if (earlyWarningWindow > 0) {
    label = `Steps ${startStep}–${endStep} (${earlyWarningWindow}-step detection lead)`
  } else {
    label = `Steps ${startStep}–${endStep}`
  }

  return {
    id: `event_${index}`,
    startStep,
    endStep,
    count,
    steps,
    peakStep,
    peakLoss,
    earlyWarningWindow,
    label,
  }
}

/** Grouped spike events as chart bands. */
export function spikeBands(events = []) {
  return events.map((e) => ({ from: e.startStep, to: e.endStep, kind: 'spike' }))
}
