import { CHART_COLORS } from '../theme.js'

/**
 * Group raw spike detection triggers into coherent spike events / anomaly cascades.
 *
 * @param {Array} globalData - Array of global step row objects { step, loss, is_spike, z_score, ... }
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
        zScore: r.z_score ?? null,
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

/**
 * Generate Plotly shape objects for clustered spike events.
 */
export function buildSpikeClusterShapes(events = [], options = {}) {
  const shapes = []
  const baseColor = options.color || CHART_COLORS.spike || '#f87171'

  for (const event of events) {
    if (event.startStep === event.endStep) {
      // Single isolated spike: vertical dashed line
      shapes.push({
        type: 'line',
        xref: 'x',
        x0: event.startStep,
        x1: event.startStep,
        yref: 'paper',
        y0: 0,
        y1: 1,
        line: {
          color: baseColor,
          width: 1.5,
          dash: 'dash',
        },
      })
    } else {
      // Multi-step anomaly event: shaded background region
      shapes.push({
        type: 'rect',
        xref: 'x',
        x0: event.startStep - 0.4,
        x1: event.endStep + 0.4,
        yref: 'paper',
        y0: 0,
        y1: 1,
        fillcolor: 'rgba(239, 68, 68, 0.12)',
        line: {
          color: 'rgba(239, 68, 68, 0.45)',
          width: 1,
          dash: 'dash',
        },
      })

      // First detected anomaly line (Start Step)
      shapes.push({
        type: 'line',
        xref: 'x',
        x0: event.startStep,
        x1: event.startStep,
        yref: 'paper',
        y0: 0,
        y1: 1,
        line: {
          color: 'rgba(239, 68, 68, 0.85)',
          width: 1.5,
          dash: 'dash',
        },
      })

      // Peak Catastrophe line (Peak Step), if distinct from startStep
      if (event.peakStep > event.startStep) {
        shapes.push({
          type: 'line',
          xref: 'x',
          x0: event.peakStep,
          x1: event.peakStep,
          yref: 'paper',
          y0: 0,
          y1: 1,
          line: {
            color: 'rgba(239, 68, 68, 0.95)',
            width: 1.5,
            dash: 'solid',
          },
        })
      }
    }
  }

  return shapes
}

/**
 * Generate Plotly annotation objects for clustered spike events.
 */
export function buildSpikeClusterAnnotations(events = [], _options = {}) {
  const annotations = []

  for (const event of events) {
    const isSingle = event.startStep === event.endStep
    const midX = isSingle ? event.startStep : (event.startStep + event.endStep) / 2

    let text = ''
    if (isSingle) {
      text = `⚡ Spike @ Step ${event.startStep}`
    } else if (event.earlyWarningWindow > 0) {
      text = `⚡ Detected Anomaly Window (Steps ${event.startStep}–${event.endStep})`
    } else {
      text = `⚡ Spike Window (Steps ${event.startStep}–${event.endStep})`
    }

    annotations.push({
      x: midX,
      y: 0.98,
      xref: 'x',
      yref: 'paper',
      text,
      showarrow: false,
      font: {
        color: CHART_COLORS.spike || '#f87171',
        size: 11,
        weight: 'bold',
      },
      bgcolor: 'rgba(15, 23, 42, 0.85)',
      bordercolor: 'rgba(239, 68, 68, 0.4)',
      borderwidth: 1,
      borderpad: 5,
      yanchor: 'top',
    })
  }

  return annotations
}
