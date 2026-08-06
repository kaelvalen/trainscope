import { describe, it, expect } from 'vitest'
import {
  groupSpikes,
  buildSpikeClusterShapes,
  buildSpikeClusterAnnotations,
} from '../utils/spikeCluster.js'

describe('spikeCluster utils', () => {
  it('returns empty array when no spikes are present', () => {
    expect(groupSpikes([])).toEqual([])
    expect(groupSpikes([{ step: 0, is_spike: false }])).toEqual([])
  })

  it('groups consecutive raw spike detections into a single event with early warning window', () => {
    const globalData = []
    // 150 steps, step 80 to 99 drifting, step 100 peak
    for (let i = 0; i <= 150; i++) {
      const isSpike = i >= 80 && i <= 100
      globalData.push({
        step: i,
        loss: i === 100 ? 558.95 : isSpike ? 5.5 + (i - 80) * 0.8 : 5.5,
        is_spike: isSpike,
      })
    }

    const events = groupSpikes(globalData)
    expect(events.length).toBe(1)

    const ev = events[0]
    expect(ev.startStep).toBe(80)
    expect(ev.endStep).toBe(100)
    expect(ev.count).toBe(21)
    expect(ev.peakStep).toBe(100)
    expect(ev.earlyWarningWindow).toBe(20)
    expect(ev.label).toContain('20-step early warning')
  })

  it('handles multiple separate spike events', () => {
    const globalData = [
      { step: 10, loss: 10.0, is_spike: true },
      { step: 11, loss: 12.0, is_spike: true },
      { step: 50, loss: 15.0, is_spike: true },
      { step: 80, loss: 6.0, is_spike: true },
      { step: 81, loss: 7.0, is_spike: true },
      { step: 82, loss: 100.0, is_spike: true },
    ]

    const events = groupSpikes(globalData, [], 5)
    expect(events.length).toBe(3)
    expect(events[0].startStep).toBe(10)
    expect(events[0].endStep).toBe(11)

    expect(events[1].startStep).toBe(50)
    expect(events[1].endStep).toBe(50)

    expect(events[2].startStep).toBe(80)
    expect(events[2].endStep).toBe(82)
    expect(events[2].peakStep).toBe(82)
    expect(events[2].earlyWarningWindow).toBe(2)
  })

  it('generates Plotly region shape and single annotation badge for grouped event', () => {
    const events = [
      {
        id: 'event_0',
        startStep: 80,
        endStep: 100,
        count: 21,
        peakStep: 100,
        earlyWarningWindow: 20,
        label: 'Steps 80–100 (20-step early warning)',
      },
    ]

    const shapes = buildSpikeClusterShapes(events)
    expect(shapes.length).toBe(3) // rect region, start trigger line, peak line
    expect(shapes[0].type).toBe('rect')
    expect(shapes[0].x0).toBe(79.6)
    expect(shapes[0].x1).toBe(100.4)

    const annotations = buildSpikeClusterAnnotations(events)
    expect(annotations.length).toBe(1) // ONLY 1 clean annotation instead of 21
    expect(annotations[0].text).toContain('Early Warning Window (Steps 80–100)')
    expect(annotations[0].x).toBe(90)
  })
})
