import { expect, test } from 'vitest'
import { runStatus } from '../utils/runStatus.js'

const rows = (losses) => losses.map((loss, step) => ({ step, loss }))

test('no rows is waiting', () => {
  expect(runStatus({ globalData: [], spikeEvents: [], live: true }).kind).toBe('waiting')
})

test('live without spikes is training', () => {
  const s = runStatus({ globalData: rows([3, 2, 2]), spikeEvents: [], live: true })
  expect(s).toMatchObject({ kind: 'training', label: 'Training' })
})

test('finished without spikes is stable', () => {
  const s = runStatus({ globalData: rows([3, 2, 2]), spikeEvents: [], live: false })
  expect(s).toEqual({ kind: 'stable', label: 'Stable', detail: '3 steps, no spike' })
})

test('a recovered spike is spiked', () => {
  const s = runStatus({
    globalData: rows([2, 2, 2, 9, 2, 2]),
    spikeEvents: [{ startStep: 3, endStep: 3 }],
    live: false,
  })
  expect(s).toEqual({ kind: 'spiked', label: 'Spiked', detail: '1 spike, recovered' })
})

test('a loss still 10x the pre-spike median is diverged', () => {
  const s = runStatus({
    globalData: rows([2, 2, 2, 9, 25, 30]),
    spikeEvents: [
      { startStep: 3, endStep: 5 },
      { startStep: 8, endStep: 8 },
    ],
    live: false,
  })
  expect(s).toEqual({
    kind: 'diverged',
    label: 'Diverged',
    detail: 'Diverged at step 4, 6 steps recorded',
  })
})

test('a non-finite last loss is diverged', () => {
  const s = runStatus({ globalData: rows([2, 2, NaN]), spikeEvents: [], live: true })
  expect(s.kind).toBe('diverged')
  expect(s.detail).toBe('Diverged at step 2, 3 steps recorded')
})
