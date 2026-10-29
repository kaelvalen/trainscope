import { expect, test } from 'vitest'
import { buildLineOptions, lossYMax, offScaleMarker } from '../components/charts/chartOptions.js'

test('lossYMax is 1.25x the pre-spike max', () => {
  expect(lossYMax([0, 1, 2, 3], [4, 2, 3, 60], 3)).toBeCloseTo(5)
  expect(lossYMax([0, 1], [4, 2], null)).toBeNull()
  expect(lossYMax([5, 6], [4, 2], 3)).toBeNull()
})

test('offScaleMarker reports peak or non-finite', () => {
  expect(offScaleMarker([0, 1, 2], [1, 9, 63.6], 5)).toEqual({
    at: 1,
    text: '63.6 at step 2, off scale',
  })
  expect(offScaleMarker([0, 1], [1, NaN], 5).text).toBe('non-finite at step 1')
  expect(offScaleMarker([0, 1], [1, 2], 5)).toBeNull()
  expect(offScaleMarker([0, 1], [1, 9], null)).toBeNull()
})

test('options: one y scale, 2px lines, single-step safe', () => {
  const o = buildLineOptions(
    { x: [0], series: [{ label: 'Loss', values: [Infinity], color: 'var(--ink)' }], height: 200 },
    { width: 600 }
  )
  expect(Object.keys(o.scales)).toEqual(['x', 'y'])
  expect(o.series[1].width).toBe(2)
  expect(o.width).toBe(600)
  expect(o.height).toBe(200)
})

test('options: yMax pins the y range and a legend appears only for 2+ series', () => {
  const one = buildLineOptions(
    { x: [0, 1], series: [{ label: 'Loss', values: [1, 2], color: '#000' }], yMax: 5, height: 100 },
    { width: 300 }
  )
  expect(one.scales.y.range(null, 0, 99)).toEqual([0, 5])
  expect(one.legend.show).toBe(false)
  const two = buildLineOptions(
    {
      x: [0, 1],
      series: [
        { label: 'A', values: [1, 2], color: '#000' },
        { label: 'B', values: [2, 1], color: '#00f' },
      ],
      height: 100,
    },
    { width: 300 }
  )
  expect(two.legend.show).toBe(true)
})

test('a null loss is what NaN looks like on the wire, so it counts as non-finite', () => {
  expect(offScaleMarker([0, 1], [1, null], 5).text).toBe('non-finite at step 1')
})

test('bandSpan keeps single-step bands visible', async () => {
  const { bandSpan } = await import('../components/charts/chartOptions.js')
  expect(bandSpan(100, 100, 3)).toEqual([98.5, 101.5])
  expect(bandSpan(10, 50, 3)).toEqual([10, 50])
})

test('the y range follows the latest yMax and limit without a rebuild', () => {
  let current = { yMax: null, limit: null }
  const o = buildLineOptions(
    { x: [0, 1], series: [{ label: 'Loss', values: [1, 2], color: '#000' }], height: 100 },
    { width: 300 },
    () => current
  )
  expect(o.scales.y.range(null, 0, 10)[1]).toBeGreaterThan(10)
  current = { yMax: 5, limit: null }
  expect(o.scales.y.range(null, 0, 10)).toEqual([0, 5])
})
