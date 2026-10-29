import { expect, test } from 'vitest'
import {
  firstCrossings,
  groupRows,
  heatmapSummary,
  sortRows,
  zColor,
} from '../utils/layerMatrix.js'

const matrixOf = (layers, zFor) => ({
  layers,
  bucket_starts: [0, 10, 20, 30],
  values: layers.map(() => [1, 1, 1, 1]),
  z: layers.map(zFor),
})

test('more than 24 rows group by module path up to the first number', () => {
  const layers = Array.from({ length: 15 }, (_, i) => [`blocks.${i}.w`, `blocks.${i}.b`]).flat()
  const rows = groupRows(
    matrixOf(layers, (name) => (name === 'blocks.3.b' ? [0, 5, 1, 0] : [0, 1, 2, 0]))
  )
  expect(rows).toHaveLength(15)
  expect(rows[3]).toMatchObject({
    key: 'blocks.3',
    isGroup: true,
    children: ['blocks.3.w', 'blocks.3.b'],
  })
  expect(rows[3].z).toEqual([0, 5, 2, 0])
})

test('expanded groups also list their children', () => {
  const layers = Array.from({ length: 13 }, (_, i) => [`blocks.${i}.w`, `blocks.${i}.b`]).flat()
  const rows = groupRows(
    matrixOf(layers, () => [0, 0, 0, 0]),
    { expanded: new Set(['blocks.0']) }
  )
  expect(rows.slice(0, 3).map((r) => r.key)).toEqual(['blocks.0', 'blocks.0.w', 'blocks.0.b'])
})

test('24 rows or fewer stay ungrouped', () => {
  const rows = groupRows(matrixOf(['a.0.w', 'a.1.w'], () => [0, 0, 0, 0]))
  expect(rows.map((r) => r.key)).toEqual(['a.0.w', 'a.1.w'])
  expect(rows.every((r) => !r.isGroup)).toBe(true)
})

test('firstCrossings is the first bucket start at or above the threshold', () => {
  const rows = groupRows(
    matrixOf(['x', 'y', 'z'], (n) => ({ x: [0, 3, 6, 6], y: [0, 0, 0, 4], z: [0, 1, null, 2] })[n])
  )
  const c = firstCrossings(rows, [0, 10, 20, 30])
  expect([c.get('x'), c.get('y'), c.get('z')]).toEqual([10, 30, null])
  expect(sortRows(rows, 'first', c).map((r) => r.key)).toEqual(['x', 'y', 'z'])
  expect(sortRows(rows, 'model', c).map((r) => r.key)).toEqual(['x', 'y', 'z'])
})

test('sortRows puts the earliest crossing first and nulls last', () => {
  const rows = groupRows(
    matrixOf(['a', 'b', 'c'], (n) => ({ a: [0, 0, 0, 0], b: [0, 0, 4, 4], c: [0, 5, 5, 5] })[n])
  )
  const c = firstCrossings(rows, [0, 10, 20, 30])
  expect(sortRows(rows, 'first', c).map((r) => r.key)).toEqual(['c', 'b', 'a'])
})

test('heatmapSummary names up to three first crossers', () => {
  const rows = groupRows(
    matrixOf(
      ['blocks.1.attn.qkv', 'blocks.1.mlp.fc1', 'blocks.1.mlp.fc2', 'emb'],
      (n) =>
        ({
          'blocks.1.mlp.fc2': [3, 3, 3, 3],
          'blocks.1.mlp.fc1': [0, 3, 3, 3],
          'blocks.1.attn.qkv': [0, 0, 3, 3],
          emb: [0, 0, 0, 3],
        })[n]
    )
  )
  const c = firstCrossings(rows, [98, 101, 104, 110])
  expect(heatmapSummary(rows, c)).toBe(
    'First above 3σ: blocks.1.mlp.fc2 at step 98, blocks.1.mlp.fc1 at step 101, blocks.1.attn.qkv at step 104'
  )
  const quiet = groupRows(matrixOf(['a'], () => [0, 0, 0, 0]))
  expect(heatmapSummary(quiet, firstCrossings(quiet, [0, 1, 2, 3]))).toBe('No layer crossed 3σ')
})

test('zColor ramps from the surface to red', () => {
  const surface = [255, 255, 255]
  const red = [140, 22, 28]
  expect(zColor(0, surface, red)).toBe('rgb(255,255,255)')
  expect(zColor(6, surface, red)).toBe('rgb(140,22,28)')
  expect(zColor(null, surface, red)).toBe('rgb(255,255,255)')
  expect(zColor(12, surface, red)).toBe('rgb(140,22,28)')
})

test('empty matrix gives no rows', () => {
  expect(groupRows({ layers: [], bucket_starts: [], values: [], z: [] })).toEqual([])
})
