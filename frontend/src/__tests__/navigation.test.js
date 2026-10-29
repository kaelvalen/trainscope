import { expect, test } from 'vitest'
import { NAV_ITEMS, resolveTab } from '../navigation.js'

const ids = ['overview', 'layers', 'spikes', 'routing', 'compare', 'replay']

test('tabs are the six views in order', () => {
  expect(NAV_ITEMS.map((item) => item.id)).toEqual(ids)
  expect(NAV_ITEMS.map((item) => item.label)).toEqual([
    'Overview',
    'Layers',
    'Spikes',
    'Routing',
    'Compare',
    'Replay',
  ])
})

test('1.9.0 hashes map onto the new views', () => {
  expect(resolveTab('#diff', ids)).toEqual({ view: 'compare', canonical: '#compare' })
  expect(resolveTab('#timeline', ids).view).toBe('overview')
  expect(resolveTab('#moe', ids).view).toBe('routing')
})

test('unknown, empty and hidden tabs fall back to the overview', () => {
  expect(resolveTab('#garbage', ids)).toEqual({ view: 'overview', canonical: '#overview' })
  expect(resolveTab('', ids).view).toBe('overview')
  expect(
    resolveTab(
      '#routing',
      ids.filter((i) => i !== 'routing')
    ).view
  ).toBe('overview')
})

test('the runs index is its own view', () => {
  expect(resolveTab('#runs', ids)).toEqual({ view: 'runs', canonical: '#runs' })
})

test('a query after the tab id is kept out of the view name', () => {
  expect(resolveTab('#layers?layer=blocks.1.mlp.fc2', ids)).toEqual({
    view: 'layers',
    canonical: '#layers?layer=blocks.1.mlp.fc2',
  })
})
