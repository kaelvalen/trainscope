import { render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'

let runState
vi.mock('../RunContext.jsx', () => ({ useRun: () => runState }))
vi.mock('../api.js', () => ({
  fetchLayerMatrix: vi.fn(async () => ({
    layers: [],
    bucket_starts: [],
    values: [],
    z: [],
    baseline: null,
  })),
  fetchSignals: vi.fn(async () => signals),
}))
vi.mock('../components/charts/LineChart.jsx', () => ({
  default: () => <div data-testid="chart" />,
}))
vi.mock('../components/charts/LayerHeatmap.jsx', () => ({
  default: () => <div data-testid="heatmap" />,
}))

import Overview from '../views/Overview.jsx'

let signals
const rows = (n, spikeAt = null) =>
  Array.from({ length: n }, (_, step) => ({
    step,
    loss: spikeAt != null && step >= spikeAt ? 9 : 2,
    grad_norm_before_clip: 1,
  }))

beforeEach(() => {
  signals = { fired: [], first: null, first_steps: {}, explosion_step: null }
  runState = {
    meta: { trainscope_config: { run_name: 'run_a' } },
    globalData: rows(20),
    spikeEvents: [],
    runs: [],
    switchRun: vi.fn(),
    liveStatus: 'disconnected',
  }
})
afterEach(() => vi.clearAllMocks())

test('waits for the first step when nothing is recorded', () => {
  runState.globalData = []
  render(<Overview />)
  expect(screen.getByText(/Waiting for the first step from run_a/)).toBeInTheDocument()
})

test('a stable run says so in the strip and the signals panel', async () => {
  render(<Overview />)
  expect(screen.getByText('No spike')).toBeInTheDocument()
  expect(await screen.findByText('No signal crossed its threshold')).toBeInTheDocument()
})

test('several spikes are summarised by the first window', () => {
  runState.globalData = rows(200)
  runState.spikeEvents = [
    { startStep: 113, endStep: 128 },
    { startStep: 150, endStep: 152 },
    { startStep: 170, endStep: 171 },
  ]
  render(<Overview />)
  expect(screen.getByText('3 spikes, first 113–128')).toBeInTheDocument()
})

test('signals are listed chronologically with leads to the first spike', async () => {
  runState.globalData = rows(140, 113)
  runState.spikeEvents = [{ startStep: 113, endStep: 128 }]
  signals = {
    fired: ['kurtosis', 'grad_norm', 'loss'],
    first: 'kurtosis',
    first_steps: { grad_norm: 108, loss: 113, kurtosis: 98 },
    explosion_step: 124,
  }
  render(<Overview />)
  const table = await screen.findByRole('table')
  const cells = within(table)
    .getAllByRole('row')
    .slice(1)
    .map((row) =>
      within(row)
        .getAllByRole('cell')
        .map((c) => c.textContent)
    )
  expect(cells).toEqual([
    ['Activation kurtosis', '98', '15'],
    ['Grad norm', '108', '5'],
    ['Loss', '113', '0'],
  ])
  expect(screen.getByText('step 98')).toBeInTheDocument()
})

test('a spike arriving later updates the strip without a remount', () => {
  const { rerender } = render(<Overview />)
  expect(screen.getByText('No spike')).toBeInTheDocument()
  runState = { ...runState, globalData: rows(30), spikeEvents: [{ startStep: 22, endStep: 25 }] }
  rerender(<Overview />)
  expect(screen.getByText('22–25')).toBeInTheDocument()
})

test('a single recorded step renders', async () => {
  runState.globalData = rows(1)
  render(<Overview />)
  await waitFor(() => expect(screen.getAllByTestId('chart').length).toBe(2))
})
