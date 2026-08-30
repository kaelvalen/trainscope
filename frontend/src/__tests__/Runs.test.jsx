import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { RunProvider } from '../RunContext.jsx'
import { ToastProvider } from '../context/ToastContext.jsx'
import Runs from '../views/Runs.jsx'

class MockWebSocket {
  static instances = []

  constructor(url) {
    this.url = url
    MockWebSocket.instances.push(this)
    window.setTimeout(() => this.onopen?.(), 0)
  }

  close() {
    this.onclose?.()
  }
}

const RUNS = [
  {
    name: 'run_a',
    model_name: 'MiniGPT',
    detector: { name: 'changepoint', threshold: 6.0 },
    n_global_rows: 150,
    start_time: '2026-08-06T17:39:55Z',
    spike_count: 2,
    last_loss: 3.2,
    is_active: true,
  },
  {
    name: 'run_b',
    model_name: 'MiniGPT',
    detector: { name: 'z_score', threshold: 3.5 },
    n_global_rows: 100,
    start_time: '2026-08-06T18:00:00Z',
    spike_count: 0,
    last_loss: 2.1,
    is_active: false,
  },
]

function mockFetch(payloads = {}) {
  globalThis.fetch = vi.fn(async (url) => {
    const path = new URL(url, window.location.origin).pathname
    const payload =
      payloads[path] !== undefined
        ? payloads[path]
        : {
            '/api/meta': { model_name: 'MiniGPT', trainscope_config: { run_name: 'run_a' } },
            '/api/global': [],
            '/api/layers': [],
            '/api/spikes': [],
            '/api/moe': [],
            '/api/runs': RUNS,
            '/api/cluster': { clusters: [], unclustered: [] },
          }[path]
    return {
      ok: true,
      json: async () => payload,
    }
  })
}

describe('Runs view', () => {
  beforeEach(() => {
    MockWebSocket.instances = []
    globalThis.WebSocket = MockWebSocket
    mockFetch()
  })

  afterEach(() => {
    delete globalThis.WebSocket
    delete globalThis.fetch
  })

  it('lists every run with model, spike count and last loss', async () => {
    render(
      <ToastProvider>
        <RunProvider>
          <Runs />
        </RunProvider>
      </ToastProvider>
    )

    await waitFor(() => expect(screen.getByText('run_a')).toBeInTheDocument())
    expect(screen.getByText('run_b')).toBeInTheDocument()
    // Model + detector info is a composite text node per run row.
    expect(screen.getByText(/MiniGPT · changepoint detector/)).toBeInTheDocument()
    expect(screen.getByText(/MiniGPT · z_score detector/)).toBeInTheDocument()
    // Last loss appears in each row (and again in the summary stat card).
    expect(screen.getAllByText('3.2000').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('2.1000').length).toBeGreaterThanOrEqual(1)
    // Spike counts render per row.
    expect(screen.getByText('2 spikes')).toBeInTheDocument()
  })

  it('switches to the selected run via POST /api/runs/select', async () => {
    let selectBody = null
    globalThis.fetch = vi.fn(async (url, options) => {
      const path = new URL(url, window.location.origin).pathname
      if (path === '/api/runs/select' && options?.method === 'POST') {
        selectBody = JSON.parse(options.body)
        return {
          ok: true,
          json: async () => ({ ...RUNS[1], is_active: true }),
        }
      }
      const payload = {
        '/api/meta': { model_name: 'MiniGPT', trainscope_config: { run_name: 'run_a' } },
        '/api/global': [],
        '/api/layers': [],
        '/api/spikes': [],
        '/api/moe': [],
        '/api/runs': RUNS,
        '/api/cluster': { clusters: [], unclustered: [] },
      }[path]
      return { ok: true, json: async () => payload }
    })

    render(
      <ToastProvider>
        <RunProvider>
          <Runs />
        </RunProvider>
      </ToastProvider>
    )

    await waitFor(() => expect(screen.getByText('run_b')).toBeInTheDocument())
    const openButtons = screen.getAllByRole('button', { name: 'Open' })
    fireEvent.click(openButtons[0]) // newest run (run_b) sorts first
    await waitFor(() => expect(selectBody).toEqual({ name: 'run_b' }))
  })

  it('compares selected runs and shows divergence and common cause', async () => {
    globalThis.fetch = vi.fn(async (url) => {
      const path = new URL(url, window.location.origin).pathname
      if (path === '/api/compare') {
        return {
          ok: true,
          json: async () => ({
            runs: ['run_a', 'run_b'],
            summaries: RUNS,
            loss_series: {
              run_a: [
                { step: 0, loss: 1.0 },
                { step: 1, loss: 1.0 },
              ],
              run_b: [
                { step: 0, loss: 1.0 },
                { step: 1, loss: 2.0 },
              ],
            },
            divergence: { step: 1, baseline_gap: 0.0, threshold: 1e-6, min_run: 1 },
            config_diff: [
              { field: 'config.full_resolution_window', values: { run_a: 500, run_b: 1000 } },
            ],
            common_cause: [
              { field: 'config.detector.threshold', spiked_value: 6.0, stable_value: 3.5 },
              {
                field: 'max routing concentration',
                spiked_value: 0.95,
                stable_value: 0.3,
              },
            ],
            concentration_series: {
              run_a: [
                { step: 0, max_share: 0.3 },
                { step: 1, max_share: 0.95 },
              ],
              run_b: [
                { step: 0, max_share: 0.3 },
                { step: 1, max_share: 0.3 },
              ],
            },
          }),
        }
      }
      const payload = {
        '/api/meta': { model_name: 'MiniGPT', trainscope_config: { run_name: 'run_a' } },
        '/api/global': [],
        '/api/layers': [],
        '/api/spikes': [],
        '/api/moe': [],
        '/api/runs': RUNS,
        '/api/cluster': { clusters: [], unclustered: [] },
      }[path]
      return { ok: true, json: async () => payload }
    })

    render(
      <ToastProvider>
        <RunProvider>
          <Runs />
        </RunProvider>
      </ToastProvider>
    )

    await waitFor(() => expect(screen.getByText('run_a')).toBeInTheDocument())
    fireEvent.click(screen.getByLabelText('Select run run_a'))
    fireEvent.click(screen.getByLabelText('Select run run_b'))
    fireEvent.click(screen.getByRole('button', { name: 'Compare (2)' }))

    await waitFor(() => expect(screen.getByText('Loss curves')).toBeInTheDocument())
    expect(screen.getByText('Step 1')).toBeInTheDocument()
    expect(screen.getByText('Common cause')).toBeInTheDocument()
    expect(
      screen.getByText((content) => content.includes('every run with spikes has'))
    ).toBeInTheDocument()
    // Concentration overlay + architecture-aware common cause sentence.
    expect(screen.getByText('Routing / addressing concentration')).toBeInTheDocument()
    expect(
      screen.getByText((content) => content.includes('every run with spikes concentrated'))
    ).toBeInTheDocument()
    expect(screen.getByText('config.full_resolution_window')).toBeInTheDocument()
    expect(screen.getByText('1000')).toBeInTheDocument()
  })
})

describe('Runs view clusters', () => {
  beforeEach(() => {
    MockWebSocket.instances = []
    globalThis.WebSocket = MockWebSocket
    globalThis.fetch = vi.fn(async (url) => {
      const path = new URL(url, window.location.origin).pathname
      const payload = {
        '/api/meta': { model_name: 'MiniGPT', trainscope_config: { run_name: 'run_a' } },
        '/api/global': [],
        '/api/layers': [],
        '/api/spikes': [],
        '/api/moe': [],
        '/api/runs': RUNS,
        '/api/cluster': {
          clusters: [
            {
              label: 'gradient-led',
              first_signal: 'grad_norm',
              fired_signals: ['grad_norm'],
              runs: ['run_a', 'run_b'],
              n_runs: 2,
              typical_lead_steps: 23.5,
              discriminant_traits: [{ field: 'model.lr', value: 5e-4 }],
              crossing_steps: [38, 40],
              loss_band: {
                steps: [0, 1, 2],
                median: [1.0, 1.05, 1.1],
                lower: [0.98, 1.0, 1.05],
                upper: [1.02, 1.1, 1.2],
              },
            },
          ],
          unclustered: [],
        },
        '/api/counterexample': {
          query_run: 'run_a',
          counterexample_run: 'run_c',
          config_distance: 0.05,
          first_signal: 'grad_norm',
          crossing_step: 40,
          loss_series: {
            run_a: [
              { step: 0, loss: 1.0 },
              { step: 40, loss: 2.0 },
              { step: 41, loss: 8.0 },
            ],
            run_c: [
              { step: 0, loss: 1.0 },
              { step: 40, loss: 1.1 },
              { step: 41, loss: 1.1 },
            ],
          },
          config_diff: [{ field: 'model.lr', query_value: 0.0005, stable_value: 0.0001 }],
        },
      }
      return { ok: true, json: async () => payload[path] }
    })
  })

  afterEach(() => {
    delete globalThis.WebSocket
    delete globalThis.fetch
  })

  it('renders behavior clusters with run buttons', async () => {
    render(
      <ToastProvider>
        <RunProvider>
          <Runs />
        </RunProvider>
      </ToastProvider>
    )
    await waitFor(() => expect(screen.getByText('gradient-led')).toBeInTheDocument())
    expect(
      screen.getByText((content) => content.includes('Run behavior clusters'))
    ).toBeInTheDocument()
    expect(screen.getByText('2 runs')).toBeInTheDocument()
    expect(screen.getByText('~23.5-step lead')).toBeInTheDocument()
    expect(screen.getByText('distinct: model.lr=0.0005')).toBeInTheDocument()
    expect(screen.getByText('first signal at')).toBeInTheDocument()
    expect(screen.getByText('step 38')).toBeInTheDocument()
    expect(screen.getByText('step 40')).toBeInTheDocument()
  })

  it('shows the nearest stable counterexample on demand', async () => {
    render(
      <ToastProvider>
        <RunProvider>
          <Runs />
        </RunProvider>
      </ToastProvider>
    )
    await waitFor(() => expect(screen.getByText('gradient-led')).toBeInTheDocument())
    fireEvent.click(screen.getByRole('button', { name: /Why this.*vs\. stable/i }))
    await waitFor(() =>
      expect(screen.getByText(/Why run_a vs\. run_c/i)).toBeInTheDocument()
    )
    expect(screen.getByText(/First signal: grad_norm at step 40/i)).toBeInTheDocument()
    expect(screen.getByText('model.lr')).toBeInTheDocument()
    expect(screen.getByText('0.0005')).toBeInTheDocument()
  })
})
