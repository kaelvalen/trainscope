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
  globalThis.fetch = vi.fn(async (url, options) => {
    const path = new URL(url, window.location.origin).pathname
    const payload =
      payloads[path] !== undefined
        ? payloads[path]
        : {
            '/api/meta': { model_name: 'MiniGPT', trainscope_config: { run_name: 'run_a' } },
            '/api/global': [],
            '/api/layers': [],
            '/api/spikes': [],
            '/api/runs': RUNS,
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
    expect(screen.getByText('2', { selector: 'span.font-mono' })).toBeInTheDocument()
    expect(screen.getByText('0', { selector: 'span.font-mono' })).toBeInTheDocument()
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
        '/api/runs': RUNS,
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
    fireEvent.click(screen.getByText('run_b'))
    await waitFor(() => expect(selectBody).toEqual({ name: 'run_b' }))
  })
})
