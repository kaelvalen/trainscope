import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ExpertUtilization from '../views/ExpertUtilization.jsx'
import { RunProvider } from '../RunContext.jsx'
import { ToastProvider } from '../context/ToastContext.jsx'

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

const MOE_ROWS = [
  { step: 0, block: 'blocks.0.router', shares: [0.25, 0.25, 0.25, 0.25] },
  { step: 0, block: 'blocks.1.router', shares: [0.25, 0.25, 0.25, 0.25] },
  { step: 1, block: 'blocks.0.router', shares: [0.9, 0.03, 0.03, 0.04] },
  { step: 1, block: 'blocks.1.router', shares: [0.25, 0.25, 0.25, 0.25] },
]

function mockFetch() {
  globalThis.fetch = vi.fn(async (url) => {
    const path = new URL(url, window.location.origin).pathname
    const payloads = {
      '/api/meta': { trainscope_config: { run_name: 'test' } },
      '/api/global': [],
      '/api/layers': [],
      '/api/spikes': [],
      '/api/runs': [],
      '/api/moe': MOE_ROWS,
    }
    return { ok: true, json: async () => payloads[path] }
  })
}

describe('ExpertUtilization view', () => {
  beforeEach(() => {
    MockWebSocket.instances = []
    globalThis.WebSocket = MockWebSocket
    mockFetch()
  })

  afterEach(() => {
    delete globalThis.WebSocket
    delete globalThis.fetch
  })

  it('shows empty state when no MoE data exists', async () => {
    globalThis.fetch = vi.fn(async (url) => {
      const path = new URL(url, window.location.origin).pathname
      const payloads = {
        '/api/meta': { trainscope_config: { run_name: 'test' } },
        '/api/global': [],
        '/api/layers': [],
        '/api/spikes': [],
        '/api/runs': [],
        '/api/moe': [],
      }
      return { ok: true, json: async () => payloads[path] }
    })
    render(
      <ToastProvider>
        <RunProvider>
          <ExpertUtilization />
        </RunProvider>
      </ToastProvider>
    )
    await waitFor(() =>
      expect(screen.getByText(/No MoE routing data in this run/)).toBeInTheDocument()
    )
  })

  it('renders per-block charts and flags concentrated routing', async () => {
    render(
      <ToastProvider>
        <RunProvider>
          <ExpertUtilization />
        </RunProvider>
      </ToastProvider>
    )
    await waitFor(() =>
      expect(screen.getByText(/Routing shares — blocks\.0\.router/)).toBeInTheDocument()
    )
    expect(screen.getByText(/Routing shares — blocks\.1\.router/)).toBeInTheDocument()
    expect(screen.getByText('Router blocks')).toBeInTheDocument()
    // Expert 1 of block 0 has 0.90 share -> danger badge.
    expect(screen.getByText('expert 1: 0.90')).toBeInTheDocument()
    expect(screen.getByText(/Routing concentrated on one expert/)).toBeInTheDocument()
    expect(
      screen.getByText((content) => content.includes('Expert utilization warning'))
    ).toBeInTheDocument()
  })
})
