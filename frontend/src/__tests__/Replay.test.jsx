import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import Replay from '../views/Replay.jsx'
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

const REPLAY_PLAN = {
  config: {
    checkpoint: '/runs/run_a/checkpoints/4.pt',
    skip_batches: [2, 3],
    total_skipped: 2,
    generated_at: '2026-08-30T12:00:00',
  },
  skipped_batches: [2, 3],
  skipped_steps: [2, 3],
  n_global_steps: 5,
}

function mockFetch(payloads = {}) {
  globalThis.fetch = vi.fn(async (url) => {
    const path = new URL(url, window.location.origin).pathname
    const defaults = {
      '/api/meta': { trainscope_config: { run_name: 'run_a' } },
      '/api/global': [
        { step: 0, loss: 1.0 },
        { step: 1, loss: 1.1 },
        { step: 2, loss: 2.0 },
        { step: 3, loss: 4.0 },
        { step: 4, loss: 1.2 },
      ],
      '/api/layers': [],
      '/api/spikes': [],
      '/api/runs': [],
      '/api/replay': REPLAY_PLAN,
    }
    return { ok: true, json: async () => payloads[path] ?? defaults[path] }
  })
}

function renderReplay() {
  return render(
    <ToastProvider>
      <RunProvider>
        <Replay />
      </RunProvider>
    </ToastProvider>
  )
}

describe('Replay view', () => {
  beforeEach(() => {
    MockWebSocket.instances = []
    globalThis.WebSocket = MockWebSocket
    mockFetch()
  })

  afterEach(() => {
    delete globalThis.WebSocket
    delete globalThis.fetch
  })

  it('shows the replay plan when a config exists', async () => {
    renderReplay()
    await waitFor(() => expect(screen.getByText('Replay plan')).toBeInTheDocument())
    // skipped batches count = 2 (also "2" for steps to skip; use getAllByText).
    expect(screen.getAllByText('2').length).toBeGreaterThan(0)
    expect(screen.getByText('batch 2')).toBeInTheDocument()
    expect(screen.getByText('batch 3')).toBeInTheDocument()
    expect(screen.getByText(/checkpoints\/4\.pt/)).toBeInTheDocument()
    expect(screen.getByText('Loss with skipped steps')).toBeInTheDocument()
  })

  it('shows empty state when no replay config exists', async () => {
    mockFetch({
      '/api/replay': { config: null, skipped_batches: [], skipped_steps: [], n_global_steps: 5 },
    })
    renderReplay()
    await waitFor(() =>
      expect(screen.getByText(/No replay plan found/i)).toBeInTheDocument()
    )
    expect(screen.getByText(/trainscope replay/)).toBeInTheDocument()
  })
})