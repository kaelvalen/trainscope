import { act, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { RunProvider, useRun } from './RunContext.jsx'
import { ToastProvider } from './context/ToastContext.jsx'

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

function Probe() {
  const { globalData, moeData, liveStatus } = useRun()
  return (
    <div>
      <output data-testid="row-count">{globalData.length}</output>
      <output data-testid="moe-count">{moeData.length}</output>
      <output data-testid="live-status">{liveStatus}</output>
    </div>
  )
}

describe('RunProvider live updates', () => {
  beforeEach(() => {
    MockWebSocket.instances = []
    globalThis.WebSocket = MockWebSocket
    globalThis.fetch = vi.fn(async (url) => {
      const path = new URL(url, window.location.origin).pathname
      const payloads = {
        '/api/meta': { trainscope_config: { run_name: 'live-test' } },
        '/api/global': [],
        '/api/layers': [],
        '/api/spikes': [],
        '/api/moe': [],
        '/api/runs': [],
      }
      return {
        ok: true,
        json: async () => payloads[path],
      }
    })
  })

  afterEach(() => {
    delete globalThis.WebSocket
    delete globalThis.fetch
  })

  it('renders rows received after an initially empty REST response', async () => {
    render(
      <ToastProvider>
        <RunProvider>
          <Probe />
        </RunProvider>
      </ToastProvider>
    )

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    await waitFor(() => expect(screen.getByTestId('live-status')).toHaveTextContent('connected'))

    await act(async () => {
      MockWebSocket.instances[0].onmessage({
        data: JSON.stringify({
          type: 'global',
          payload: [{ step: 4, loss: 1.25 }],
        }),
      })
    })

    expect(screen.getByTestId('row-count')).toHaveTextContent('1')

    await act(async () => {
      MockWebSocket.instances[0].onmessage({
        data: JSON.stringify({
          type: 'global_delta',
          payload: [{ step: 5, loss: 1.5 }],
        }),
      })
    })

    expect(screen.getByTestId('row-count')).toHaveTextContent('2')
  })

  it('groups contiguous spike steps into one notification', async () => {
    render(
      <ToastProvider>
        <RunProvider>
          <Probe />
        </RunProvider>
      </ToastProvider>
    )

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    await waitFor(() => expect(screen.getByTestId('live-status')).toHaveTextContent('connected'))

    await act(async () => {
      for (const step of [80, 81, 85, 91]) {
        MockWebSocket.instances[0].onmessage({
          data: JSON.stringify({ type: 'spike', payload: { step } }),
        })
      }
    })

    expect(screen.getAllByText('Spike event detected')).toHaveLength(2)
  })

  it('appends moe rows from the live WebSocket', async () => {
    render(
      <ToastProvider>
        <RunProvider>
          <Probe />
        </RunProvider>
      </ToastProvider>
    )

    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    await waitFor(() => expect(screen.getByTestId('live-status')).toHaveTextContent('connected'))

    await act(async () => {
      MockWebSocket.instances[0].onmessage({
        data: JSON.stringify({
          type: 'moe',
          payload: [{ step: 0, block: 'blocks.0.router', shares: [0.25, 0.25, 0.25, 0.25] }],
        }),
      })
    })
    expect(screen.getByTestId('moe-count')).toHaveTextContent('1')

    await act(async () => {
      MockWebSocket.instances[0].onmessage({
        data: JSON.stringify({
          type: 'moe_delta',
          payload: [{ step: 1, block: 'blocks.0.router', shares: [0.9, 0.03, 0.03, 0.04] }],
        }),
      })
    })
    expect(screen.getByTestId('moe-count')).toHaveTextContent('2')
  })
})
