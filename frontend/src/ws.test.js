import { renderHook, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { WS_STATUS, useWebSocket } from './ws.js'

class MockWebSocket {
  static instances = []

  constructor(url) {
    this.url = url
    this.readyState = 0
    MockWebSocket.instances.push(this)
  }

  send() {}

  close() {
    this.readyState = 3
    if (this.onclose) this.onclose()
  }
}

describe('useWebSocket', () => {
  beforeEach(() => {
    MockWebSocket.instances = []
    globalThis.WebSocket = MockWebSocket
  })

  afterEach(() => {
    delete globalThis.WebSocket
  })

  it('reports connecting then connected', async () => {
    const onStatusChange = vi.fn()
    renderHook(() => useWebSocket({ onStatusChange, enabled: true }))

    expect(MockWebSocket.instances).toHaveLength(1)
    expect(onStatusChange).toHaveBeenCalledWith(WS_STATUS.CONNECTING)

    MockWebSocket.instances[0].onopen()
    await waitFor(() => expect(onStatusChange).toHaveBeenCalledWith(WS_STATUS.CONNECTED))
  })
})
