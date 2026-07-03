import { useEffect, useRef } from 'react'

const WS_URL = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`

export const WS_STATUS = {
  CONNECTING: 'connecting',
  CONNECTED: 'connected',
  DISCONNECTED: 'disconnected',
  UNAVAILABLE: 'unavailable',
}

const MAX_RECONNECT_ATTEMPTS = 5

/**
 * Subscribe to the TrainScope WebSocket stream.
 *
 * The hook reconnects with exponential backoff and finally reports
 * `unavailable` so callers can fall back to REST polling.
 */
export function useWebSocket({ onMessage, onStatusChange, enabled = true }) {
  const onMessageRef = useRef(onMessage)
  const onStatusChangeRef = useRef(onStatusChange)

  onMessageRef.current = onMessage
  onStatusChangeRef.current = onStatusChange

  useEffect(() => {
    if (!enabled) return undefined

    let ws
    let reconnectTimer
    let attempt = 0
    let active = true

    function setStatus(status) {
      if (active) {
        onStatusChangeRef.current?.(status)
      }
    }

    function connect() {
      if (!active) return

      setStatus(WS_STATUS.CONNECTING)

      try {
        ws = new WebSocket(WS_URL)
      } catch {
        setStatus(WS_STATUS.UNAVAILABLE)
        return
      }

      ws.onopen = () => {
        attempt = 0
        setStatus(WS_STATUS.CONNECTED)
      }

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)
          onMessageRef.current?.(message)
        } catch {
          // Ignore malformed messages.
        }
      }

      ws.onerror = () => {
        setStatus(WS_STATUS.DISCONNECTED)
      }

      ws.onclose = () => {
        setStatus(WS_STATUS.DISCONNECTED)
        if (!active) return

        if (attempt < MAX_RECONNECT_ATTEMPTS) {
          const delay = Math.min(1000 * 2 ** attempt, 30000)
          reconnectTimer = window.setTimeout(() => {
            attempt += 1
            connect()
          }, delay)
        } else {
          setStatus(WS_STATUS.UNAVAILABLE)
        }
      }
    }

    connect()

    return () => {
      active = false
      window.clearTimeout(reconnectTimer)
      if (ws) {
        ws.onopen = null
        ws.onclose = null
        ws.onerror = null
        ws.onmessage = null
        ws.close()
      }
    }
  }, [enabled])
}
