import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import { fetchMeta, fetchGlobal, fetchLayers, fetchSpikes, abortAllRequests } from './api.js'
import { useWebSocket } from './ws.js'
import { useToast } from './context/ToastContext.jsx'

const RunContext = createContext(null)

export function useRun() {
  const ctx = useContext(RunContext)
  if (!ctx) {
    throw new Error('useRun must be used within a RunProvider')
  }
  return ctx
}

/**
 * Provide run-level data to the rest of the application.
 *
 * Data is fetched once on mount, then kept up-to-date via WebSocket. If the
 * WebSocket is unavailable, the provider silently polls the REST endpoints.
 */
export function RunProvider({ children }) {
  const { addToast } = useToast()

  const [meta, setMeta] = useState(null)
  const [globalData, setGlobalData] = useState([])
  const [layerNames, setLayerNames] = useState([])
  const [spikes, setSpikes] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [liveStatus, setLiveStatus] = useState('connecting')

  const initialLoadComplete = useRef(false)

  const load = useCallback(async ({ silent = false } = {}) => {
    if (!silent) {
      setLoading(true)
    }
    setError(null)

    try {
      const [metaData, gData, names, spikeList] = await Promise.all([
        fetchMeta(),
        fetchGlobal(),
        fetchLayers(),
        fetchSpikes(),
      ])
      setMeta(metaData)
      setGlobalData(Array.isArray(gData) ? gData : [])
      setLayerNames(Array.isArray(names) ? names : [])
      setSpikes(Array.isArray(spikeList) ? spikeList : [])
      initialLoadComplete.current = true
    } catch (err) {
      if (err.name !== 'AbortError') {
        setError(err?.message || 'Failed to load run data.')
      }
    } finally {
      if (!silent) {
        setLoading(false)
      }
    }
  }, [])

  useEffect(() => {
    load()
    return () => {
      abortAllRequests()
    }
  }, [load])

  const handleWebSocketMessage = useCallback(
    (message) => {
      if (!message || typeof message !== 'object') return

      switch (message.type) {
        case 'meta':
          setMeta(message.payload)
          break
        case 'global':
          setGlobalData(Array.isArray(message.payload) ? message.payload : [])
          break
        case 'layers':
          setLayerNames(Array.isArray(message.payload) ? message.payload : [])
          break
        case 'spike': {
          const spike = message.payload
          if (!spike || typeof spike !== 'object') break
          setSpikes((prev) => {
            if (prev.some((s) => s.step === spike.step)) return prev
            return [...prev, spike].sort((a, b) => a.step - b.step)
          })
          addToast({
            title: 'Spike detected',
            message: `Anomaly at step ${spike.step}`,
            variant: 'danger',
          })
          break
        }
        default:
          break
      }
    },
    [addToast]
  )

  useWebSocket({
    onMessage: handleWebSocketMessage,
    onStatusChange: setLiveStatus,
    enabled: initialLoadComplete.current || !loading,
  })

  // Fallback to REST polling when the WebSocket is unavailable.
  const poll = useCallback(() => {
    load({ silent: true })
  }, [load])

  useEffect(() => {
    if (liveStatus !== 'unavailable') return undefined
    const interval = window.setInterval(poll, 3000)
    return () => window.clearInterval(interval)
  }, [liveStatus, poll])

  const value = useMemo(
    () => ({
      meta,
      globalData,
      layerNames,
      spikes,
      loading,
      error,
      refresh: () => load(),
      isReady: !loading && !error && meta != null,
      liveStatus,
    }),
    [meta, globalData, layerNames, spikes, loading, error, load, liveStatus]
  )

  return <RunContext.Provider value={value}>{children}</RunContext.Provider>
}
