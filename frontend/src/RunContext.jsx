import { createContext, useContext, useEffect, useMemo, useState, useCallback } from 'react'
import { fetchMeta, fetchGlobal, fetchLayers, fetchSpikes, abortAllRequests } from './api.js'

const RunContext = createContext(null)

export function useRun() {
  const ctx = useContext(RunContext)
  if (!ctx) {
    throw new Error('useRun must be used within a RunProvider')
  }
  return ctx
}

/**
 * Provide run-level data (meta, global rows, layer names, spikes) to the rest
 * of the application. Data is fetched once when the provider mounts and shared
 * across all views so each tab does not issue redundant requests.
 */
export function RunProvider({ children }) {
  const [meta, setMeta] = useState(null)
  const [globalData, setGlobalData] = useState([])
  const [layerNames, setLayerNames] = useState([])
  const [spikes, setSpikes] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const load = useCallback(async () => {
    setLoading(true)
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
    } catch (err) {
      if (err.name !== 'AbortError') {
        setError(err?.message || 'Failed to load run data.')
      }
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    load()
    return () => {
      // Abort any in-flight requests when the provider unmounts.
      abortAllRequests()
    }
  }, [load])

  const value = useMemo(
    () => ({
      meta,
      globalData,
      layerNames,
      spikes,
      loading,
      error,
      refresh: load,
      isReady: !loading && !error && meta != null,
    }),
    [meta, globalData, layerNames, spikes, loading, error, load]
  )

  return <RunContext.Provider value={value}>{children}</RunContext.Provider>
}
