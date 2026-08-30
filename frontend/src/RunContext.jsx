import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import {
  fetchMeta,
  fetchGlobal,
  fetchLayers,
  fetchSpikes,
  fetchMoe,
  abortAllRequests,
  fetchRuns,
  selectRun,
} from './api.js'
import { useWebSocket } from './ws.js'
import { useToast } from './context/ToastContext.jsx'

import { groupSpikes } from './utils/spikeCluster.js'

const RunContext = createContext(null)
const LIVE_RECONCILE_INTERVAL_MS = 3000

function sameRows(previous, next) {
  if (previous.length !== next.length) return false
  return previous.every((row, index) => {
    const other = next[index]
    return (
      row.step === other?.step &&
      row.loss === other?.loss &&
      row.grad_norm_before_clip === other?.grad_norm_before_clip &&
      row.is_spike === other?.is_spike
    )
  })
}

function sameMoeRows(previous, next) {
  if (previous.length !== next.length) return false
  return previous.every((row, index) => {
    const other = next[index]
    return (
      row.step === other?.step &&
      row.block === other?.block &&
      JSON.stringify(row.shares ?? null) === JSON.stringify(other?.shares ?? null)
    )
  })
}

function sameValues(previous, next) {
  return previous.length === next.length && previous.every((value, index) => value === next[index])
}

function sameSpikes(previous, next) {
  return (
    previous.length === next.length &&
    previous.every((spike, index) => spike.step === next[index]?.step)
  )
}

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
  const [moeData, setMoeData] = useState([])
  const [runs, setRuns] = useState([])
  const [activeRunName, setActiveRunName] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [liveStatus, setLiveStatus] = useState('connecting')
  const [lastUpdatedAt, setLastUpdatedAt] = useState(null)

  const toastedStepsRef = useRef(new Set())
  const lastNotifiedSpikeStepRef = useRef(null)

  const load = useCallback(async ({ silent = false } = {}) => {
    if (!silent) {
      setLoading(true)
    }
    setError(null)

    try {
      const [metaData, gData, names, spikeList, runList, moeList] = await Promise.all([
        silent ? Promise.resolve(null) : fetchMeta(),
        fetchGlobal(),
        fetchLayers(),
        fetchSpikes(),
        fetchRuns(),
        fetchMoe(),
      ])
      if (!silent) setMeta(metaData)
      const nextGlobalData = Array.isArray(gData) ? gData : []
      const nextLayerNames = Array.isArray(names) ? names : []
      const nextSpikes = Array.isArray(spikeList) ? spikeList : []
      const nextRuns = Array.isArray(runList) ? runList : []
      const nextMoeData = Array.isArray(moeList) ? moeList : []

      // An Arrow file can be momentarily unavailable while it is being
      // replaced. Keep the last good snapshot instead of turning the live
      // dashboard into an empty state during that short window.
      setGlobalData((previous) => {
        if (previous.length === 0 || nextGlobalData.length >= previous.length) {
          return sameRows(previous, nextGlobalData) ? previous : nextGlobalData
        }

        // A response that started before the latest socket delta can be
        // shorter than the current state. Merge it without rolling the chart
        // back to an older snapshot.
        const rowsByStep = new Map(previous.map((row) => [row.step, row]))
        for (const row of nextGlobalData) {
          if (row && row.step != null) rowsByStep.set(row.step, row)
        }
        const mergedRows = [...rowsByStep.values()].sort((a, b) => a.step - b.step)
        return sameRows(previous, mergedRows) ? previous : mergedRows
      })
      setLayerNames((previous) => {
        if (previous.length > 0 && nextLayerNames.length === 0) return previous
        return sameValues(previous, nextLayerNames) ? previous : nextLayerNames
      })
      setSpikes((previous) => {
        if (previous.length > 0 && nextSpikes.length === 0) return previous
        return sameSpikes(previous, nextSpikes) ? previous : nextSpikes
      })
      setMoeData((previous) => {
        if (previous.length > 0 && nextMoeData.length === 0) return previous
        return sameMoeRows(previous, nextMoeData) ? previous : nextMoeData
      })
      setRuns((previous) =>
        sameValues(
          previous.map((r) => r.name),
          nextRuns.map((r) => r.name)
        )
          ? previous
          : nextRuns
      )
      if (!silent) setLastUpdatedAt(Date.now())
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
          setLastUpdatedAt(Date.now())
          setGlobalData((previous) => {
            const nextRows = Array.isArray(message.payload) ? message.payload : []
            if (previous.length > 0 && nextRows.length === 0) return previous
            return sameRows(previous, nextRows) ? previous : nextRows
          })
          break
        case 'global_delta':
          setLastUpdatedAt(Date.now())
          setGlobalData((prev) => {
            const newRows = Array.isArray(message.payload) ? message.payload : []
            if (newRows.length === 0) return prev
            const rowsByStep = new Map(prev.map((row) => [row.step, row]))
            for (const row of newRows) {
              if (row && row.step != null) rowsByStep.set(row.step, row)
            }
            return [...rowsByStep.values()].sort((a, b) => a.step - b.step)
          })
          break
        case 'layers':
          setLayerNames((previous) => {
            const nextLayers = Array.isArray(message.payload) ? message.payload : []
            return sameValues(previous, nextLayers) ? previous : nextLayers
          })
          break
        case 'moe':
          setLastUpdatedAt(Date.now())
          setMoeData((previous) => {
            const nextRows = Array.isArray(message.payload) ? message.payload : []
            if (previous.length > 0 && nextRows.length === 0) return previous
            return sameMoeRows(previous, nextRows) ? previous : nextRows
          })
          break
        case 'moe_delta':
          setLastUpdatedAt(Date.now())
          setMoeData((previous) => {
            const newRows = Array.isArray(message.payload) ? message.payload : []
            if (newRows.length === 0) return previous
            return sameMoeRows(previous, [...previous, ...newRows]) ? previous : [
              ...previous,
              ...newRows,
            ]
          })
          break
        case 'spike': {
          const spike = message.payload
          if (!spike || typeof spike !== 'object') break
          setLastUpdatedAt(Date.now())
          setSpikes((prev) => {
            if (prev.some((s) => s.step === spike.step)) return prev
            return [...prev, spike].sort((a, b) => a.step - b.step)
          })
          // The server emits one message per detected step. Notify once for a
          // contiguous anomaly window instead of flooding the toast stack.
          if (!toastedStepsRef.current.has(spike.step)) {
            toastedStepsRef.current.add(spike.step)
            const previousStep = lastNotifiedSpikeStepRef.current
            const startsNewEvent = previousStep == null || spike.step - previousStep > 5
            lastNotifiedSpikeStepRef.current = Math.max(previousStep ?? spike.step, spike.step)

            if (!startsNewEvent) break

            addToast({
              title: 'Spike event detected',
              message: `Anomaly window starts at step ${spike.step}`,
              variant: 'danger',
            })
          }
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
    // Connect immediately so a run that starts with an empty Arrow file can
    // still push its first batch into the UI without requiring a refresh.
    // resetKey forces a fresh connection when the active run changes.
    enabled: true,
    resetKey: activeRunName,
  })

  const switchRun = useCallback(
    async (name) => {
      try {
        const summary = await selectRun(name)
        setActiveRunName(summary?.name ?? name)
        // Abort in-flight requests from the previous run, then reload.
        abortAllRequests()
        await load()
        setRuns((previous) =>
          previous.map((r) => ({ ...r, is_active: r.name === (summary?.name ?? name) }))
        )
        return summary
      } catch (err) {
        setError(err?.message || 'Failed to switch run.')
        return null
      }
    },
    [load]
  )

  // Reconcile periodically even while connected so a missed delta or a
  // reconnect cannot leave the chart stuck on the first streamed batch.
  const poll = useCallback(() => {
    load({ silent: true })
  }, [load])

  useEffect(() => {
    if (loading) return undefined
    const intervalMs = liveStatus === 'connected' ? LIVE_RECONCILE_INTERVAL_MS : 1500
    const interval = window.setInterval(poll, intervalMs)
    return () => window.clearInterval(interval)
  }, [liveStatus, loading, poll])

  const spikeEvents = useMemo(() => groupSpikes(globalData, spikes), [globalData, spikes])

  const value = useMemo(
    () => ({
      meta,
      globalData,
      layerNames,
      spikes,
      spikeEvents,
      moeData,
      runs,
      activeRunName,
      switchRun,
      loading,
      error,
      refresh: () => load(),
      isReady: !loading && !error && meta != null,
      liveStatus,
      lastUpdatedAt,
    }),
    [
      meta,
      globalData,
      layerNames,
      spikes,
      spikeEvents,
      moeData,
      runs,
      activeRunName,
      switchRun,
      loading,
      error,
      load,
      liveStatus,
      lastUpdatedAt,
    ]
  )

  return <RunContext.Provider value={value}>{children}</RunContext.Provider>
}
