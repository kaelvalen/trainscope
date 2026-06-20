import { useState, useEffect, useMemo } from 'react'
import { useRun } from '../RunContext.jsx'
import { fetchLayersRanked, fetchLayer } from '../api.js'
import { spikeShape, scrubLineShape, truncateLayerName, CHART_COLORS } from '../theme.js'
import Chart from '../components/Chart.jsx'
import StepScrubber from '../components/StepScrubber.jsx'
import LoadingSpinner from '../components/LoadingSpinner.jsx'
import ErrorMessage from '../components/ErrorMessage.jsx'
import EmptyState from '../components/EmptyState.jsx'

function buildSpikeShapes(rows) {
  return rows.filter((r) => r.is_spike).map((r) => spikeShape(r.step))
}

export default function Timeline() {
  const { globalData, layerNames } = useRun()
  const [layerGradNorms, setLayerGradNorms] = useState({})
  const [scrubStep, setScrubStep] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (globalData.length > 0) {
      setScrubStep(globalData[0].step)
    }
  }, [globalData])

  useEffect(() => {
    if (!layerNames.length) {
      setLoading(false)
      return
    }

    let cancelled = false
    setLoading(true)
    setError(null)

    async function load() {
      try {
        const ranked = await fetchLayersRanked(8).catch(() => [])
        const sample = ranked.length > 0 ? ranked : layerNames.slice(0, 8)
        const layerDataMap = {}
        await Promise.all(
          sample.map(async (name) => {
            const rows = await fetchLayer(name)
            layerDataMap[name] = rows
          })
        )
        if (!cancelled) setLayerGradNorms(layerDataMap)
      } catch (err) {
        if (!cancelled) setError(err?.message || 'Failed to load layer data.')
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    load()
    return () => {
      cancelled = true
    }
  }, [layerNames])

  const steps = useMemo(() => globalData.map((r) => r.step), [globalData])
  const losses = useMemo(() => globalData.map((r) => r.loss), [globalData])
  const gradNorms = useMemo(() => globalData.map((r) => r.grad_norm_before_clip), [globalData])
  const spikeShapes = useMemo(() => buildSpikeShapes(globalData), [globalData])
  const scrubShapes = useMemo(
    () => (scrubStep != null ? [scrubLineShape(scrubStep)] : []),
    [scrubStep]
  )

  const scrubRow = useMemo(() => {
    if (!globalData.length) return null
    return globalData.find((r) => r.step === scrubStep) || globalData[0]
  }, [globalData, scrubStep])

  const layerGradTraces = useMemo(
    () =>
      Object.entries(layerGradNorms).map(([name, rows]) => ({
        x: rows.map((r) => r.step),
        y: rows.map((r) => r.grad_l2_norm),
        type: 'scatter',
        mode: 'lines',
        name: truncateLayerName(name, 30),
        line: { width: 1 },
      })),
    [layerGradNorms]
  )

  if (loading) {
    return <LoadingSpinner message="Loading timeline data…" />
  }

  if (globalData.length === 0) {
    return <EmptyState icon="📉">No global data found. Has a training run completed?</EmptyState>
  }

  return (
    <div>
      <ErrorMessage message={error} />

      <Chart
        data={[
          {
            x: steps,
            y: losses,
            type: 'scatter',
            mode: 'lines',
            name: 'Loss',
            line: { color: CHART_COLORS.loss, width: 1.5 },
          },
        ]}
        layout={{
          title: { text: 'Training Loss', font: { size: 14 } },
          height: 280,
          shapes: [...spikeShapes, ...scrubShapes],
        }}
      />

      <Chart
        data={[
          {
            x: steps,
            y: gradNorms,
            type: 'scatter',
            mode: 'lines',
            name: 'Grad Norm',
            line: { color: CHART_COLORS.gradNorm, width: 1.5 },
          },
        ]}
        layout={{
          title: { text: 'Global Gradient Norm (before clip)', font: { size: 14 } },
          height: 240,
          shapes: [...spikeShapes, ...scrubShapes],
        }}
      />

      <StepScrubber steps={steps} value={scrubStep} onChange={setScrubStep} />

      {scrubRow && (
        <div className="ts-panel" style={{ marginTop: '12px', fontSize: '12px' }}>
          <strong>Step {scrubRow.step}</strong>
          {' — '}
          Loss: <span style={{ color: 'var(--accent)' }}>{scrubRow.loss?.toFixed(4)}</span>
          {' | '}
          Grad:{' '}
          <span style={{ color: 'var(--success)' }}>
            {scrubRow.grad_norm_before_clip?.toFixed(4)}
          </span>
          {scrubRow.is_spike && (
            <span style={{ color: 'var(--danger)', marginLeft: 8, fontWeight: 600 }}>SPIKE</span>
          )}
        </div>
      )}

      {layerGradTraces.length > 0 && (
        <Chart
          data={layerGradTraces}
          layout={{
            title: {
              text: 'Per-Layer Gradient Norms (top 8 by grad variance)',
              font: { size: 14 },
            },
            height: 300,
            showlegend: true,
            legend: { font: { size: 10 } },
            shapes: scrubShapes,
          }}
        />
      )}
    </div>
  )
}
