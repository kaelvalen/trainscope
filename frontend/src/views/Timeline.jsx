import { useState, useEffect, useMemo } from 'react'
import { useRun } from '../RunContext.jsx'
import { fetchLayersRanked, fetchLayer } from '../api.js'
import { spikeShape, scrubLineShape, truncateLayerName, CHART_COLORS } from '../theme.js'
import Chart from '../components/Chart.jsx'
import StepScrubber from '../components/StepScrubber.jsx'
import ErrorMessage from '../components/ErrorMessage.jsx'
import EmptyState from '../components/EmptyState.jsx'
import { Skeleton } from '../components/ui/Skeleton.jsx'
import { StatCard } from '../components/ui/StatCard.jsx'
import { Card, CardContent } from '../components/ui/Card.jsx'
import { Activity, TrendingDown, Activity as Pulse, Zap } from 'lucide-react'

function buildSpikeAnnotations(rows) {
  return rows
    .filter((r) => r.is_spike)
    .map((r) => ({
      x: r.step,
      y: 1,
      xref: 'x',
      yref: 'paper',
      text: 'spike',
      showarrow: false,
      font: { color: CHART_COLORS.spike, size: 10 },
      yanchor: 'top',
    }))
}

function buildZoomLayout(extra = {}) {
  return {
    dragmode: 'zoom',
    xaxis: { rangeslider: { visible: false }, ...extra.xaxis },
    ...extra,
  }
}

export default function Timeline() {
  const { globalData, layerNames } = useRun()
  const [layerGradNorms, setLayerGradNorms] = useState({})
  const [scrubStep, setScrubStep] = useState(0)
  const [layerLoading, setLayerLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (globalData.length > 0) {
      setScrubStep(globalData[globalData.length - 1].step)
    }
  }, [globalData])

  useEffect(() => {
    if (!layerNames.length) {
      setLayerLoading(false)
      return
    }

    let cancelled = false
    setLayerLoading(true)
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
        if (!cancelled) setLayerLoading(false)
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
  const spikeShapes = useMemo(() => {
    const shapes = globalData.filter((r) => r.is_spike).map((r) => spikeShape(r.step))
    return shapes
  }, [globalData])
  const spikeAnnotations = useMemo(() => buildSpikeAnnotations(globalData), [globalData])
  const scrubShapes = useMemo(
    () => (scrubStep != null ? [scrubLineShape(scrubStep)] : []),
    [scrubStep]
  )

  const latestRow = globalData[globalData.length - 1] || null
  const spikeCount = useMemo(() => globalData.filter((r) => r.is_spike).length, [globalData])

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

  if (globalData.length === 0) {
    return <EmptyState icon="📉">No global data found. Has a training run completed?</EmptyState>
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Steps"
          value={globalData.length}
          subtitle={latestRow ? `up to step ${latestRow.step}` : ''}
          icon={Activity}
        />
        <StatCard
          label="Latest Loss"
          value={latestRow?.loss?.toFixed(4) ?? '—'}
          icon={TrendingDown}
        />
        <StatCard
          label="Latest Grad Norm"
          value={latestRow?.grad_norm_before_clip?.toFixed(4) ?? '—'}
          icon={Pulse}
        />
        <StatCard label="Spikes" value={spikeCount} icon={Zap} />
      </div>

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
        layout={buildZoomLayout({
          title: { text: 'Training Loss', font: { size: 14 } },
          height: 320,
          shapes: [...spikeShapes, ...scrubShapes],
          annotations: spikeAnnotations,
          uirevision: 'timeline-loss',
        })}
        config={{ scrollZoom: true }}
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
        layout={buildZoomLayout({
          title: { text: 'Global Gradient Norm (before clip)', font: { size: 14 } },
          height: 280,
          shapes: [...spikeShapes, ...scrubShapes],
          uirevision: 'timeline-grad',
        })}
        config={{ scrollZoom: true }}
      />

      <StepScrubber steps={steps} value={scrubStep} onChange={setScrubStep} />

      {scrubRow && (
        <Card>
          <CardContent className="text-sm">
            <strong className="text-foreground">Step {scrubRow.step}</strong>
            {' — '}
            Loss: <span className="text-accent">{scrubRow.loss?.toFixed(4)}</span>
            {' | '}
            Grad: <span className="text-success">{scrubRow.grad_norm_before_clip?.toFixed(4)}</span>
            {scrubRow.is_spike && <span className="ml-2 font-semibold text-danger">SPIKE</span>}
          </CardContent>
        </Card>
      )}

      {layerLoading ? (
        <Skeleton className="h-72" />
      ) : layerGradTraces.length > 0 ? (
        <Chart
          data={layerGradTraces}
          layout={buildZoomLayout({
            title: {
              text: 'Per-Layer Gradient Norms (top 8 by grad variance)',
              font: { size: 14 },
            },
            height: 340,
            showlegend: true,
            legend: { font: { size: 10 } },
            shapes: scrubShapes,
            uirevision: 'timeline-layers',
          })}
          config={{ scrollZoom: true }}
        />
      ) : null}
    </div>
  )
}
