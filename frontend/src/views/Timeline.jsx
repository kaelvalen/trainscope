import { useState, useEffect, useMemo } from 'react'
import { useRun } from '../RunContext.jsx'
import { fetchLayersRanked, fetchLayer } from '../api.js'
import { scrubLineShape, truncateLayerName, CHART_COLORS, warmupBandShape, warmupAnnotation } from '../theme.js'
import { buildSpikeClusterShapes, buildSpikeClusterAnnotations } from '../utils/spikeCluster.js'
import { computeWarmupBand } from '../utils/warmup.js'
import Chart from '../components/Chart.jsx'
import StepScrubber from '../components/StepScrubber.jsx'
import ErrorMessage from '../components/ErrorMessage.jsx'
import EmptyState from '../components/EmptyState.jsx'
import { Skeleton } from '../components/ui/Skeleton.jsx'
import { StatCard } from '../components/ui/StatCard.jsx'
import { ChartCard } from '../components/ui/ChartCard.jsx'
import { Card, CardContent } from '../components/ui/Card.jsx'
import { Badge } from '../components/ui/Badge.jsx'
import { Activity, BarChart3, TrendingDown, Activity as Pulse, Zap } from 'lucide-react'

function buildZoomLayout(extra = {}) {
  return {
    dragmode: 'zoom',
    xaxis: { rangeslider: { visible: false }, ...extra.xaxis },
    ...extra,
  }
}

function AnomalyRail({ events, onSelect }) {
  if (events.length === 0) return null

  const visibleEvents = events.slice(-6)

  return (
    <Card className="anomaly-rail">
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold text-foreground">Detected anomaly windows</h2>
          <p className="chart-card__description">
            Select a window to move the scrubber to its first detected step.
          </p>
        </div>
        <Badge variant="danger">
          {events.length} {events.length === 1 ? 'event' : 'events'}
        </Badge>
      </div>
      <div className="anomaly-rail__items">
        {visibleEvents.map((event, index) => {
          const rangeLabel =
            event.startStep === event.endStep
              ? `Step ${event.startStep}`
              : `Steps ${event.startStep}–${event.endStep}`
          const leadLabel =
            event.earlyWarningWindow > 0
              ? `+${event.earlyWarningWindow} to peak`
              : `Peak ${event.peakStep}`

          return (
            <button
              key={event.id}
              type="button"
              className="anomaly-chip"
              onClick={() => onSelect(event.startStep)}
              aria-label={`Inspect anomaly ${rangeLabel}`}
            >
              <span className="anomaly-chip__index">
                {String(events.length - visibleEvents.length + index + 1).padStart(2, '0')}
              </span>
              <span className="anomaly-chip__body">
                <span className="anomaly-chip__range">{rangeLabel}</span>
                <span className="anomaly-chip__meta">
                  {event.count} detection{event.count === 1 ? '' : 's'} · peak {event.peakStep}
                </span>
              </span>
              <span className="anomaly-chip__lead">{leadLabel}</span>
            </button>
          )
        })}
      </div>
    </Card>
  )
}

export default function Timeline() {
  const { meta, globalData, layerNames, spikeEvents } = useRun()
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

  const spikeShapes = useMemo(() => buildSpikeClusterShapes(spikeEvents), [spikeEvents])
  const spikeAnnotations = useMemo(() => buildSpikeClusterAnnotations(spikeEvents), [spikeEvents])

  // Detector warmup: until min_observations losses have been seen the detector
  // cannot report spikes, so early explosions stay invisible. Shade that band.
  const warmupBand = useMemo(() => computeWarmupBand(meta, globalData), [meta, globalData])
  const warmupShapes = useMemo(
    () => (warmupBand ? [warmupBandShape(warmupBand.startStep, warmupBand.endStep)] : []),
    [warmupBand]
  )
  const warmupAnnotations = useMemo(
    () =>
      warmupBand
        ? [warmupAnnotation(warmupBand.startStep, warmupBand.endStep)]
        : [],
    [warmupBand]
  )

  const scrubShapes = useMemo(
    () => (scrubStep != null ? [scrubLineShape(scrubStep)] : []),
    [scrubStep]
  )

  const latestRow = globalData[globalData.length - 1] || null
  const rawSpikeCount = useMemo(() => globalData.filter((r) => r.is_spike).length, [globalData])

  const spikeStatCardProps = useMemo(() => {
    const count = spikeEvents.length
    if (count === 0) {
      return { value: '0', subtitle: 'No anomalies' }
    }
    if (count === 1) {
      const ev = spikeEvents[0]
      if (ev.earlyWarningWindow > 0) {
        return {
          value: '1 event',
          subtitle: `${ev.earlyWarningWindow}-step detection lead (Step ${ev.startStep} → ${ev.endStep})`,
        }
      }
      return {
        value: '1 event',
        subtitle: `Step ${ev.startStep}`,
      }
    }
    return {
      value: `${count} events`,
      subtitle: `${rawSpikeCount} raw detections grouped`,
    }
  }, [spikeEvents, rawSpikeCount])

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
    return (
      <EmptyState icon={<BarChart3 className="h-5 w-5" />}>
        No global data found. Has a training run completed?
      </EmptyState>
    )
  }

  return (
    <div className="space-y-5">
      <div className="section-row">
        <div>
          <p className="section-label">Run snapshot</p>
          <p className="section-note">Latest recorded signals across the training window</p>
        </div>
        <span className="section-note">{globalData.length.toLocaleString()} steps recorded</span>
      </div>

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
        <StatCard
          label="Spikes"
          value={spikeStatCardProps.value}
          subtitle={spikeStatCardProps.subtitle}
          icon={Zap}
        />
      </div>

      <AnomalyRail events={spikeEvents} onSelect={setScrubStep} />

      <ErrorMessage message={error} />

      <div className="grid grid-cols-1 gap-5 2xl:grid-cols-2">
        <ChartCard
          title="Training loss"
          description="Loss trajectory with grouped anomaly windows and the current scrub position."
        >
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
              height: 320,
              shapes: [...warmupShapes, ...spikeShapes, ...scrubShapes],
              annotations: [...warmupAnnotations, ...spikeAnnotations],
              uirevision: 'timeline-loss',
            })}
            config={{ scrollZoom: true }}
          />
        </ChartCard>

        <ChartCard
          title="Global gradient norm"
          description="Pre-clip gradient magnitude makes sudden update instability easier to isolate."
        >
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
              height: 280,
              shapes: [...warmupShapes, ...spikeShapes, ...scrubShapes],
              uirevision: 'timeline-grad',
            })}
            config={{ scrollZoom: true }}
          />
        </ChartCard>
      </div>

      <StepScrubber steps={steps} value={scrubStep} onChange={setScrubStep} />

      {scrubRow && (
        <Card className="scrub-readout">
          <CardContent className="text-sm">
            <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
              <strong className="text-foreground">Step {scrubRow.step}</strong>
              <span>
                Loss <span className="font-semibold text-accent">{scrubRow.loss?.toFixed(4)}</span>
              </span>
              <span>
                Grad{' '}
                <span className="font-semibold text-success">
                  {scrubRow.grad_norm_before_clip?.toFixed(4)}
                </span>
              </span>
              {scrubRow.is_spike && (
                <span className="font-semibold text-danger">SPIKE DETECTED</span>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {layerLoading ? (
        <Skeleton className="h-72" />
      ) : layerGradTraces.length > 0 ? (
        <ChartCard
          title="Per-layer gradient norms"
          description="Top eight layers ranked by gradient variance."
        >
          <Chart
            data={layerGradTraces}
            layout={buildZoomLayout({
              height: 340,
              showlegend: true,
              legend: { font: { size: 10 } },
              shapes: [...warmupShapes, ...scrubShapes],
              uirevision: 'timeline-layers',
            })}
            config={{ scrollZoom: true }}
          />
        </ChartCard>
      ) : null}
    </div>
  )
}
