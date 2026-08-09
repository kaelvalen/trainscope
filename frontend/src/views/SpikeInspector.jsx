import { useState, useEffect, useMemo } from 'react'
import { useRun } from '../RunContext.jsx'
import { fetchSpike, fetchSpikeLayerNames, fetchSpikeLayer } from '../api.js'
import { spikeShape, CHART_COLORS } from '../theme.js'
import { diagnoseSpike } from '../utils/diagnosis.js'
import Chart from '../components/Chart.jsx'
import LayerSelect from '../components/LayerSelect.jsx'
import ErrorMessage from '../components/ErrorMessage.jsx'
import EmptyState from '../components/EmptyState.jsx'
import { Skeleton } from '../components/ui/Skeleton.jsx'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card.jsx'
import { ChartCard } from '../components/ui/ChartCard.jsx'
import { Badge } from '../components/ui/Badge.jsx'
import { StatCard } from '../components/ui/StatCard.jsx'
import { Zap, TrendingDown, Activity } from 'lucide-react'

export default function SpikeInspector() {
  const { spikes, spikeEvents } = useRun()
  const [selectedSpike, setSelectedSpike] = useState(null)
  const [globalWindow, setGlobalWindow] = useState([])
  const [layerNames, setLayerNames] = useState([])
  const [selectedLayer, setSelectedLayer] = useState('')
  const [layerWindow, setLayerWindow] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (spikes.length > 0 && selectedSpike == null) {
      setSelectedSpike(spikes[0].step)
    }
  }, [spikes, selectedSpike])

  useEffect(() => {
    if (selectedSpike == null) return

    let cancelled = false
    setLoading(true)
    setError(null)

    Promise.all([fetchSpike(selectedSpike), fetchSpikeLayerNames(selectedSpike)])
      .then(([gRows, lNames]) => {
        if (cancelled) return
        setGlobalWindow(gRows)
        setLayerNames(lNames)
        if (lNames.length > 0) {
          setSelectedLayer(lNames[0])
        } else {
          setSelectedLayer('')
          setLayerWindow([])
        }
      })
      .catch((err) => {
        if (!cancelled) setError(err?.message || 'Failed to load spike window.')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [selectedSpike])

  useEffect(() => {
    if (!selectedLayer || selectedSpike == null) return

    let cancelled = false
    fetchSpikeLayer(selectedSpike, selectedLayer)
      .then((rows) => {
        if (!cancelled) setLayerWindow(rows)
      })
      .catch((err) => {
        if (!cancelled) setError(err?.message || `Failed to load ${selectedLayer}.`)
      })

    return () => {
      cancelled = true
    }
  }, [selectedSpike, selectedLayer])

  const currentEvent = useMemo(() => {
    return spikeEvents.find((e) => e.steps.includes(selectedSpike)) || null
  }, [spikeEvents, selectedSpike])

  const windowSteps = useMemo(() => globalWindow.map((r) => r.step), [globalWindow])
  const shapes = useMemo(
    () =>
      selectedSpike != null
        ? [spikeShape(selectedSpike, { color: 'rgba(252, 129, 129, 0.8)' })]
        : [],
    [selectedSpike]
  )
  const layerSteps = useMemo(() => layerWindow.map((r) => r.step), [layerWindow])

  const centerRow = useMemo(() => {
    return globalWindow.find((r) => r.step === selectedSpike) || null
  }, [globalWindow, selectedSpike])

  const diagnosis = useMemo(
    () => diagnoseSpike(globalWindow, selectedSpike, currentEvent),
    [globalWindow, selectedSpike, currentEvent]
  )

  if (spikes.length === 0) {
    return (
      <EmptyState icon={<Zap className="h-5 w-5" />}>No spikes recorded in this run.</EmptyState>
    )
  }

  return (
    <div className="space-y-5">
      <Card className="control-card">
        <CardHeader>
          <div>
            <CardTitle>Select an anomaly event</CardTitle>
            <p className="chart-card__description">
              Start at the first detection or jump straight to the peak step.
            </p>
          </div>
          {diagnosis && (
            <Badge variant={diagnosis.badgeVariant}>
              <Zap className="h-3 w-3" />
              {diagnosis.primaryTrigger}
            </Badge>
          )}
        </CardHeader>
        <CardContent>
          <div className="flex min-w-0 flex-col gap-2 sm:flex-row sm:items-center sm:gap-3">
            <label htmlFor="spike-select" className="control-label shrink-0">
              Event / step
            </label>
            <select
              id="spike-select"
              value={selectedSpike ?? ''}
              onChange={(e) => setSelectedSpike(Number(e.target.value))}
              className="control-select min-w-0 flex-1 sm:max-w-[34rem]"
            >
              <option value="">— select —</option>
              {spikeEvents.length > 0
                ? spikeEvents.map((ev) => (
                    <optgroup key={ev.id} label={ev.label}>
                      <option value={ev.startStep}>Step {ev.startStep} (First Detection)</option>
                      {ev.peakStep !== ev.startStep && (
                        <option value={ev.peakStep}>
                          Step {ev.peakStep} (Highest Observed Loss)
                        </option>
                      )}
                    </optgroup>
                  ))
                : spikes.map((s) => (
                    <option key={s.step} value={s.step}>
                      Step {s.step}
                    </option>
                  ))}
            </select>
          </div>
        </CardContent>
      </Card>

      {diagnosis && (
        <Card className="story-card">
          <CardHeader className="items-start pb-2">
            <div className="min-w-0">
              <CardTitle className="flex items-center gap-2 text-sm font-semibold text-accent">
                <Activity className="h-4 w-4" />
                Spike story · Step {selectedSpike}
              </CardTitle>
              {currentEvent && currentEvent.startStep !== currentEvent.endStep && (
                <p className="mt-1 text-xs text-muted">
                  Event range: Steps {currentEvent.startStep}–{currentEvent.endStep}
                </p>
              )}
            </div>
            <Badge variant={diagnosis.badgeVariant}>{diagnosis.primaryTrigger}</Badge>
          </CardHeader>
          <CardContent className="space-y-3 text-xs leading-relaxed">
            <p>{diagnosis.desc}</p>
            <p className="story-cascade pt-3">{diagnosis.cascadePath}</p>
          </CardContent>
        </Card>
      )}

      {loading && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <Skeleton className="h-96" />
          <Skeleton className="h-96" />
        </div>
      )}

      <ErrorMessage message={error} />

      {!loading && globalWindow.length > 0 && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {/* Timeline pane */}
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
              <StatCard label="Spike Step" value={selectedSpike ?? '—'} icon={Zap} />
              <StatCard
                label="Loss at spike"
                value={centerRow?.loss?.toFixed(4) ?? '—'}
                icon={TrendingDown}
              />
              <StatCard
                label="Grad Norm"
                value={centerRow?.grad_norm_before_clip?.toFixed(4) ?? '—'}
                icon={Activity}
              />
            </div>

            <ChartCard
              title="Global pre / post-spike window"
              description="Loss and gradient norm aligned around the selected anomaly."
            >
              <Chart
                data={[
                  {
                    x: windowSteps,
                    y: globalWindow.map((r) => r.loss),
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Loss',
                    line: { color: CHART_COLORS.loss, width: 1.5 },
                  },
                  {
                    x: windowSteps,
                    y: globalWindow.map((r) => r.grad_norm_before_clip),
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Grad Norm',
                    line: { color: CHART_COLORS.gradNorm, width: 1.5 },
                    yaxis: 'y2',
                  },
                ]}
                layout={{
                  height: 320,
                  shapes,
                  yaxis2: {
                    overlaying: 'y',
                    side: 'right',
                    gridcolor: CHART_COLORS.muted,
                    color: CHART_COLORS.gradNorm,
                    title: {
                      text: 'Grad Norm',
                      font: { color: CHART_COLORS.gradNorm, size: 11 },
                    },
                  },
                  legend: { orientation: 'h', y: -0.15 },
                  margin: { t: 20, r: 60 },
                  uirevision: 'spike-global',
                }}
              />
            </ChartCard>
          </div>

          {/* Layers pane */}
          <div className="space-y-4">
            {layerNames.length > 0 && (
              <Card className="control-card">
                <LayerSelect
                  id="spike-layer-select"
                  label="Target layer"
                  layers={layerNames}
                  value={selectedLayer}
                  onChange={setSelectedLayer}
                />
              </Card>
            )}

            {layerWindow.length > 0 && selectedLayer && (
              <>
                <ChartCard title="Activation kurtosis" description={selectedLayer}>
                  <Chart
                    data={[
                      {
                        x: layerSteps,
                        y: layerWindow.map((r) => r.act_kurtosis),
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Kurtosis',
                        // Unsampled steps persist as null and render as gaps.
                        connectgaps: false,
                        line: { color: CHART_COLORS.kurtosis, width: 1.5 },
                      },
                    ]}
                    layout={{
                      height: 260,
                      shapes,
                      uirevision: `spike-layer-${selectedLayer}-kurtosis`,
                    }}
                  />
                </ChartCard>

                <ChartCard title="Gradient L2 norm" description={selectedLayer}>
                  <Chart
                    data={[
                      {
                        x: layerSteps,
                        y: layerWindow.map((r) => r.grad_l2_norm),
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Grad L2',
                        line: { color: CHART_COLORS.gradNorm, width: 1.5 },
                      },
                    ]}
                    layout={{
                      height: 260,
                      shapes,
                      uirevision: `spike-layer-${selectedLayer}-grad`,
                    }}
                  />
                </ChartCard>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
