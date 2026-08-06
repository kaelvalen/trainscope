import { useState, useEffect, useMemo } from 'react'
import { useRun } from '../RunContext.jsx'
import { fetchSpike, fetchSpikeLayerNames, fetchSpikeLayer } from '../api.js'
import { spikeShape, CHART_COLORS } from '../theme.js'
import Chart from '../components/Chart.jsx'
import LayerSelect from '../components/LayerSelect.jsx'
import ErrorMessage from '../components/ErrorMessage.jsx'
import EmptyState from '../components/EmptyState.jsx'
import { Skeleton } from '../components/ui/Skeleton.jsx'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card.jsx'
import { StatCard } from '../components/ui/StatCard.jsx'
import { Zap, TrendingDown, Activity } from 'lucide-react'

export default function SpikeInspector() {
  const { spikes } = useRun()
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

  const diagnosis = useMemo(() => {
    if (!globalWindow || globalWindow.length === 0) return null

    // Chronological scan of pre-spike window up to spike step
    let firstDriftStep = null
    let firstGradExplosionStep = null
    let firstNanStep = null

    const baselineLoss = globalWindow[0]?.loss || 1.0
    const baselineGrad = globalWindow[0]?.grad_norm_before_clip || 1.0

    for (const row of globalWindow) {
      if (firstNanStep == null && (row.grad_nan_inf_ratio > 0 || !Number.isFinite(row.loss))) {
        firstNanStep = row.step
      }
      if (
        firstGradExplosionStep == null &&
        row.grad_norm_before_clip > Math.max(5.0, baselineGrad * 3.0)
      ) {
        firstGradExplosionStep = row.step
      }
      const zScore = row.z_score ?? (row.loss - baselineLoss) / Math.max(1e-6, baselineLoss * 0.1)
      if (firstDriftStep == null && (zScore >= 3.5 || row.loss > baselineLoss * 1.5)) {
        firstDriftStep = row.step
      }
    }

    // Determine Chronological Sequence of Events
    const events = []
    if (firstDriftStep != null) events.push({ name: 'Loss Shift', step: firstDriftStep })
    if (firstGradExplosionStep != null)
      events.push({ name: 'Gradient Explosion', step: firstGradExplosionStep })
    if (firstNanStep != null) events.push({ name: 'NaN / Inf Collapse', step: firstNanStep })

    events.sort((a, b) => a.step - b.step)

    if (events.length === 0) {
      return {
        primaryTrigger: 'Distributional Shift',
        cascadePath: `Step ${selectedSpike}`,
        badgeClass: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30',
        desc: 'Anomalous deviation detected in loss stream baseline.',
      }
    }

    const primaryTrigger = events[0].name
    const cascadePath = events.map((e) => `${e.name} (Step ${e.step})`).join(' → ')

    let badgeClass = 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30'
    if (primaryTrigger === 'NaN / Inf Collapse') {
      badgeClass = 'bg-red-500/20 text-red-400 border-red-500/30'
    } else if (primaryTrigger === 'Gradient Explosion') {
      badgeClass = 'bg-amber-500/20 text-amber-400 border-amber-500/30'
    }

    return {
      primaryTrigger,
      cascadePath,
      badgeClass,
      desc: `Chronological Failure Cascade: ${cascadePath}.`,
    }
  }, [globalWindow, selectedSpike])

  if (spikes.length === 0) {
    return <EmptyState icon="⚡">No spikes recorded in this run.</EmptyState>
  }

  return (
    <div className="space-y-5">
      <Card>
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <label htmlFor="spike-select" className="text-xs font-medium text-muted">
              Select Spike Window
            </label>
            <select
              id="spike-select"
              value={selectedSpike ?? ''}
              onChange={(e) => setSelectedSpike(Number(e.target.value))}
              className="rounded-md border border-border bg-background px-3 py-1.5 text-sm text-foreground focus:border-accent"
            >
              <option value="">— select —</option>
              {spikes.map((s) => (
                <option key={s.step} value={s.step}>
                  Step {s.step}
                </option>
              ))}
            </select>
          </div>

          {diagnosis && (
            <div className={`flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs font-medium ${diagnosis.badgeClass}`}>
              <Zap className="h-4 w-4" />
              <span>Primary Trigger: {diagnosis.primaryTrigger}</span>
            </div>
          )}
        </div>
      </Card>

      {diagnosis && (
        <Card className="border-l-4 border-l-accent bg-accent/5">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm font-semibold text-accent">
              <Activity className="h-4 w-4" />
              Spike Story Narrative — Step {selectedSpike}
            </CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-muted leading-relaxed">
            {diagnosis.desc} Inspect the pre/post spike window metrics below to isolate layer-level activation kurtosis and gradient norms.
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

            <Card>
              <CardHeader>
                <CardTitle>Global Pre/Post Spike Window</CardTitle>
              </CardHeader>
              <CardContent>
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
              </CardContent>
            </Card>
          </div>

          {/* Layers pane */}
          <div className="space-y-4">
            {layerNames.length > 0 && (
              <Card>
                <LayerSelect
                  id="spike-layer-select"
                  label="Target Layer:"
                  layers={layerNames}
                  value={selectedLayer}
                  onChange={setSelectedLayer}
                />
              </Card>
            )}

            {layerWindow.length > 0 && selectedLayer && (
              <>
                <Chart
                  data={[
                    {
                      x: layerSteps,
                      y: layerWindow.map((r) => r.act_kurtosis),
                      type: 'scatter',
                      mode: 'lines',
                      name: 'Kurtosis',
                      line: { color: CHART_COLORS.kurtosis, width: 1.5 },
                    },
                  ]}
                  layout={{
                    title: {
                      text: `Activation Kurtosis — ${selectedLayer}`,
                      font: { size: 14 },
                    },
                    height: 260,
                    shapes,
                    uirevision: `spike-layer-${selectedLayer}-kurtosis`,
                  }}
                />

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
                    title: {
                      text: `Gradient L2 Norm — ${selectedLayer}`,
                      font: { size: 14 },
                    },
                    height: 260,
                    shapes,
                    uirevision: `spike-layer-${selectedLayer}-grad`,
                  }}
                />
              </>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
