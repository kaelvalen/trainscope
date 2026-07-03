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

  if (spikes.length === 0) {
    return <EmptyState icon="⚡">No spikes recorded in this run.</EmptyState>
  }

  return (
    <div className="space-y-5">
      <Card>
        <div className="flex flex-wrap items-center gap-3">
          <label htmlFor="spike-select" className="text-xs font-medium text-muted">
            Spike
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
      </Card>

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
                <CardTitle>Global window</CardTitle>
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
                  label="Layer:"
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
