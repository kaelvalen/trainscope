import { useState, useEffect, useMemo } from 'react'
import { useRun } from '../RunContext.jsx'
import { fetchSpike, fetchSpikeLayerNames, fetchSpikeLayer } from '../api.js'
import { CHART_COLORS } from '../theme.js'
import { diagnoseSpike } from '../utils/diagnosis.js'
import LineChart from '../components/charts/LineChart.jsx'
import LayerSelect from '../components/LayerSelect.jsx'
import ErrorMessage from '../components/ErrorMessage.jsx'
import EmptyState from '../components/EmptyState.jsx'
import { Badge } from '../components/ui/Badge.jsx'
import { Panel } from '../components/ui/Panel.jsx'
import { Skeleton } from '../components/ui/Skeleton.jsx'
import { StatStrip } from '../components/ui/StatStrip.jsx'

const fmt = (v) => (typeof v === 'number' && Number.isFinite(v) ? String(+v.toPrecision(4)) : '—')

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
    if (spikes.length > 0 && selectedSpike == null) setSelectedSpike(spikes[0].step)
  }, [spikes, selectedSpike])

  useEffect(() => {
    if (selectedSpike == null) return undefined
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
      .catch((err) => !cancelled && setError(err?.message || 'Failed to load spike window.'))
      .finally(() => !cancelled && setLoading(false))
    return () => {
      cancelled = true
    }
  }, [selectedSpike])

  useEffect(() => {
    if (!selectedLayer || selectedSpike == null) return undefined
    let cancelled = false
    fetchSpikeLayer(selectedSpike, selectedLayer)
      .then((rows) => !cancelled && setLayerWindow(rows))
      .catch((err) => !cancelled && setError(err?.message || `Failed to load ${selectedLayer}.`))
    return () => {
      cancelled = true
    }
  }, [selectedSpike, selectedLayer])

  const currentEvent = useMemo(
    () => spikeEvents.find((e) => e.steps.includes(selectedSpike)) || null,
    [spikeEvents, selectedSpike]
  )
  const windowSteps = useMemo(() => globalWindow.map((r) => r.step), [globalWindow])
  const layerSteps = useMemo(() => layerWindow.map((r) => r.step), [layerWindow])
  const markers = useMemo(
    () => (selectedSpike != null ? [{ at: selectedSpike, kind: 'spike' }] : []),
    [selectedSpike]
  )
  const centerRow = useMemo(
    () => globalWindow.find((r) => r.step === selectedSpike) || null,
    [globalWindow, selectedSpike]
  )
  const diagnosis = useMemo(
    () => diagnoseSpike(globalWindow, selectedSpike, currentEvent),
    [globalWindow, selectedSpike, currentEvent]
  )

  if (spikes.length === 0) return <EmptyState>No spikes recorded in this run.</EmptyState>

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Spikes</h1>
          <p className="mt-1 text-sm text-muted">
            Start at the first detection or jump to the step with the highest loss.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <label htmlFor="spike-select" className="text-sm text-muted">
            Step
          </label>
          <select
            id="spike-select"
            value={selectedSpike ?? ''}
            onChange={(e) => setSelectedSpike(Number(e.target.value))}
            className="h-8 max-w-[26rem] rounded-control border border-border bg-surface px-2 text-sm"
          >
            {spikeEvents.length > 0
              ? spikeEvents.map((ev) => (
                  <optgroup key={ev.id} label={ev.label}>
                    <option value={ev.startStep}>Step {ev.startStep}, first detection</option>
                    {ev.peakStep !== ev.startStep && (
                      <option value={ev.peakStep}>Step {ev.peakStep}, highest loss</option>
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
      </div>

      {diagnosis && (
        <Panel
          title={`What happened at step ${selectedSpike}`}
          actions={<Badge variant={diagnosis.badgeVariant}>{diagnosis.primaryTrigger}</Badge>}
        >
          {currentEvent && currentEvent.startStep !== currentEvent.endStep && (
            <p className="text-xs text-muted">
              Event spans steps {currentEvent.startStep}–{currentEvent.endStep}
            </p>
          )}
          <p className="mt-1 text-sm">{diagnosis.desc}</p>
          <p className="mt-2 border-t border-border pt-2 text-sm text-muted">
            {diagnosis.cascadePath}
          </p>
        </Panel>
      )}

      {loading && <Skeleton className="h-96" />}
      <ErrorMessage message={error} />

      {!loading && globalWindow.length > 0 && (
        <div className="grid gap-4 xl:grid-cols-2">
          <div className="min-w-0 space-y-4">
            <StatStrip
              items={[
                { label: 'Spike step', value: selectedSpike ?? '—' },
                { label: 'Loss at spike', value: fmt(centerRow?.loss) },
                { label: 'Grad norm', value: fmt(centerRow?.grad_norm_before_clip) },
                {
                  label: 'Window',
                  value: `${windowSteps[0]}–${windowSteps[windowSteps.length - 1]}`,
                },
              ]}
            />
            <Panel title="Loss and gradient norm around the spike" bodyClassName="px-2 pb-2 pt-3">
              <LineChart
                x={windowSteps}
                height={180}
                xAxis={false}
                syncKey="spike-window"
                markers={markers}
                series={[
                  {
                    label: 'Loss',
                    values: globalWindow.map((r) => r.loss),
                    color: CHART_COLORS.loss,
                  },
                ]}
                ariaLabel="Loss around the selected spike"
              />
              <LineChart
                x={windowSteps}
                height={150}
                syncKey="spike-window"
                markers={markers}
                series={[
                  {
                    label: 'Grad norm',
                    values: globalWindow.map((r) => r.grad_norm_before_clip),
                    color: CHART_COLORS.gradNorm,
                  },
                ]}
                ariaLabel="Gradient norm around the selected spike"
              />
            </Panel>
          </div>

          <div className="min-w-0 space-y-4">
            {layerNames.length > 0 && (
              <div className="rounded-panel border border-border bg-surface px-4 py-3">
                <LayerSelect
                  id="spike-layer-select"
                  label="Layer"
                  layers={layerNames}
                  value={selectedLayer}
                  onChange={setSelectedLayer}
                />
              </div>
            )}
            {layerWindow.length > 0 && selectedLayer && (
              <Panel title={selectedLayer} bodyClassName="px-2 pb-2 pt-3">
                {layerWindow.some((r) => Number.isFinite(r.act_kurtosis)) ? (
                  <LineChart
                    x={layerSteps}
                    height={150}
                    xAxis={false}
                    syncKey="spike-layer"
                    markers={markers}
                    series={[
                      {
                        label: 'Activation kurtosis',
                        values: layerWindow.map((r) => r.act_kurtosis),
                        color: CHART_COLORS.kurtosis,
                      },
                    ]}
                    ariaLabel={`Activation kurtosis of ${selectedLayer}`}
                  />
                ) : (
                  <p className="px-2 pb-2 text-sm text-muted">
                    Activation kurtosis is not recorded for this tensor.
                  </p>
                )}
                <LineChart
                  x={layerSteps}
                  height={150}
                  syncKey="spike-layer"
                  markers={markers}
                  series={[
                    {
                      label: 'Gradient L2',
                      values: layerWindow.map((r) => r.grad_l2_norm),
                      color: CHART_COLORS.gradNorm,
                    },
                  ]}
                  ariaLabel={`Gradient L2 norm of ${selectedLayer}`}
                />
              </Panel>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
