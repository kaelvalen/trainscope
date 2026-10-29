import { useState, useEffect, useMemo } from 'react'
import { useRun } from '../RunContext.jsx'
import { fetchLayer } from '../api.js'
import { CHART_COLORS } from '../theme.js'
import BarChart from '../components/charts/BarChart.jsx'
import LineChart from '../components/charts/LineChart.jsx'
import LayerSelect from '../components/LayerSelect.jsx'
import StepScrubber from '../components/StepScrubber.jsx'
import ErrorMessage from '../components/ErrorMessage.jsx'
import EmptyState from '../components/EmptyState.jsx'
import { Panel } from '../components/ui/Panel.jsx'
import { Skeleton } from '../components/ui/Skeleton.jsx'
import { StatStrip } from '../components/ui/StatStrip.jsx'

function finiteValues(arr) {
  return (arr || []).filter((v) => typeof v === 'number' && Number.isFinite(v))
}

function mean(arr) {
  const vals = finiteValues(arr)
  return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : NaN
}

function stddev(arr) {
  const vals = finiteValues(arr)
  if (vals.length < 2) return NaN
  const m = mean(vals)
  return Math.sqrt(vals.reduce((a, b) => a + (b - m) ** 2, 0) / vals.length)
}

const fmt = (v) => (typeof v === 'number' && Number.isFinite(v) ? String(+v.toPrecision(4)) : '—')

/** `#layers?layer=<name>` (set by the Overview heatmap) preselects a layer. */
function layerFromHash() {
  const query = window.location.hash.split('?')[1]
  return query ? new URLSearchParams(query).get('layer') : null
}

const METRICS = {
  gradient: {
    label: 'Gradient L2',
    description: 'Magnitude of the layer update signal over time.',
    key: 'grad_l2_norm',
    color: CHART_COLORS.gradNorm,
  },
  weight: {
    label: 'Weight L2',
    description: 'Weight magnitude exposes slow parameter drift or runaway updates.',
    key: 'weight_l2_norm',
    color: CHART_COLORS.loss,
  },
  kurtosis: {
    label: 'Activation kurtosis',
    description:
      'Excess kurtosis is an early signal for heavy-tailed activations. It is sampled every few steps; gaps are steps that were not measured.',
    key: 'act_kurtosis',
    color: CHART_COLORS.kurtosis,
  },
}

export default function LayerDrilldown() {
  const { layerNames } = useRun()
  const [selectedLayer, setSelectedLayer] = useState('')
  const [layerData, setLayerData] = useState([])
  const [scrubStep, setScrubStep] = useState(null)
  const [activeMetric, setActiveMetric] = useState('gradient')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (layerNames.length === 0 || selectedLayer) return
    const wanted = layerFromHash()
    setSelectedLayer(wanted && layerNames.includes(wanted) ? wanted : layerNames[0])
  }, [layerNames, selectedLayer])

  useEffect(() => {
    if (!selectedLayer) return undefined
    let cancelled = false
    setLoading(true)
    setError(null)
    fetchLayer(selectedLayer)
      .then((rows) => {
        if (cancelled) return
        setLayerData(rows)
        if (rows.length > 0) setScrubStep(rows[rows.length - 1].step)
      })
      .catch((err) => !cancelled && setError(err?.message || `Failed to load ${selectedLayer}.`))
      .finally(() => !cancelled && setLoading(false))
    return () => {
      cancelled = true
    }
  }, [selectedLayer])

  const steps = useMemo(() => layerData.map((r) => r.step), [layerData])
  const metric = METRICS[activeMetric]
  const values = useMemo(() => layerData.map((r) => r[metric.key]), [layerData, metric.key])
  const columns = useMemo(
    () => ({
      grad: layerData.map((r) => r.grad_l2_norm),
      weight: layerData.map((r) => r.weight_l2_norm),
      kurtosis: layerData.map((r) => r.act_kurtosis),
    }),
    [layerData]
  )

  const scrubRow = useMemo(() => {
    if (scrubStep == null) return layerData[layerData.length - 1] ?? null
    return layerData.find((r) => r.step === scrubStep) ?? layerData[layerData.length - 1] ?? null
  }, [layerData, scrubStep])

  const histogram = useMemo(() => {
    const counts = scrubRow?.hist_counts || []
    const edges = scrubRow?.hist_edges || []
    const m = mean(counts)
    const s = stddev(counts)
    const outlier = (v) => Number.isFinite(s) && Math.abs(v - m) > 2 * s
    return {
      x: edges.slice(0, -1),
      normal: counts.map((v) => (outlier(v) ? null : v)),
      outliers: counts.map((v) => (outlier(v) ? v : null)),
      any: counts.some(outlier),
    }
  }, [scrubRow])

  if (layerNames.length === 0) {
    return (
      <EmptyState>No layers recorded yet. Layers appear after the first traced step.</EmptyState>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Layers</h1>
          <p className="mt-1 text-sm text-muted">
            Gradient, weight and activation signals for one parameter tensor.
          </p>
        </div>
        <LayerSelect
          id="layer-drilldown-select"
          layers={layerNames}
          value={selectedLayer}
          onChange={setSelectedLayer}
        />
      </div>

      {loading && (
        <div className="space-y-3">
          <Skeleton className="h-20" />
          <Skeleton className="h-72" />
        </div>
      )}

      <ErrorMessage message={error} />

      {!loading && layerData.length === 0 && selectedLayer && !error && (
        <EmptyState>No data recorded for {selectedLayer}.</EmptyState>
      )}

      {!loading && layerData.length > 0 && (
        <>
          <StatStrip
            items={[
              { label: 'Mean grad L2', value: fmt(mean(columns.grad)) },
              { label: 'Grad L2 σ', value: fmt(stddev(columns.grad)) },
              { label: 'Mean weight L2', value: fmt(mean(columns.weight)) },
              { label: 'Max kurtosis', value: fmt(Math.max(...finiteValues(columns.kurtosis))) },
            ]}
          />

          <Panel
            title={metric.label}
            actions={
              <div role="tablist" aria-label="Layer metric" className="flex gap-1">
                {Object.entries(METRICS).map(([key, m]) => (
                  <button
                    key={key}
                    type="button"
                    role="tab"
                    aria-selected={activeMetric === key}
                    onClick={() => setActiveMetric(key)}
                    className={`h-7 rounded-control px-2.5 text-xs transition-colors duration-150 ${
                      activeMetric === key
                        ? 'bg-ink text-surface'
                        : 'text-muted hover:bg-hover hover:text-ink'
                    }`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
            }
          >
            <p className="text-sm text-muted">{metric.description}</p>
            <div className="mt-3">
              <LineChart
                x={steps}
                height={280}
                series={[{ label: metric.label, values, color: metric.color }]}
                limit={activeMetric === 'kurtosis' ? { value: 0, label: '0' } : null}
                markers={scrubRow ? [{ at: scrubRow.step, kind: 'warning' }] : []}
                ariaLabel={`${metric.label} for ${selectedLayer}`}
              />
            </div>
          </Panel>

          <StepScrubber steps={steps} value={scrubStep} onChange={setScrubStep} />

          {histogram.x.length > 0 && (
            <Panel title={`Weight histogram at step ${scrubRow?.step ?? '—'}`}>
              <p className="text-sm text-muted">
                {histogram.any
                  ? 'Red bins sit more than two standard deviations from the mean bin count.'
                  : 'No bin sits more than two standard deviations from the mean bin count.'}
              </p>
              <div className="mt-3">
                <BarChart
                  x={histogram.x}
                  height={240}
                  series={[
                    { label: 'Weights', values: histogram.normal, color: 'var(--muted)' },
                    { label: 'Outlier bins', values: histogram.outliers, color: 'var(--red)' },
                  ]}
                  ariaLabel={`Weight histogram for ${selectedLayer} at step ${scrubRow?.step}`}
                />
              </div>
            </Panel>
          )}
        </>
      )}
    </div>
  )
}
