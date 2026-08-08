import { useState, useEffect, useMemo } from 'react'
import { useRun } from '../RunContext.jsx'
import { fetchLayer } from '../api.js'
import { CHART_COLORS } from '../theme.js'
import Chart from '../components/Chart.jsx'
import LayerSelect from '../components/LayerSelect.jsx'
import StepScrubber from '../components/StepScrubber.jsx'
import ErrorMessage from '../components/ErrorMessage.jsx'
import EmptyState from '../components/EmptyState.jsx'
import { Skeleton } from '../components/ui/Skeleton.jsx'
import { StatCard } from '../components/ui/StatCard.jsx'
import { ChartCard } from '../components/ui/ChartCard.jsx'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card.jsx'
import { Activity, BarChart3, FolderOpen, Scale, Search } from 'lucide-react'

function mean(arr) {
  const vals = finiteValues(arr)
  if (!vals.length) return NaN
  return vals.reduce((a, b) => a + b, 0) / vals.length
}

function stddev(arr) {
  const vals = finiteValues(arr)
  if (vals.length < 2) return NaN
  const m = mean(vals)
  return Math.sqrt(vals.reduce((a, b) => a + (b - m) ** 2, 0) / vals.length)
}

function finiteValues(arr) {
  return (arr || []).filter((v) => typeof v === 'number' && Number.isFinite(v))
}

function formatFloat(v) {
  return typeof v === 'number' && Number.isFinite(v) ? v.toFixed(4) : '—'
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
    if (layerNames.length > 0 && !selectedLayer) {
      setSelectedLayer(layerNames[0])
    }
  }, [layerNames, selectedLayer])

  useEffect(() => {
    if (!selectedLayer) return

    let cancelled = false
    setLoading(true)
    setError(null)

    fetchLayer(selectedLayer)
      .then((rows) => {
        if (cancelled) return
        setLayerData(rows)
        if (rows.length > 0) setScrubStep(rows[rows.length - 1].step)
      })
      .catch((err) => {
        if (!cancelled) setError(err?.message || `Failed to load ${selectedLayer}.`)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [selectedLayer])

  const steps = useMemo(() => layerData.map((r) => r.step), [layerData])
  const gradNorms = useMemo(() => layerData.map((r) => r.grad_l2_norm), [layerData])
  const weightNorms = useMemo(() => layerData.map((r) => r.weight_l2_norm), [layerData])
  const kurtosisValues = useMemo(() => layerData.map((r) => r.act_kurtosis), [layerData])

  const stats = useMemo(() => {
    const mkMetric = (values) => {
      const vals = finiteValues(values)
      return {
        mean: mean(values),
        std: stddev(values),
        max: vals.length ? Math.max(...vals) : NaN,
        min: vals.length ? Math.min(...vals) : NaN,
      }
    }
    return {
      grad: mkMetric(gradNorms),
      weight: mkMetric(weightNorms),
      kurtosis: mkMetric(kurtosisValues),
    }
  }, [gradNorms, weightNorms, kurtosisValues])

  const scrubRow = useMemo(() => {
    if (scrubStep == null) return layerData.length > 0 ? layerData[layerData.length - 1] : null
    return layerData.find((r) => r.step === scrubStep) || layerData[layerData.length - 1] || null
  }, [layerData, scrubStep])

  const { histCounts, histColors, binLabels } = useMemo(() => {
    const counts = scrubRow?.hist_counts || []
    const edges = scrubRow?.hist_edges || []
    const m = mean(counts)
    const s = stddev(counts)
    const colors = counts.map((v) =>
      Math.abs(v - m) > 2 * s ? CHART_COLORS.spike : CHART_COLORS.loss
    )
    const labels = edges
      .slice(0, -1)
      .map((e, i) => `${e.toFixed(3)} – ${(edges[i + 1] || 0).toFixed(3)}`)
    return { histCounts: counts, histColors: colors, binLabels: labels }
  }, [scrubRow])

  const metricConfig = {
    gradient: {
      label: 'Gradient L2',
      title: 'Gradient L2 norm',
      description: 'Magnitude of the layer update signal over time.',
      name: 'Grad L2 Norm',
      values: gradNorms,
      color: CHART_COLORS.gradNorm,
    },
    weight: {
      label: 'Weight L2',
      title: 'Weight L2 norm',
      description: 'Weight magnitude can expose slow parameter drift or runaway updates.',
      name: 'Weight L2 Norm',
      values: weightNorms,
      color: CHART_COLORS.weight,
    },
    kurtosis: {
      label: 'Activation kurtosis',
      title: 'Activation kurtosis',
      description: 'Excess kurtosis is an early signal for heavy-tailed activations.',
      name: 'Activation Kurtosis',
      values: kurtosisValues,
      color: CHART_COLORS.kurtosis,
    },
  }
  const selectedMetric = metricConfig[activeMetric]

  if (layerNames.length === 0) {
    return (
      <EmptyState icon={<Search className="h-5 w-5" />}>
        No layers found. Has a training run completed?
      </EmptyState>
    )
  }

  return (
    <div className="space-y-5">
      <Card className="control-card">
        <CardHeader>
          <div>
            <CardTitle>Choose a layer</CardTitle>
            <p className="chart-card__description">
              Inspect gradient, weight, and activation signals.
            </p>
          </div>
        </CardHeader>
        <CardContent>
          <LayerSelect
            id="layer-drilldown-select"
            layers={layerNames}
            value={selectedLayer}
            onChange={setSelectedLayer}
          />
        </CardContent>
      </Card>

      {loading && (
        <div className="space-y-4">
          <Skeleton className="h-24" />
          <Skeleton className="h-64" />
          <Skeleton className="h-64" />
        </div>
      )}

      <ErrorMessage message={error} />

      {!loading && layerData.length === 0 && selectedLayer && !error && (
        <EmptyState icon={<FolderOpen className="h-5 w-5" />}>
          No data for layer: {selectedLayer}
        </EmptyState>
      )}

      {!loading && layerData.length > 0 && (
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <StatCard
              label="Mean Grad L2"
              value={formatFloat(stats.grad.mean)}
              subtitle={`σ = ${formatFloat(stats.grad.std)}`}
              icon={Activity}
            />
            <StatCard
              label="Mean Weight L2"
              value={formatFloat(stats.weight.mean)}
              subtitle={`max ${formatFloat(stats.weight.max)}`}
              icon={Scale}
            />
            <StatCard
              label="Mean Kurtosis"
              value={formatFloat(stats.kurtosis.mean)}
              subtitle={`max ${formatFloat(stats.kurtosis.max)}`}
              icon={BarChart3}
            />
          </div>

          <div className="metric-tabs" role="tablist" aria-label="Layer metrics">
            <span className="metric-tabs__label">Signal</span>
            {Object.entries(metricConfig).map(([key, metric]) => (
              <button
                key={key}
                type="button"
                role="tab"
                aria-selected={activeMetric === key}
                className={activeMetric === key ? 'metric-tab is-active' : 'metric-tab'}
                onClick={() => setActiveMetric(key)}
              >
                {metric.label}
              </button>
            ))}
          </div>

          <ChartCard title={selectedMetric.title} description={selectedMetric.description}>
            <Chart
              data={[
                {
                  x: steps,
                  y: selectedMetric.values,
                  type: 'scatter',
                  mode: 'lines',
                  name: selectedMetric.name,
                  // Activation metrics are only sampled every N steps; null
                  // samples (not measured) render as gaps, not as zero.
                  connectgaps: false,
                  line: { color: selectedMetric.color, width: 1.5 },
                },
              ]}
              layout={{
                height: 300,
                shapes:
                  activeMetric === 'kurtosis'
                    ? [
                        {
                          type: 'line',
                          x0: steps[0],
                          x1: steps[steps.length - 1],
                          y0: 0,
                          y1: 0,
                          line: { color: CHART_COLORS.muted, width: 1, dash: 'dot' },
                        },
                      ]
                    : [],
                uirevision: `layer-${selectedLayer}-${activeMetric}`,
              }}
            />
          </ChartCard>

          <StepScrubber steps={steps} value={scrubStep} onChange={setScrubStep} />

          {histCounts.length > 0 && (
            <ChartCard
              title={`Weight histogram at step ${scrubRow?.step ?? '—'}`}
              description="Red bins sit more than two standard deviations from the bin-count mean."
            >
              <Chart
                data={[
                  {
                    x: binLabels,
                    y: histCounts,
                    type: 'bar',
                    marker: { color: histColors },
                    name: 'Weight Histogram',
                  },
                ]}
                layout={{
                  height: 280,
                  xaxis: { tickangle: -35, tickfont: { size: 9 } },
                  uirevision: `layer-${selectedLayer}-hist`,
                }}
              />
            </ChartCard>
          )}
        </>
      )}
    </div>
  )
}
