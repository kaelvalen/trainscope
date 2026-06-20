import { useState, useEffect, useMemo } from 'react'
import { useRun } from '../RunContext.jsx'
import { fetchLayer } from '../api.js'
import { CHART_COLORS } from '../theme.js'
import Chart from '../components/Chart.jsx'
import LayerSelect from '../components/LayerSelect.jsx'
import StepScrubber from '../components/StepScrubber.jsx'
import LoadingSpinner from '../components/LoadingSpinner.jsx'
import ErrorMessage from '../components/ErrorMessage.jsx'
import EmptyState from '../components/EmptyState.jsx'

function mean(arr) {
  if (!arr.length) return 0
  return arr.reduce((a, b) => a + b, 0) / arr.length
}

function stddev(arr) {
  if (arr.length < 2) return 0
  const m = mean(arr)
  return Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length)
}

export default function LayerDrilldown() {
  const { layerNames } = useRun()
  const [selectedLayer, setSelectedLayer] = useState('')
  const [layerData, setLayerData] = useState([])
  const [scrubStep, setScrubStep] = useState(null)
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
        if (rows.length > 0) setScrubStep(rows[0].step)
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

  if (layerNames.length === 0) {
    return <EmptyState icon="🔍">No layers found. Has a training run completed?</EmptyState>
  }

  return (
    <div>
      <LayerSelect
        id="layer-drilldown-select"
        layers={layerNames}
        value={selectedLayer}
        onChange={setSelectedLayer}
      />

      {loading && <LoadingSpinner message="Loading layer data…" />}
      <ErrorMessage message={error} />

      {!loading && layerData.length === 0 && selectedLayer && !error && (
        <EmptyState icon="📂">No data for layer: {selectedLayer}</EmptyState>
      )}

      {!loading && layerData.length > 0 && (
        <>
          <Chart
            data={[
              {
                x: steps,
                y: layerData.map((r) => r.grad_l2_norm),
                type: 'scatter',
                mode: 'lines',
                name: 'Grad L2 Norm',
                line: { color: CHART_COLORS.gradNorm, width: 1.5 },
              },
            ]}
            layout={{
              title: { text: 'Gradient L2 Norm', font: { size: 14 } },
              height: 220,
            }}
          />

          <Chart
            data={[
              {
                x: steps,
                y: layerData.map((r) => r.weight_l2_norm),
                type: 'scatter',
                mode: 'lines',
                name: 'Weight L2 Norm',
                line: { color: CHART_COLORS.weight, width: 1.5 },
              },
            ]}
            layout={{
              title: { text: 'Weight L2 Norm', font: { size: 14 } },
              height: 220,
            }}
          />

          <Chart
            data={[
              {
                x: steps,
                y: layerData.map((r) => r.act_kurtosis),
                type: 'scatter',
                mode: 'lines',
                name: 'Activation Kurtosis',
                line: { color: CHART_COLORS.kurtosis, width: 1.5 },
              },
            ]}
            layout={{
              title: {
                text: 'Activation Kurtosis (excess) — early spike signal',
                font: { size: 14 },
              },
              height: 220,
              shapes: [
                {
                  type: 'line',
                  x0: steps[0],
                  x1: steps[steps.length - 1],
                  y0: 0,
                  y1: 0,
                  line: { color: CHART_COLORS.muted, width: 1, dash: 'dot' },
                },
              ],
            }}
          />

          <StepScrubber steps={steps} value={scrubStep} onChange={setScrubStep} />

          {histCounts.length > 0 && (
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
                title: {
                  text: `Weight Histogram at Step ${scrubRow?.step ?? '—'} (red = >2σ from mean)`,
                  font: { size: 14 },
                },
                height: 260,
                xaxis: { tickangle: -35, tickfont: { size: 9 } },
              }}
            />
          )}
        </>
      )}
    </div>
  )
}
