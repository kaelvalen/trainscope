import { useState, useEffect, useMemo } from 'react'
import { useRun } from '../RunContext.jsx'
import { fetchSpike, fetchSpikeLayerNames, fetchSpikeLayer } from '../api.js'
import { spikeShape, CHART_COLORS } from '../theme.js'
import Chart from '../components/Chart.jsx'
import LayerSelect from '../components/LayerSelect.jsx'
import LoadingSpinner from '../components/LoadingSpinner.jsx'
import ErrorMessage from '../components/ErrorMessage.jsx'
import EmptyState from '../components/EmptyState.jsx'

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

  if (spikes.length === 0) {
    return <EmptyState icon="⚡">No spikes recorded in this run.</EmptyState>
  }

  return (
    <div>
      <div className="ts-control-row">
        <label htmlFor="spike-select" className="ts-label">
          Spike:
        </label>
        <select
          id="spike-select"
          value={selectedSpike ?? ''}
          onChange={(e) => setSelectedSpike(Number(e.target.value))}
          className="ts-select"
        >
          <option value="">— select —</option>
          {spikes.map((s) => (
            <option key={s.step} value={s.step}>
              Step {s.step}
            </option>
          ))}
        </select>

        {layerNames.length > 0 && (
          <LayerSelect
            id="spike-layer-select"
            label="Layer:"
            layers={layerNames}
            value={selectedLayer}
            onChange={setSelectedLayer}
          />
        )}
      </div>

      {loading && <LoadingSpinner message="Loading spike window…" />}
      <ErrorMessage message={error} />

      {!loading && globalWindow.length > 0 && (
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
            title: {
              text: `Loss + Grad Norm — spike window (step ${selectedSpike})`,
              font: { size: 14 },
            },
            height: 260,
            shapes,
            yaxis2: {
              overlaying: 'y',
              side: 'right',
              gridcolor: CHART_COLORS.muted,
              color: CHART_COLORS.gradNorm,
              title: { text: 'Grad Norm', font: { color: CHART_COLORS.gradNorm, size: 11 } },
            },
            legend: { orientation: 'h', y: -0.15 },
          }}
        />
      )}

      {!loading && layerWindow.length > 0 && selectedLayer && (
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
              title: { text: `Activation Kurtosis — ${selectedLayer}`, font: { size: 14 } },
              height: 220,
              shapes,
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
              title: { text: `Gradient L2 Norm — ${selectedLayer}`, font: { size: 14 } },
              height: 220,
              shapes,
            }}
          />
        </>
      )}
    </div>
  )
}
