import { useState, useEffect } from 'react'
import Plot from 'react-plotly.js'
import { fetchLayers, fetchLayer } from '../api.js'

const DARK_LAYOUT = {
  paper_bgcolor: '#1a1f2e',
  plot_bgcolor: '#0f1117',
  font: { color: '#e2e8f0', size: 12 },
  margin: { l: 60, r: 20, t: 40, b: 40 },
  xaxis: { gridcolor: '#2d3748', zerolinecolor: '#2d3748' },
  yaxis: { gridcolor: '#2d3748', zerolinecolor: '#2d3748' },
}

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
  const [layerNames, setLayerNames] = useState([])
  const [selectedLayer, setSelectedLayer] = useState('')
  const [layerData, setLayerData] = useState([])
  const [scrubStep, setScrubStep] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetchLayers().then(names => {
      setLayerNames(names)
      if (names.length > 0) setSelectedLayer(names[0])
    })
  }, [])

  useEffect(() => {
    if (!selectedLayer) return
    setLoading(true)
    fetchLayer(selectedLayer)
      .then(rows => {
        setLayerData(rows)
        if (rows.length > 0) setScrubStep(rows[0].step)
      })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [selectedLayer])

  const steps = layerData.map(r => r.step)
  const minStep = steps.length > 0 ? steps[0] : 0
  const maxStep = steps.length > 0 ? steps[steps.length - 1] : 0

  const scrubIndex = scrubStep != null ? steps.indexOf(scrubStep) : -1
  const scrubRow = scrubIndex >= 0 ? layerData[scrubIndex] : (layerData.length > 0 ? layerData[layerData.length - 1] : null)

  const histRow = scrubRow || (layerData.length > 0 ? layerData[layerData.length - 1] : null)
  const histCounts = histRow?.hist_counts || []
  const histEdges = histRow?.hist_edges || []
  const histMean = mean(histCounts)
  const histStd = stddev(histCounts)
  const histColors = histCounts.map(v =>
    Math.abs(v - histMean) > 2 * histStd ? '#fc8181' : '#63b3ed'
  )
  const binLabels = histEdges.slice(0, -1).map((e, i) =>
    `${e.toFixed(3)} – ${(histEdges[i + 1] || 0).toFixed(3)}`
  )

  return (
    <div>
      <div style={{ marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '12px' }}>
        <label style={{ fontSize: '13px', color: '#718096' }}>Layer:</label>
        <select
          value={selectedLayer}
          onChange={e => setSelectedLayer(e.target.value)}
          style={{
            background: '#1a1f2e',
            border: '1px solid #2d3748',
            color: '#e2e8f0',
            padding: '6px 12px',
            borderRadius: '6px',
            fontSize: '13px',
            maxWidth: '400px',
            flex: 1,
          }}
        >
          {layerNames.map(name => (
            <option key={name} value={name}>{name}</option>
          ))}
        </select>
      </div>

      {loading && (
        <div style={{ color: '#718096', padding: '40px', textAlign: 'center' }}>Loading layer data...</div>
      )}

      {!loading && layerData.length === 0 && selectedLayer && (
        <div style={{ color: '#718096', padding: '40px', textAlign: 'center' }}>No data for layer: {selectedLayer}</div>
      )}

      {!loading && layerData.length > 0 && (
        <>
          <Plot
            data={[{
              x: steps,
              y: layerData.map(r => r.grad_l2_norm),
              type: 'scatter',
              mode: 'lines',
              name: 'Grad L2 Norm',
              line: { color: '#68d391', width: 1.5 },
            }]}
            layout={{
              ...DARK_LAYOUT,
              title: { text: 'Gradient L2 Norm', font: { size: 14 } },
              height: 220,
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
            useResizeHandler
          />

          <Plot
            data={[{
              x: steps,
              y: layerData.map(r => r.weight_l2_norm),
              type: 'scatter',
              mode: 'lines',
              name: 'Weight L2 Norm',
              line: { color: '#b794f4', width: 1.5 },
            }]}
            layout={{
              ...DARK_LAYOUT,
              title: { text: 'Weight L2 Norm', font: { size: 14 } },
              height: 220,
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
            useResizeHandler
          />

          <Plot
            data={[{
              x: steps,
              y: layerData.map(r => r.act_kurtosis),
              type: 'scatter',
              mode: 'lines',
              name: 'Activation Kurtosis',
              line: { color: '#f6ad55', width: 1.5 },
            }]}
            layout={{
              ...DARK_LAYOUT,
              title: { text: 'Activation Kurtosis (excess) — early spike signal', font: { size: 14 } },
              height: 220,
              shapes: [{
                type: 'line',
                x0: minStep, x1: maxStep,
                y0: 0, y1: 0,
                line: { color: '#718096', width: 1, dash: 'dot' },
              }],
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
            useResizeHandler
          />

          <div style={{ margin: '16px 0 8px', display: 'flex', alignItems: 'center', gap: '16px' }}>
            <label style={{ fontSize: '13px', color: '#718096' }}>Step scrubber:</label>
            <input
              type="range"
              min={minStep}
              max={maxStep}
              value={scrubStep ?? minStep}
              onChange={e => {
                const target = Number(e.target.value)
                const closest = steps.reduce((prev, curr) =>
                  Math.abs(curr - target) < Math.abs(prev - target) ? curr : prev
                , steps[0])
                setScrubStep(closest)
              }}
              style={{ flex: 1 }}
            />
            <span style={{ fontSize: '12px', color: '#718096' }}>Step {scrubStep ?? minStep}</span>
          </div>

          {histCounts.length > 0 && (
            <Plot
              data={[{
                x: binLabels,
                y: histCounts,
                type: 'bar',
                marker: { color: histColors },
                name: 'Weight Histogram',
              }]}
              layout={{
                ...DARK_LAYOUT,
                title: { text: `Weight Histogram at Step ${scrubRow?.step ?? '—'} (red = >2σ from mean)`, font: { size: 14 } },
                height: 260,
                xaxis: { ...DARK_LAYOUT.xaxis, tickangle: -35, tickfont: { size: 9 } },
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: '100%' }}
              useResizeHandler
            />
          )}
        </>
      )}
    </div>
  )
}
