import { useState, useEffect, useCallback } from 'react'
import Plot from 'react-plotly.js'
import { fetchGlobal, fetchLayers, fetchLayersRanked, fetchLayer } from '../api.js'

const DARK_LAYOUT = {
  paper_bgcolor: '#1a1f2e',
  plot_bgcolor: '#0f1117',
  font: { color: '#e2e8f0', size: 12 },
  margin: { l: 60, r: 20, t: 40, b: 40 },
  xaxis: { gridcolor: '#2d3748', zerolinecolor: '#2d3748' },
  yaxis: { gridcolor: '#2d3748', zerolinecolor: '#2d3748' },
}

function buildSpikeShapes(rows) {
  return rows
    .filter(r => r.is_spike)
    .map(r => ({
      type: 'line',
      x0: r.step,
      x1: r.step,
      yref: 'paper',
      y0: 0,
      y1: 1,
      line: { color: 'rgba(252, 129, 129, 0.6)', width: 1.5, dash: 'dot' },
    }))
}

export default function Timeline() {
  const [globalData, setGlobalData] = useState([])
  const [layerNames, setLayerNames] = useState([])
  const [layerGradNorms, setLayerGradNorms] = useState({})
  const [scrubStep, setScrubStep] = useState(0)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([fetchGlobal(), fetchLayers(), fetchLayersRanked(8).catch(() => [])])
      .then(async ([gData, names, ranked]) => {
        setGlobalData(gData)
        setLayerNames(names)
        if (gData.length > 0) setScrubStep(gData[0].step)

        // Use top-8 layers by grad variance; fall back to first 8 alphabetical
        const sample = ranked.length > 0 ? ranked : names.slice(0, 8)
        const layerDataMap = {}
        await Promise.all(
          sample.map(async name => {
            const rows = await fetchLayer(name)
            layerDataMap[name] = rows
          })
        )
        setLayerGradNorms(layerDataMap)
      })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  const spikeShapes = buildSpikeShapes(globalData)
  const steps = globalData.map(r => r.step)
  const losses = globalData.map(r => r.loss)
  const gradNorms = globalData.map(r => r.grad_norm_before_clip)

  const scrubIndex = steps.indexOf(scrubStep)
  const scrubRow = scrubIndex >= 0 ? globalData[scrubIndex] : null
  const minStep = steps.length > 0 ? steps[0] : 0
  const maxStep = steps.length > 0 ? steps[steps.length - 1] : 0

  const scrubLine = scrubStep != null
    ? [{
        type: 'line',
        x0: scrubStep, x1: scrubStep,
        yref: 'paper', y0: 0, y1: 1,
        line: { color: '#63b3ed', width: 1.5 },
      }]
    : []

  const layerGradTraces = Object.entries(layerGradNorms).map(([name, rows]) => ({
    x: rows.map(r => r.step),
    y: rows.map(r => r.grad_l2_norm),
    type: 'scatter',
    mode: 'lines',
    name: name.length > 30 ? '...' + name.slice(-28) : name,
    line: { width: 1 },
  }))

  if (loading) {
    return <div style={{ color: '#718096', padding: '40px', textAlign: 'center' }}>Loading timeline data...</div>
  }

  if (globalData.length === 0) {
    return <div style={{ color: '#718096', padding: '40px', textAlign: 'center' }}>No global data found. Has a training run completed?</div>
  }

  return (
    <div>
      <Plot
        data={[{
          x: steps,
          y: losses,
          type: 'scatter',
          mode: 'lines',
          name: 'Loss',
          line: { color: '#63b3ed', width: 1.5 },
        }]}
        layout={{
          ...DARK_LAYOUT,
          title: { text: 'Training Loss', font: { size: 14 } },
          height: 280,
          shapes: [...spikeShapes, ...scrubLine],
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%' }}
        useResizeHandler
      />

      <Plot
        data={[{
          x: steps,
          y: gradNorms,
          type: 'scatter',
          mode: 'lines',
          name: 'Grad Norm',
          line: { color: '#68d391', width: 1.5 },
        }]}
        layout={{
          ...DARK_LAYOUT,
          title: { text: 'Global Gradient Norm (before clip)', font: { size: 14 } },
          height: 240,
          shapes: [...spikeShapes, ...scrubLine],
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
          value={scrubStep}
          onChange={e => {
            const target = Number(e.target.value)
            const closest = steps.reduce((prev, curr) =>
              Math.abs(curr - target) < Math.abs(prev - target) ? curr : prev
            , steps[0])
            setScrubStep(closest)
          }}
          style={{ flex: 1 }}
        />
        {scrubRow && (
          <div style={{
            background: '#1a1f2e',
            border: '1px solid #2d3748',
            borderRadius: '6px',
            padding: '6px 12px',
            fontSize: '12px',
            minWidth: '200px',
          }}>
            <strong>Step {scrubRow.step}</strong>
            {' — '}
            Loss: <span style={{ color: '#63b3ed' }}>{scrubRow.loss?.toFixed(4)}</span>
            {' | '}
            Grad: <span style={{ color: '#68d391' }}>{scrubRow.grad_norm_before_clip?.toFixed(4)}</span>
            {scrubRow.is_spike && <span style={{ color: '#fc8181', marginLeft: 8 }}>SPIKE</span>}
          </div>
        )}
      </div>

      {layerGradTraces.length > 0 && (
        <Plot
          data={layerGradTraces}
          layout={{
            ...DARK_LAYOUT,
            title: { text: 'Per-Layer Gradient Norms (top 8 by grad variance)', font: { size: 14 } },
            height: 300,
            showlegend: true,
            legend: { font: { size: 10 } },
            shapes: scrubLine,
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%' }}
          useResizeHandler
        />
      )}
    </div>
  )
}
