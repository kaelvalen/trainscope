import { useState, useEffect } from 'react'
import Plot from 'react-plotly.js'
import { fetchSpikes, fetchSpike, fetchSpikeLayerNames, fetchSpikeLayer } from '../api.js'

const DARK_LAYOUT = {
  paper_bgcolor: '#1a1f2e',
  plot_bgcolor: '#0f1117',
  font: { color: '#e2e8f0', size: 12 },
  margin: { l: 60, r: 20, t: 40, b: 40 },
  xaxis: { gridcolor: '#2d3748', zerolinecolor: '#2d3748' },
  yaxis: { gridcolor: '#2d3748', zerolinecolor: '#2d3748' },
}

function spikeShape(step) {
  return [{
    type: 'line',
    x0: step, x1: step,
    yref: 'paper', y0: 0, y1: 1,
    line: { color: 'rgba(252, 129, 129, 0.8)', width: 1.5, dash: 'dot' },
  }]
}

export default function SpikeInspector() {
  const [spikes, setSpikes] = useState([])
  const [selectedSpike, setSelectedSpike] = useState(null)
  const [globalWindow, setGlobalWindow] = useState([])
  const [layerNames, setLayerNames] = useState([])
  const [selectedLayer, setSelectedLayer] = useState('')
  const [layerWindow, setLayerWindow] = useState([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetchSpikes().then(setSpikes).catch(() => {})
  }, [])

  useEffect(() => {
    if (selectedSpike == null) return
    setLoading(true)
    Promise.all([
      fetchSpike(selectedSpike),
      fetchSpikeLayerNames(selectedSpike),
    ])
      .then(([gRows, lNames]) => {
        setGlobalWindow(gRows)
        setLayerNames(lNames)
        if (lNames.length > 0) {
          setSelectedLayer(lNames[0])
        } else {
          setSelectedLayer('')
          setLayerWindow([])
        }
      })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [selectedSpike])

  useEffect(() => {
    if (!selectedLayer || selectedSpike == null) return
    fetchSpikeLayer(selectedSpike, selectedLayer)
      .then(setLayerWindow)
      .catch(console.error)
  }, [selectedSpike, selectedLayer])

  if (spikes.length === 0) {
    return (
      <div style={{ color: '#718096', padding: '40px', textAlign: 'center' }}>
        No spikes recorded in this run.
      </div>
    )
  }

  const windowSteps = globalWindow.map(r => r.step)
  const shapes = selectedSpike != null ? spikeShape(selectedSpike) : []

  const layerSteps = layerWindow.map(r => r.step)

  return (
    <div>
      <div style={{ marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
        <label style={{ fontSize: '13px', color: '#718096' }}>Spike:</label>
        <select
          value={selectedSpike ?? ''}
          onChange={e => setSelectedSpike(Number(e.target.value))}
          style={{
            background: '#1a1f2e', border: '1px solid #2d3748', color: '#e2e8f0',
            padding: '6px 12px', borderRadius: '6px', fontSize: '13px',
          }}
        >
          <option value="">— select —</option>
          {spikes.map(s => (
            <option key={s.step} value={s.step}>Step {s.step}</option>
          ))}
        </select>

        {layerNames.length > 0 && (
          <>
            <label style={{ fontSize: '13px', color: '#718096' }}>Layer:</label>
            <select
              value={selectedLayer}
              onChange={e => setSelectedLayer(e.target.value)}
              style={{
                background: '#1a1f2e', border: '1px solid #2d3748', color: '#e2e8f0',
                padding: '6px 12px', borderRadius: '6px', fontSize: '13px',
                maxWidth: '360px', flex: 1,
              }}
            >
              {layerNames.map(name => (
                <option key={name} value={name}>{name}</option>
              ))}
            </select>
          </>
        )}
      </div>

      {loading && (
        <div style={{ color: '#718096', padding: '40px', textAlign: 'center' }}>
          Loading spike window…
        </div>
      )}

      {!loading && globalWindow.length > 0 && (
        <>
          <Plot
            data={[
              {
                x: windowSteps,
                y: globalWindow.map(r => r.loss),
                type: 'scatter', mode: 'lines',
                name: 'Loss',
                line: { color: '#63b3ed', width: 1.5 },
              },
              {
                x: windowSteps,
                y: globalWindow.map(r => r.grad_norm_before_clip),
                type: 'scatter', mode: 'lines',
                name: 'Grad Norm',
                line: { color: '#68d391', width: 1.5 },
                yaxis: 'y2',
              },
            ]}
            layout={{
              ...DARK_LAYOUT,
              title: { text: `Loss + Grad Norm — spike window (step ${selectedSpike})`, font: { size: 14 } },
              height: 260,
              shapes,
              yaxis2: {
                overlaying: 'y', side: 'right',
                gridcolor: '#2d3748', color: '#68d391',
                title: { text: 'Grad Norm', font: { color: '#68d391', size: 11 } },
              },
              legend: { orientation: 'h', y: -0.15 },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
            useResizeHandler
          />
        </>
      )}

      {!loading && layerWindow.length > 0 && selectedLayer && (
        <>
          <Plot
            data={[{
              x: layerSteps,
              y: layerWindow.map(r => r.act_kurtosis),
              type: 'scatter', mode: 'lines',
              name: 'Kurtosis',
              line: { color: '#f6ad55', width: 1.5 },
            }]}
            layout={{
              ...DARK_LAYOUT,
              title: { text: `Activation Kurtosis — ${selectedLayer}`, font: { size: 14 } },
              height: 220,
              shapes,
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
            useResizeHandler
          />

          <Plot
            data={[{
              x: layerSteps,
              y: layerWindow.map(r => r.grad_l2_norm),
              type: 'scatter', mode: 'lines',
              name: 'Grad L2',
              line: { color: '#68d391', width: 1.5 },
            }]}
            layout={{
              ...DARK_LAYOUT,
              title: { text: `Gradient L2 Norm — ${selectedLayer}`, font: { size: 14 } },
              height: 220,
              shapes,
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
            useResizeHandler
          />
        </>
      )}
    </div>
  )
}
