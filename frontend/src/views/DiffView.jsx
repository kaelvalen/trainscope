import { useState } from 'react'
import Plot from 'react-plotly.js'
import { fetchDiff } from '../api.js'

const DARK_LAYOUT = {
  paper_bgcolor: '#1a1f2e',
  plot_bgcolor: '#0f1117',
  font: { color: '#e2e8f0', size: 12 },
  margin: { l: 240, r: 20, t: 60, b: 40 },
  xaxis: { gridcolor: '#2d3748', zerolinecolor: '#2d3748' },
  yaxis: { gridcolor: '#2d3748', zerolinecolor: '#2d3748', automargin: true },
}

export default function DiffView() {
  const [stepA, setStepA] = useState('')
  const [stepB, setStepB] = useState('')
  const [diffData, setDiffData] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [compared, setCompared] = useState(null)

  async function handleCompare() {
    const a = parseInt(stepA, 10)
    const b = parseInt(stepB, 10)
    if (isNaN(a) || isNaN(b)) {
      setError('Please enter valid step numbers.')
      return
    }
    setError(null)
    setLoading(true)
    try {
      const data = await fetchDiff(a, b)
      setDiffData(data)
      setCompared({ a, b })
    } catch (e) {
      setError('Failed to fetch diff data.')
    } finally {
      setLoading(false)
    }
  }

  const layers = diffData.map(d => d.layer)
  const kls = diffData.map(d => d.kl_divergence)
  const barColors = layers.map((_, i) =>
    i < 3 ? '#fc8181' : '#63b3ed'
  )

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px', flexWrap: 'wrap' }}>
        <label style={{ fontSize: '13px', color: '#718096' }}>Step A:</label>
        <input
          type="number"
          value={stepA}
          onChange={e => setStepA(e.target.value)}
          placeholder="e.g. 4400"
          style={{
            background: '#1a1f2e',
            border: '1px solid #2d3748',
            color: '#e2e8f0',
            padding: '6px 12px',
            borderRadius: '6px',
            fontSize: '13px',
            width: '120px',
          }}
        />
        <label style={{ fontSize: '13px', color: '#718096' }}>Step B:</label>
        <input
          type="number"
          value={stepB}
          onChange={e => setStepB(e.target.value)}
          placeholder="e.g. 4450"
          style={{
            background: '#1a1f2e',
            border: '1px solid #2d3748',
            color: '#e2e8f0',
            padding: '6px 12px',
            borderRadius: '6px',
            fontSize: '13px',
            width: '120px',
          }}
        />
        <button
          onClick={handleCompare}
          disabled={loading}
          style={{
            background: '#2b6cb0',
            color: '#fff',
            border: 'none',
            borderRadius: '6px',
            padding: '7px 18px',
            fontSize: '13px',
            fontWeight: 600,
            cursor: loading ? 'not-allowed' : 'pointer',
            opacity: loading ? 0.6 : 1,
          }}
        >
          {loading ? 'Comparing...' : 'Compare'}
        </button>
      </div>

      {error && (
        <div style={{ color: '#fc8181', marginBottom: '16px', fontSize: '13px' }}>{error}</div>
      )}

      {diffData.length === 0 && !loading && !error && compared && (
        <div style={{ color: '#718096', padding: '40px', textAlign: 'center' }}>
          No layer data available for the selected steps.
        </div>
      )}

      {diffData.length > 0 && compared && (
        <Plot
          data={[{
            x: kls,
            y: layers.map(l => l.length > 40 ? '...' + l.slice(-38) : l),
            type: 'bar',
            orientation: 'h',
            marker: { color: barColors },
            name: 'KL Divergence',
          }]}
          layout={{
            ...DARK_LAYOUT,
            title: {
              text: `Weight Distribution KL Divergence: Step ${compared.a} vs Step ${compared.b}`,
              font: { size: 14 },
            },
            height: Math.max(400, 30 * layers.length + 120),
            yaxis: {
              ...DARK_LAYOUT.yaxis,
              categoryorder: 'array',
              categoryarray: [...layers].reverse(),
            },
            annotations: [{
              text: 'Red = top 3 diverged layers',
              xref: 'paper', yref: 'paper',
              x: 1, y: 1.05,
              showarrow: false,
              font: { size: 11, color: '#718096' },
              xanchor: 'right',
            }],
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%' }}
          useResizeHandler
        />
      )}
    </div>
  )
}
