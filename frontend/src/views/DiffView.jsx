import { useState } from 'react'
import { fetchDiff } from '../api.js'
import { truncateLayerName, CHART_COLORS } from '../theme.js'
import Chart from '../components/Chart.jsx'
import ErrorMessage from '../components/ErrorMessage.jsx'
import LoadingSpinner from '../components/LoadingSpinner.jsx'
import EmptyState from '../components/EmptyState.jsx'

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
    } catch (err) {
      setError(err?.message || 'Failed to fetch diff data.')
    } finally {
      setLoading(false)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter') handleCompare()
  }

  const layers = diffData.map((d) => d.layer)
  const kls = diffData.map((d) => d.kl_divergence)
  const barColors = layers.map((_, i) => (i < 3 ? CHART_COLORS.diffTop : CHART_COLORS.diffRest))

  return (
    <div>
      <div className="ts-control-row">
        <label htmlFor="diff-step-a" className="ts-label">
          Step A:
        </label>
        <input
          id="diff-step-a"
          type="number"
          value={stepA}
          onChange={(e) => setStepA(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="e.g. 4400"
          className="ts-input"
          style={{ width: '120px' }}
        />
        <label htmlFor="diff-step-b" className="ts-label">
          Step B:
        </label>
        <input
          id="diff-step-b"
          type="number"
          value={stepB}
          onChange={(e) => setStepB(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="e.g. 4450"
          className="ts-input"
          style={{ width: '120px' }}
        />
        <button onClick={handleCompare} disabled={loading} className="ts-button">
          {loading ? 'Comparing…' : 'Compare'}
        </button>
      </div>

      {loading && <LoadingSpinner message="Comparing steps…" />}
      <ErrorMessage message={error} onRetry={handleCompare} />

      {!loading && diffData.length === 0 && compared && !error && (
        <EmptyState icon="🌗">No layer data available for the selected steps.</EmptyState>
      )}

      {!loading && diffData.length > 0 && compared && (
        <Chart
          data={[
            {
              x: kls,
              y: layers.map((l) => truncateLayerName(l, 40)),
              type: 'bar',
              orientation: 'h',
              marker: { color: barColors },
              name: 'KL Divergence',
            },
          ]}
          layout={{
            title: {
              text: `Weight Distribution KL Divergence: Step ${compared.a} vs Step ${compared.b}`,
              font: { size: 14 },
            },
            height: Math.max(400, 30 * layers.length + 120),
            margin: { l: 240, r: 20, t: 60, b: 40 },
            yaxis: {
              automargin: true,
              categoryorder: 'array',
              categoryarray: [...layers].reverse(),
            },
            annotations: [
              {
                text: 'Red = top 3 diverged layers',
                xref: 'paper',
                yref: 'paper',
                x: 1,
                y: 1.05,
                showarrow: false,
                font: { size: 11, color: CHART_COLORS.muted },
                xanchor: 'right',
              },
            ],
          }}
        />
      )}
    </div>
  )
}
