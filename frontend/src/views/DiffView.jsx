import { useState } from 'react'
import { GitCompare } from 'lucide-react'
import { fetchDiff } from '../api.js'
import { truncateLayerName, CHART_COLORS } from '../theme.js'
import Chart from '../components/Chart.jsx'
import ErrorMessage from '../components/ErrorMessage.jsx'
import LoadingSpinner from '../components/LoadingSpinner.jsx'
import EmptyState from '../components/EmptyState.jsx'
import { Button } from '../components/ui/Button.jsx'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card.jsx'
import { Badge } from '../components/ui/Badge.jsx'
import { StatCard } from '../components/ui/StatCard.jsx'

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
    if (Number.isNaN(a) || Number.isNaN(b)) {
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

  const topLayers = diffData.slice(0, 3)
  const maxKL = diffData.length > 0 ? Math.max(...kls) : 0
  const totalKL = diffData.length > 0 ? kls.reduce((a, b) => a + b, 0) : 0

  return (
    <div className="space-y-5">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <GitCompare className="h-4 w-4" />
            Compare two steps
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-end gap-3">
            <div className="flex flex-col gap-1">
              <label htmlFor="diff-step-a" className="text-xs font-medium text-muted">
                Step A
              </label>
              <input
                id="diff-step-a"
                type="number"
                value={stepA}
                onChange={(e) => setStepA(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="e.g. 4400"
                className="w-32 rounded-md border border-border bg-background px-3 py-1.5 text-sm text-foreground focus:border-accent"
              />
            </div>

            <div className="flex flex-col gap-1">
              <label htmlFor="diff-step-b" className="text-xs font-medium text-muted">
                Step B
              </label>
              <input
                id="diff-step-b"
                type="number"
                value={stepB}
                onChange={(e) => setStepB(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="e.g. 4450"
                className="w-32 rounded-md border border-border bg-background px-3 py-1.5 text-sm text-foreground focus:border-accent"
              />
            </div>

            <Button onClick={handleCompare} disabled={loading}>
              {loading ? 'Comparing…' : 'Compare'}
            </Button>
          </div>
        </CardContent>
      </Card>

      {loading && <LoadingSpinner message="Comparing steps…" />}
      <ErrorMessage message={error} onRetry={handleCompare} />

      {!loading && diffData.length === 0 && compared && !error && (
        <EmptyState icon="🌗">No layer data available for the selected steps.</EmptyState>
      )}

      {!loading && diffData.length > 0 && compared && (
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard label="Step A" value={compared.a} />
            <StatCard label="Step B" value={compared.b} />
            <StatCard label="Max KL Divergence" value={maxKL.toFixed(4)} />
            <StatCard label="Total KL Divergence" value={totalKL.toFixed(4)} />
          </div>

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
            <Card className="lg:col-span-1">
              <CardHeader>
                <CardTitle>Top diverged layers</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {topLayers.map((layer, index) => (
                    <li
                      key={layer.layer}
                      className="flex items-center justify-between rounded-md bg-background p-2 text-sm"
                    >
                      <span className="truncate pr-2 text-foreground" title={layer.layer}>
                        {index + 1}. {truncateLayerName(layer.layer, 35)}
                      </span>
                      <Badge variant={index === 0 ? 'danger' : index === 1 ? 'warning' : 'accent'}>
                        {layer.kl_divergence.toFixed(4)}
                      </Badge>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>

            <Card className="lg:col-span-2">
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
            </Card>
          </div>
        </>
      )}
    </div>
  )
}
