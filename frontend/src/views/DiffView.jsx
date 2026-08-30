import { useEffect, useState } from 'react'
import { ArrowLeftRight, GitCompare } from 'lucide-react'
import { useRun } from '../RunContext.jsx'
import { fetchDiff } from '../api.js'
import { truncateLayerName, CHART_COLORS } from '../theme.js'
import Chart from '../components/Chart.jsx'
import ErrorMessage from '../components/ErrorMessage.jsx'
import LoadingSpinner from '../components/LoadingSpinner.jsx'
import EmptyState from '../components/EmptyState.jsx'
import { Button } from '../components/ui/Button.jsx'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card.jsx'
import { ChartCard } from '../components/ui/ChartCard.jsx'
import { Badge } from '../components/ui/Badge.jsx'
import { StatCard } from '../components/ui/StatCard.jsx'

export default function DiffView() {
  const { globalData } = useRun()
  const [stepA, setStepA] = useState('')
  const [stepB, setStepB] = useState('')
  const [diffData, setDiffData] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [compared, setCompared] = useState(null)

  const availableSteps = globalData.map((row) => row.step)
  const latestStep = availableSteps[availableSteps.length - 1]
  const previousStep = availableSteps[availableSteps.length - 2] ?? latestStep

  useEffect(() => {
    if (latestStep == null) return
    setStepA((value) => (value === '' ? String(previousStep ?? latestStep) : value))
    setStepB((value) => (value === '' ? String(latestStep) : value))
  }, [latestStep, previousStep])

  async function handleCompare() {
    const a = parseInt(stepA, 10)
    const b = parseInt(stepB, 10)
    if (Number.isNaN(a) || Number.isNaN(b)) {
      setError('Please enter valid step numbers.')
      return
    }
    if (a === b) {
      setError('Choose two different steps to compare.')
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

  function setComparison(a, b) {
    if (a == null || b == null) return
    setStepA(String(a))
    setStepB(String(b))
    setError(null)
  }

  function handleSwap() {
    setStepA(stepB)
    setStepB(stepA)
  }

  const layers = diffData.map((d) => d.layer)
  const kls = diffData.map((d) => d.kl_divergence)
  const barColors = layers.map((_, i) => (i < 3 ? CHART_COLORS.diffTop : CHART_COLORS.diffRest))

  const topLayers = diffData.slice(0, 3)
  const maxKL = diffData.length > 0 ? Math.max(...kls) : 0
  const totalKL = diffData.length > 0 ? kls.reduce((a, b) => a + b, 0) : 0

  return (
    <div className="space-y-5">
      <Card className="control-card">
        <CardHeader>
          <div>
            <CardTitle className="flex items-center gap-2">
              <span className="metric-card__icon" aria-hidden="true">
                <GitCompare className="h-3.5 w-3.5" />
              </span>
              Compare two steps
            </CardTitle>
            <p className="chart-card__description">
              Find the layers whose weight distributions changed most between checkpoints. The
              Δgrad badge shows each layer's gradient-norm change between the two steps.
            </p>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-end gap-3 sm:gap-4">
            <div className="flex min-w-[9rem] flex-1 flex-col gap-2 sm:max-w-[12rem]">
              <label htmlFor="diff-step-a" className="control-label">
                Step A
              </label>
              <input
                id="diff-step-a"
                type="number"
                min="0"
                value={stepA}
                onChange={(e) => setStepA(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="e.g. 4400"
                className="control-input w-full"
              />
            </div>

            <div className="flex min-w-[9rem] flex-1 flex-col gap-2 sm:max-w-[12rem]">
              <label htmlFor="diff-step-b" className="control-label">
                Step B
              </label>
              <input
                id="diff-step-b"
                type="number"
                min="0"
                value={stepB}
                onChange={(e) => setStepB(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="e.g. 4450"
                className="control-input w-full"
              />
            </div>

            <Button onClick={handleCompare} disabled={loading} className="min-w-[7.5rem]">
              {loading ? 'Comparing…' : 'Compare'}
            </Button>
          </div>
          <div className="mt-4 flex flex-wrap items-center gap-2">
            <span className="control-hint">Quick compare</span>
            <Button
              variant="muted"
              size="sm"
              onClick={() => setComparison(previousStep, latestStep)}
              disabled={latestStep == null || previousStep == null || previousStep === latestStep}
            >
              Latest pair
            </Button>
            <Button
              variant="muted"
              size="sm"
              onClick={() => setComparison(availableSteps[0], latestStep)}
              disabled={availableSteps.length < 2}
            >
              First vs latest
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleSwap}
              disabled={!stepA || !stepB}
              title="Swap steps"
            >
              <ArrowLeftRight className="h-3.5 w-3.5" />
              Swap
            </Button>
          </div>
        </CardContent>
      </Card>

      {loading && <LoadingSpinner message="Comparing steps…" />}
      <ErrorMessage message={error} onRetry={handleCompare} />

      {!loading && diffData.length === 0 && compared && !error && (
        <EmptyState icon={<GitCompare className="h-5 w-5" />}>
          No layer data available for the selected steps.
        </EmptyState>
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
                      <div className="flex shrink-0 items-center gap-1.5">
                        {layer.grad_norm_change != null && (
                          <Badge
                            variant={
                              Math.abs(layer.grad_norm_change) > 1
                                ? 'danger'
                                : layer.grad_norm_change > 0
                                  ? 'warning'
                                  : 'default'
                            }
                          >
                            Δgrad {layer.grad_norm_change > 0 ? '+' : ''}
                            {layer.grad_norm_change.toFixed(2)}
                          </Badge>
                        )}
                        <Badge variant={index === 0 ? 'danger' : index === 1 ? 'warning' : 'accent'}>
                          {layer.kl_divergence.toFixed(4)}
                        </Badge>
                      </div>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>

            <ChartCard
              className="lg:col-span-2"
              title={`Distribution divergence: ${compared.a} vs ${compared.b}`}
              description="Red marks the three layers with the largest KL divergence."
            >
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
                  height: Math.max(400, 30 * layers.length + 120),
                  margin: { l: 180, r: 20, t: 24, b: 40 },
                  yaxis: {
                    automargin: true,
                    categoryorder: 'array',
                    categoryarray: [...layers].reverse(),
                  },
                }}
              />
            </ChartCard>
          </div>
        </>
      )}
    </div>
  )
}
