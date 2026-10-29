import { useEffect, useState } from 'react'
import { useRun } from '../RunContext.jsx'
import { fetchDiff } from '../api.js'
import ErrorMessage from '../components/ErrorMessage.jsx'
import EmptyState from '../components/EmptyState.jsx'
import { Button } from '../components/ui/Button.jsx'
import { Panel } from '../components/ui/Panel.jsx'
import { Skeleton } from '../components/ui/Skeleton.jsx'
import { StatStrip } from '../components/ui/StatStrip.jsx'

const fmt = (v) => (Number.isFinite(v) ? String(+v.toPrecision(4)) : '—')
const inputClass =
  'h-8 w-32 rounded-control border border-border bg-surface px-2.5 text-sm placeholder:text-subtle'

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
      setError('Enter two step numbers to compare.')
      return
    }
    if (a === b) {
      setError('Choose two different steps to compare.')
      return
    }
    setError(null)
    setLoading(true)
    try {
      setDiffData(await fetchDiff(a, b))
      setCompared({ a, b })
    } catch (err) {
      setError(err?.message || 'Failed to fetch diff data.')
    } finally {
      setLoading(false)
    }
  }

  const setComparison = (a, b) => {
    if (a == null || b == null) return
    setStepA(String(a))
    setStepB(String(b))
    setError(null)
  }

  const kls = diffData.map((d) => d.kl_divergence)
  const maxKL = kls.length ? Math.max(...kls) : 0
  const totalKL = kls.reduce((a, b) => a + b, 0)

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Compare steps</h1>
        <p className="mt-1 text-sm text-muted">
          Layers whose weight distributions changed most between two recorded steps.
        </p>
      </div>

      <div className="flex flex-wrap items-end gap-3 rounded-panel border border-border bg-surface px-4 py-3">
        <label className="flex flex-col gap-1 text-xs text-muted" htmlFor="diff-step-a">
          Step A
          <input
            id="diff-step-a"
            type="number"
            min="0"
            value={stepA}
            onChange={(e) => setStepA(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleCompare()}
            className={inputClass}
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted" htmlFor="diff-step-b">
          Step B
          <input
            id="diff-step-b"
            type="number"
            min="0"
            value={stepB}
            onChange={(e) => setStepB(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleCompare()}
            className={inputClass}
          />
        </label>
        <Button onClick={handleCompare} disabled={loading}>
          {loading ? 'Comparing…' : 'Compare'}
        </Button>
        <div className="ml-auto flex flex-wrap gap-2">
          <Button
            variant="secondary"
            size="sm"
            onClick={() => setComparison(previousStep, latestStep)}
            disabled={latestStep == null || previousStep === latestStep}
          >
            Latest pair
          </Button>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => setComparison(availableSteps[0], latestStep)}
            disabled={availableSteps.length < 2}
          >
            First vs latest
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setComparison(stepB, stepA)}
            disabled={!stepA || !stepB}
          >
            Swap
          </Button>
        </div>
      </div>

      {loading && <Skeleton className="h-64" />}
      <ErrorMessage message={error} onRetry={handleCompare} />

      {!loading && diffData.length === 0 && compared && !error && (
        <EmptyState>
          No layer data recorded for steps {compared.a} and {compared.b}.
        </EmptyState>
      )}

      {!loading && diffData.length > 0 && compared && (
        <>
          <StatStrip
            items={[
              { label: 'Step A', value: compared.a },
              { label: 'Step B', value: compared.b },
              { label: 'Max KL divergence', value: fmt(maxKL) },
              { label: 'Total KL divergence', value: fmt(totalKL) },
            ]}
          />
          <Panel
            title={`Distribution divergence, step ${compared.a} vs ${compared.b}`}
            bodyClassName="overflow-x-auto"
          >
            <table className="w-full min-w-[560px] text-sm">
              <thead>
                <tr className="text-left text-xs text-muted">
                  <th className="px-4 py-2 font-normal">Layer</th>
                  <th className="w-1/3 px-4 py-2 font-normal">KL divergence</th>
                  <th className="px-4 py-2 text-right font-normal">Δ grad norm</th>
                </tr>
              </thead>
              <tbody>
                {diffData.map((d, i) => (
                  <tr key={d.layer} className="border-t border-border">
                    <td className="max-w-[24rem] truncate px-4 py-2" title={d.layer}>
                      {d.layer}
                    </td>
                    <td className="px-4 py-2">
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 flex-1 rounded-full bg-hover">
                          <div
                            className={`h-1.5 rounded-full ${i < 3 ? 'bg-red' : 'bg-ink'}`}
                            style={{ width: `${maxKL > 0 ? (d.kl_divergence / maxKL) * 100 : 0}%` }}
                          />
                        </div>
                        <span className="w-16 text-right">{fmt(d.kl_divergence)}</span>
                      </div>
                    </td>
                    <td
                      className={`px-4 py-2 text-right ${
                        Math.abs(d.grad_norm_change ?? 0) > 1 ? 'text-red' : 'text-muted'
                      }`}
                    >
                      {d.grad_norm_change == null
                        ? '—'
                        : `${d.grad_norm_change > 0 ? '+' : ''}${d.grad_norm_change.toFixed(2)}`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="border-t border-border px-4 py-2 text-xs text-muted">
              Red marks the three layers with the largest KL divergence.
            </p>
          </Panel>
        </>
      )}
    </div>
  )
}
