import { useEffect, useMemo, useState } from 'react'
import { RotateCcw } from 'lucide-react'
import { useRun } from '../RunContext.jsx'
import { fetchReplay } from '../api.js'
import Chart from '../components/Chart.jsx'
import ErrorMessage from '../components/ErrorMessage.jsx'
import EmptyState from '../components/EmptyState.jsx'
import { Skeleton } from '../components/ui/Skeleton.jsx'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card.jsx'
import { Badge } from '../components/ui/Badge.jsx'
import { StatCard } from '../components/ui/StatCard.jsx'

export default function Replay() {
  const { globalData } = useRun()
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    fetchReplay()
      .then((payload) => {
        if (!cancelled) setData(payload)
      })
      .catch((err) => {
        if (!cancelled) setError(err?.message || 'Failed to load replay plan.')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const skippedStepSet = useMemo(() => new Set(data?.skipped_steps ?? []), [data])

  // Loss curve with skipped steps highlighted: a skip that lands exactly on a
  // step whose batch_index caused the spike is the interesting one.
  const chartData = useMemo(() => {
    const loss = globalData
      .filter((r) => r.step != null && r.loss != null)
      .map((r) => ({ x: r.step, y: r.loss }))
    return [
      {
        x: loss.map((p) => p.x),
        y: loss.map((p) => p.y),
        name: 'loss',
        line: { color: '#38bdf8', width: 2 },
        mode: 'lines',
      },
      {
        x: loss.filter((p) => skippedStepSet.has(p.x)).map((p) => p.x),
        y: loss.filter((p) => skippedStepSet.has(p.x)).map((p) => p.y),
        name: 'skipped',
        mode: 'markers',
        marker: { color: '#f87171', size: 8, symbol: 'x' },
      },
    ]
  }, [globalData, skippedStepSet])

  if (loading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-16" />
        <Skeleton className="h-16" />
        <Skeleton className="h-80" />
      </div>
    )
  }

  if (error) {
    return <ErrorMessage message={error} />
  }

  const config = data?.config

  if (!config || !Array.isArray(data?.skipped_batches) || data.skipped_batches.length === 0) {
    return (
      <EmptyState icon={<RotateCcw className="h-5 w-5" />}>
        No replay plan found for this run. Generate one with{' '}
        <code className="rounded bg-foreground/5 px-1.5 py-0.5 font-mono text-xs">
          trainscope replay --checkpoint &lt;path&gt; --skip-batches ...
        </code>{' '}
        — the UI will show exactly which training steps those batches map to.
      </EmptyState>
    )
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <StatCard
          label="Skipped batches"
          value={data.skipped_batches.length}
          icon={RotateCcw}
        />
        <StatCard label="Steps to skip" value={data.skipped_steps.length} icon={RotateCcw} />
        <StatCard label="Total training steps" value={data.n_global_steps} icon={RotateCcw} />
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <span className="metric-card__icon" aria-hidden="true">
              <RotateCcw className="h-3.5 w-3.5" />
            </span>
            Replay plan
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid grid-cols-1 gap-2 text-sm sm:grid-cols-2">
            <div>
              <span className="text-muted">Checkpoint: </span>
              <span className="font-mono text-xs">{config.checkpoint}</span>
            </div>
            {config.generated_at && (
              <div>
                <span className="text-muted">Generated: </span>
                <span className="font-mono text-xs">{config.generated_at}</span>
              </div>
            )}
          </div>
          <div className="flex flex-wrap gap-1.5">
            {data.skipped_batches.map((batch) => (
              <Badge key={batch} variant="warning">
                batch {batch}
              </Badge>
            ))}
          </div>
          {config.notes && <p className="text-xs text-muted">{config.notes}</p>}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Loss with skipped steps</CardTitle>
          <p className="chart-card__description">
            Red × markers mark training steps whose batch_index is in the replay&apos;s skip set —
            the steps SkippingDataLoader will skip on resume.
          </p>
        </CardHeader>
        <CardContent>
          <Chart
            data={chartData}
            layout={{ height: 320, margin: { l: 52, r: 12, t: 8, b: 36 } }}
          />
        </CardContent>
      </Card>
    </div>
  )
}