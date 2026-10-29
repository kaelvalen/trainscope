import { useEffect, useMemo, useState } from 'react'
import { useRun } from '../RunContext.jsx'
import { fetchReplay } from '../api.js'
import { CHART_COLORS } from '../theme.js'
import LineChart from '../components/charts/LineChart.jsx'
import ErrorMessage from '../components/ErrorMessage.jsx'
import EmptyState from '../components/EmptyState.jsx'
import { Panel } from '../components/ui/Panel.jsx'
import { Skeleton } from '../components/ui/Skeleton.jsx'
import { StatStrip } from '../components/ui/StatStrip.jsx'

export default function Replay() {
  const { globalData } = useRun()
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    fetchReplay()
      .then((payload) => !cancelled && setData(payload))
      .catch((err) => !cancelled && setError(err?.message || 'Failed to load replay plan.'))
      .finally(() => !cancelled && setLoading(false))
    return () => {
      cancelled = true
    }
  }, [])

  // A skip that lands on the step whose batch caused the spike is the interesting one.
  const chart = useMemo(() => {
    const skipped = new Set(data?.skipped_steps ?? [])
    const rows = globalData.filter((r) => r.step != null && r.loss != null)
    return {
      x: rows.map((r) => r.step),
      loss: rows.map((r) => r.loss),
      skipped: rows.map((r) => (skipped.has(r.step) ? r.loss : null)),
    }
  }, [globalData, data])

  if (loading) return <Skeleton className="h-80" />
  if (error) return <ErrorMessage message={error} />

  const config = data?.config
  if (!config || !Array.isArray(data?.skipped_batches) || data.skipped_batches.length === 0) {
    return (
      <EmptyState>
        No replay plan found for this run. Generate one with{' '}
        <code>trainscope replay --checkpoint &lt;path&gt; --skip-batches …</code> and this view
        shows which training steps those batches map to.
      </EmptyState>
    )
  }

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Replay</h1>
        <p className="mt-1 text-sm text-muted">
          Which training steps a resume with this plan will skip.
        </p>
      </div>

      <StatStrip
        items={[
          { label: 'Skipped batches', value: data.skipped_batches.length },
          { label: 'Steps to skip', value: data.skipped_steps.length },
          { label: 'Training steps', value: data.n_global_steps },
          { label: 'Generated', value: config.generated_at?.slice(0, 10) ?? '—' },
        ]}
      />

      <Panel title="Replay plan">
        <dl className="grid gap-x-6 gap-y-1 text-sm sm:grid-cols-[auto_1fr]">
          <dt className="text-muted">Checkpoint</dt>
          <dd className="break-all">{config.checkpoint}</dd>
          {config.generated_at && (
            <>
              <dt className="text-muted">Generated</dt>
              <dd>{config.generated_at}</dd>
            </>
          )}
          <dt className="text-muted">Batches</dt>
          <dd className="flex flex-wrap gap-x-3">
            {data.skipped_batches.map((batch) => (
              <span key={batch}>batch {batch}</span>
            ))}
          </dd>
        </dl>
        {config.notes && <p className="mt-2 text-sm text-muted">{config.notes}</p>}
      </Panel>

      <Panel title="Loss with skipped steps">
        <p className="text-sm text-muted">
          Red points are steps whose batch is in the skip set, the steps SkippingDataLoader skips on
          resume.
        </p>
        <div className="mt-3">
          <LineChart
            x={chart.x}
            height={280}
            series={[
              { label: 'Loss', values: chart.loss, color: CHART_COLORS.loss },
              { label: 'Skipped', values: chart.skipped, color: 'var(--red)', points: true },
            ]}
            ariaLabel="Loss with the steps the replay skips"
          />
        </div>
      </Panel>
    </div>
  )
}
