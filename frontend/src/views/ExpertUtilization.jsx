import { useMemo } from 'react'
import { useRun } from '../RunContext.jsx'
import { SERIES_COLORS, seriesColor } from '../theme.js'
import LineChart from '../components/charts/LineChart.jsx'
import ErrorMessage from '../components/ErrorMessage.jsx'
import EmptyState from '../components/EmptyState.jsx'
import { Badge } from '../components/ui/Badge.jsx'
import { Panel } from '../components/ui/Panel.jsx'
import { Skeleton } from '../components/ui/Skeleton.jsx'
import { StatStrip } from '../components/ui/StatStrip.jsx'

const SLOTS = SERIES_COLORS.length

/** One series per expert; past the palette, the tail folds into a single "Other" share. */
function expertSeries(blockRows, nExperts) {
  const named = nExperts > SLOTS ? SLOTS - 1 : nExperts
  const series = Array.from({ length: named }, (_, expert) => ({
    label: `expert ${expert + 1}`,
    values: blockRows.map((r) => r.shares[expert] ?? null),
    color: seriesColor(expert),
    width: 1.5,
  }))
  if (nExperts > SLOTS) {
    series.push({
      label: `other ${nExperts - named} experts`,
      values: blockRows.map((r) => r.shares.slice(named).reduce((a, b) => a + (b ?? 0), 0)),
      color: 'var(--subtle)',
      width: 1.5,
    })
  }
  return series
}

export default function ExpertUtilization() {
  const { moeData, loading, error } = useRun()
  const rows = useMemo(() => moeData ?? [], [moeData])

  const blocks = useMemo(() => [...new Set(rows.map((r) => r.block))].sort(), [rows])
  const anyBlockConcentrated = useMemo(
    () =>
      blocks.some((block) => {
        const blockRows = rows.filter((r) => r.block === block)
        const last = blockRows[blockRows.length - 1]?.shares ?? []
        return last.length > 0 && Math.max(...last) >= 0.85
      }),
    [blocks, rows]
  )
  const nExperts = useMemo(
    () =>
      rows.reduce((max, r) => (Array.isArray(r.shares) ? Math.max(max, r.shares.length) : max), 0),
    [rows]
  )

  if (loading) return <Skeleton className="h-80" />
  if (error) return <ErrorMessage message={error} />
  if (rows.length === 0) {
    return (
      <EmptyState>
        No routing or addressing data in this run. This view appears for Mixtral-style models with a
        module named <code>router</code> (routing shares) or memory-augmented models with an{' '}
        <code>addressor</code> (memory-slot shares), recorded per step.
      </EmptyState>
    )
  }

  const latestShares = rows[rows.length - 1]?.shares ?? []
  const latestMax = latestShares.length ? Math.max(...latestShares) : 0

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Routing</h1>
        <p className="mt-1 text-sm text-muted">
          Share of tokens routed to each expert or memory slot, per step.
        </p>
      </div>

      <StatStrip
        items={[
          { label: 'Router blocks', value: blocks.length },
          { label: 'Experts', value: nExperts },
          { label: 'Latest max share', value: latestMax ? latestMax.toFixed(2) : '—' },
          { label: 'Concentrated', value: anyBlockConcentrated ? 'yes' : 'no' },
        ]}
      />

      {anyBlockConcentrated && (
        <Panel title="Expert utilization warning">
          <p className="text-sm">
            One expert currently receives {(latestMax * 100).toFixed(0)}% of tokens. In the v1.3.0
            experiment, routing concentration above 85% preceded loss divergence by 4–12 steps; if
            it persists, the <code>expert_utilization_drift</code> detector fires.
          </p>
        </Panel>
      )}

      {blocks.map((block) => {
        const blockRows = rows.filter((r) => r.block === block)
        const current = blockRows[blockRows.length - 1]?.shares ?? []
        const maxShare = current.length ? Math.max(...current) : 0
        return (
          <Panel key={block} title={`Routing shares in ${block}`}>
            <LineChart
              x={blockRows.map((r) => r.step)}
              height={260}
              series={expertSeries(blockRows, nExperts)}
              limit={[
                { value: 0.6, label: '0.6' },
                { value: 0.85, label: '0.85' },
              ]}
              ariaLabel={`Routing shares in ${block}`}
            />
            <div className="mt-3 flex flex-wrap items-center gap-2">
              {current.map((share, expert) => (
                <Badge
                  key={expert}
                  variant={share >= 0.85 ? 'danger' : share >= 0.6 ? 'warning' : 'default'}
                >
                  expert {expert + 1}: {share.toFixed(2)}
                </Badge>
              ))}
              {maxShare >= 0.85 && (
                <span className="text-sm text-red">Routing concentrated on one expert</span>
              )}
            </div>
          </Panel>
        )
      })}
    </div>
  )
}
