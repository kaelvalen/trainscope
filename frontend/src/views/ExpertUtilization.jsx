import { useMemo } from 'react'
import { Network } from 'lucide-react'
import { useRun } from '../RunContext.jsx'
import Chart from '../components/Chart.jsx'
import ErrorMessage from '../components/ErrorMessage.jsx'
import EmptyState from '../components/EmptyState.jsx'
import { Skeleton } from '../components/ui/Skeleton.jsx'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card.jsx'
import { ChartCard } from '../components/ui/ChartCard.jsx'
import { Badge } from '../components/ui/Badge.jsx'
import { StatCard } from '../components/ui/StatCard.jsx'

const EXPERT_PALETTE = [
  '#4ade80',
  '#60a5fa',
  '#fbbf24',
  '#f472b6',
  '#a78bfa',
  '#34d399',
  '#fb923c',
  '#22d3ee',
]

export default function ExpertUtilization() {
  const { moeData, loading, error } = useRun()
  const rows = moeData ?? []

  const blocks = useMemo(() => {
    const names = []
    for (const row of rows) {
      if (!names.includes(row.block)) names.push(row.block)
    }
    return names.sort()
  }, [rows])

  // A warning should fire if ANY block's latest routing is concentrated.
  const anyBlockConcentrated = useMemo(() => {
    return blocks.some((block) => {
      const blockRows = rows.filter((r) => r.block === block)
      const last = blockRows[blockRows.length - 1]?.shares ?? []
      return last.length > 0 && Math.max(...last) >= 0.85
    })
  }, [blocks, rows])

  const nExperts = useMemo(() => {
    let max = 0
    for (const row of rows) {
      if (Array.isArray(row.shares)) max = Math.max(max, row.shares.length)
    }
    return max
  }, [rows])

  if (loading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-40" />
        <Skeleton className="h-80" />
      </div>
    )
  }

  if (error) {
    return <ErrorMessage message={error} />
  }

  if (rows.length === 0) {
    return (
      <EmptyState icon={<Network className="h-5 w-5" />}>
        No routing or addressing data in this run. This panel appears for Mixtral-style models with
        a module named <code className="text-foreground">router</code> (routing shares) or
        memory-augmented models with an <code className="text-foreground">addressor</code>
        (memory-slot shares), recorded per step.
      </EmptyState>
    )
  }

  const latestShares = rows[rows.length - 1]?.shares ?? []
  const latestMax = latestShares.length ? Math.max(...latestShares) : 0

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <StatCard label="Router blocks" value={blocks.length} icon={Network} />
        <StatCard label="Experts" value={nExperts} icon={Network} />
        <StatCard
          label="Latest max share"
          value={latestMax ? latestMax.toFixed(2) : '—'}
          icon={Network}
        />
      </div>

      {blocks.map((block) => {
        const blockRows = rows.filter((r) => r.block === block)
        const steps = blockRows.map((r) => r.step)
        const current = blockRows[blockRows.length - 1]?.shares ?? []
        const maxShare = current.length ? Math.max(...current) : 0
        return (
          <ChartCard
            key={block}
            title={`Routing shares — ${block}`}
            description="Share of tokens routed to each expert / memory slot, per step."
          >
            <Chart
              data={Array.from({ length: nExperts }, (_, expert) => ({
                x: steps,
                y: blockRows.map((r) => r.shares[expert] ?? null),
                type: 'scatter',
                mode: 'lines',
                name: `expert ${expert + 1}`,
                line: { color: EXPERT_PALETTE[expert % EXPERT_PALETTE.length], width: 1.5 },
                connectgaps: false,
              }))}
              layout={{
                height: 300,
                legend: { orientation: 'h', y: -0.15 },
                yaxis: { range: [0, 1] },
                uirevision: `moe-${block}`,
              }}
            />
            <div className="mt-3 flex flex-wrap items-center gap-2">
              {current.map((share, expert) => (
                <Badge
                  key={expert}
                  variant={share >= 0.85 ? 'danger' : share >= 0.6 ? 'warning' : 'accent'}
                >
                  expert {expert + 1}: {share.toFixed(2)}
                </Badge>
              ))}
              {maxShare >= 0.85 && (
                <Badge variant="danger">Routing concentrated on one expert</Badge>
              )}
            </div>
          </ChartCard>
        )
      })}

      {anyBlockConcentrated && (
        <Card className="story-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm font-semibold text-accent">
              <Network className="h-4 w-4" />
              Expert utilization warning
            </CardTitle>
          </CardHeader>
          <CardContent className="text-xs leading-relaxed">
            One expert currently receives {latestMax.toFixed(0)}% of tokens. The v1.3.0 experiment
            showed routing concentration above 85% precedes loss divergence by 4–12 steps — if this
            persists, the <code className="text-foreground">expert_utilization_drift</code> detector
            fires.
          </CardContent>
        </Card>
      )}
    </div>
  )
}
