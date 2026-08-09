import { useMemo } from 'react'
import { useRun } from '../RunContext.jsx'
import EmptyState from '../components/EmptyState.jsx'
import { Badge } from '../components/ui/Badge.jsx'
import { StatCard } from '../components/ui/StatCard.jsx'
import { Card } from '../components/ui/Card.jsx'
import { Skeleton } from '../components/ui/Skeleton.jsx'
import { FolderKanban, TrendingUp, Zap } from 'lucide-react'
import { cn } from '../utils.js'

function formatTime(iso) {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleString()
  } catch {
    return iso
  }
}

function formatLoss(value) {
  if (value == null || !Number.isFinite(value)) return '—'
  if (value >= 1000) return value.toExponential(2)
  return value.toFixed(4)
}

export default function Runs() {
  const { runs, loading, error, switchRun, activeRunName } = useRun()

  const sortedRuns = useMemo(
    () => [...runs].sort((a, b) => (b.start_time ?? '').localeCompare(a.start_time ?? '')),
    [runs]
  )

  if (loading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-16" />
        <Skeleton className="h-16" />
        <Skeleton className="h-16" />
      </div>
    )
  }

  if (error) {
    return <p className="text-sm text-danger">Failed to load runs: {error}</p>
  }

  if (sortedRuns.length === 0) {
    return (
      <EmptyState icon={<FolderKanban className="h-5 w-5" />}>
        No runs found in this directory.
      </EmptyState>
    )
  }

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <StatCard label="Runs" value={sortedRuns.length} icon={FolderKanban} />
        <StatCard
          label="Runs with spikes"
          value={sortedRuns.filter((r) => r.spike_count > 0).length}
          icon={Zap}
        />
        <StatCard
          label="Latest loss (last run)"
          value={formatLoss(sortedRuns[0]?.last_loss)}
          icon={TrendingUp}
        />
      </div>

      <div className="space-y-3">
        {sortedRuns.map((run) => {
          const isActive = run.is_active || run.name === activeRunName
          return (
            <button
              key={run.name}
              type="button"
              onClick={() => switchRun(run.name)}
              className={cn(
                'w-full rounded-lg border p-4 text-left transition-colors',
                isActive
                  ? 'border-accent/60 bg-accent/5'
                  : 'border-border bg-card hover:border-accent/30'
              )}
              aria-current={isActive ? 'page' : undefined}
            >
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-sm font-semibold">{run.name}</span>
                    {isActive && <Badge variant="accent">Active</Badge>}
                  </div>
                  <p className="mt-0.5 text-xs text-muted">
                    {run.model_name ?? 'Unknown model'}
                    {run.detector?.name ? ` · ${run.detector.name} detector` : ''}
                    {run.n_global_rows != null ? ` · ${run.n_global_rows} steps` : ''}
                    {' · started '}
                    {formatTime(run.start_time)}
                  </p>
                </div>
                <div className="flex shrink-0 items-center gap-4">
                  <div className="text-right">
                    <div className="text-[10px] uppercase tracking-wider text-muted">Last loss</div>
                    <div
                      className={cn(
                        'font-mono text-sm font-semibold',
                        run.last_loss != null && run.last_loss > 3
                          ? 'text-danger'
                          : 'text-foreground'
                      )}
                    >
                      {formatLoss(run.last_loss)}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-[10px] uppercase tracking-wider text-muted">Spikes</div>
                    <div className="flex items-center justify-end gap-1">
                      <Zap className="h-3 w-3" />
                      <span className="font-mono text-sm font-semibold">
                        {run.spike_count ?? 0}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </button>
          )
        })}
      </div>

      <p className="text-xs text-muted">
        Selecting a run switches the Timeline, Layer, Diff, and Spike views to that run. Click the
        rows above to compare runs at a glance.
      </p>
    </div>
  )
}
