import { useEffect, useMemo, useState } from 'react'
import { useRun } from '../RunContext.jsx'
import EmptyState from '../components/EmptyState.jsx'
import { Badge } from '../components/ui/Badge.jsx'
import { StatCard } from '../components/ui/StatCard.jsx'
import { Button } from '../components/ui/Button.jsx'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card.jsx'
import { ChartCard } from '../components/ui/ChartCard.jsx'
import { Skeleton } from '../components/ui/Skeleton.jsx'
import ErrorMessage from '../components/ErrorMessage.jsx'
import Chart from '../components/Chart.jsx'
import { FolderKanban, GitCompare, Layers, TrendingUp, Zap } from 'lucide-react'
import { cn } from '../utils.js'
import { CHART_COLORS } from '../theme.js'
import { fetchCompare, fetchClusters } from '../api.js'

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

const RUN_PALETTE = ['#4ade80', '#60a5fa', '#fbbf24', '#f472b6', '#a78bfa', '#34d399']

export default function Runs() {
  const { runs, loading, error, switchRun, activeRunName } = useRun()
  const [selected, setSelected] = useState([])
  const [compareData, setCompareData] = useState(null)
  const [comparing, setComparing] = useState(false)
  const [compareError, setCompareError] = useState(null)
  const [clusters, setClusters] = useState(null)
  const [clusterError, setClusterError] = useState(null)

  useEffect(() => {
    let cancelled = false
    fetchClusters()
      .then((data) => {
        if (!cancelled) setClusters(data)
      })
      .catch((err) => {
        if (!cancelled) setClusterError(err?.message || 'Failed to cluster runs.')
      })
    return () => {
      cancelled = true
    }
  }, [])

  const sortedRuns = useMemo(
    () => [...runs].sort((a, b) => (b.start_time ?? '').localeCompare(a.start_time ?? '')),
    [runs]
  )

  useEffect(() => {
    setSelected((prev) => prev.filter((name) => runs.some((r) => r.name === name)))
  }, [runs])

  function toggleSelect(name) {
    setCompareData(null)
    setCompareError(null)
    setSelected((prev) => (prev.includes(name) ? prev.filter((n) => n !== name) : [...prev, name]))
  }

  async function handleCompare() {
    if (selected.length < 2) return
    setComparing(true)
    setCompareError(null)
    try {
      const data = await fetchCompare(selected)
      setCompareData(data)
    } catch (err) {
      setCompareError(err?.message || 'Failed to compare runs.')
    } finally {
      setComparing(false)
    }
  }

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

      {clusterError && <ErrorMessage message={clusterError} />}

      {clusters && clusters.clusters?.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <span className="metric-card__icon" aria-hidden="true">
                <Layers className="h-3.5 w-3.5" />
              </span>
              Run behavior clusters
            </CardTitle>
            <p className="chart-card__description">
              Runs grouped by which early-warning signal fired first (v1.6.0 cascade: activation →
              gradient → routing → loss). Click a cluster to open its runs.
            </p>
          </CardHeader>
          <CardContent className="space-y-3">
            {clusters.clusters.map((cluster) => (
              <div
                key={cluster.label}
                className="rounded-lg border border-border bg-background p-3"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <Badge variant="accent">{cluster.label}</Badge>
                    <span className="text-xs text-muted">{cluster.n_runs} runs</span>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {cluster.fired_signals.map((signal) => (
                      <Badge key={signal} variant="muted">
                        {signal}
                      </Badge>
                    ))}
                  </div>
                </div>
                <div className="mt-2 flex flex-wrap gap-2">
                  {cluster.runs.map((name) => (
                    <Button
                      key={name}
                      variant="ghost"
                      size="sm"
                      onClick={() => switchRun(name)}
                      className={cn(
                        'font-mono text-xs',
                        name === activeRunName && 'border border-accent/60 text-accent'
                      )}
                    >
                      {name}
                    </Button>
                  ))}
                </div>
              </div>
            ))}
            {clusters.unclustered?.length > 0 && (
              <p className="text-xs text-muted">
                {clusters.unclustered.length} run(s) without signal data:
                {clusters.unclustered.join(', ')}
              </p>
            )}
          </CardContent>
        </Card>
      )}

      <Card className="control-card">
        <CardHeader>
          <div className="flex w-full flex-wrap items-center justify-between gap-3">
            <div>
              <CardTitle className="flex items-center gap-2">
                <span className="metric-card__icon" aria-hidden="true">
                  <GitCompare className="h-3.5 w-3.5" />
                </span>
                Compare runs
              </CardTitle>
              <p className="chart-card__description">
                Select two or more runs to compare loss curves, configs, and common causes.
              </p>
            </div>
            <Button onClick={handleCompare} disabled={selected.length < 2 || comparing}>
              {comparing ? 'Comparing…' : `Compare (${selected.length})`}
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          {sortedRuns.map((run) => {
            const isActive = run.is_active || run.name === activeRunName
            const isSelected = selected.includes(run.name)
            return (
              <label
                key={run.name}
                className={cn(
                  'flex w-full cursor-pointer items-center gap-3 rounded-lg border p-3 text-left transition-colors',
                  isSelected
                    ? 'border-accent/60 bg-accent/5'
                    : 'border-border bg-background hover:border-accent/30'
                )}
              >
                <input
                  type="checkbox"
                  className="h-4 w-4 shrink-0 accent-[var(--color-accent)]"
                  checked={isSelected}
                  onChange={() => toggleSelect(run.name)}
                  aria-label={`Select run ${run.name}`}
                />
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-mono text-sm font-semibold">{run.name}</span>
                    {isActive && <Badge variant="accent">Active</Badge>}
                    {run.spike_count > 0 && (
                      <Badge variant="danger">{run.spike_count} spikes</Badge>
                    )}
                  </div>
                  <p className="mt-0.5 truncate text-xs text-muted">
                    {run.model_name ?? 'Unknown model'}
                    {run.detector?.name ? ` · ${run.detector.name} detector` : ''}
                    {run.n_global_rows != null ? ` · ${run.n_global_rows} steps` : ''}
                    {' · started '}
                    {formatTime(run.start_time)}
                  </p>
                </div>
                <div className="shrink-0 text-right">
                  <div className="text-[10px] uppercase tracking-wider text-muted">Last loss</div>
                  <div
                    className={cn(
                      'font-mono text-sm font-semibold',
                      run.last_loss != null && run.last_loss > 3 ? 'text-danger' : 'text-foreground'
                    )}
                  >
                    {formatLoss(run.last_loss)}
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={(e) => {
                    e.preventDefault()
                    switchRun(run.name)
                  }}
                  title="Open this run"
                >
                  Open
                </Button>
              </label>
            )
          })}
        </CardContent>
      </Card>

      <ErrorMessage message={compareError} onRetry={handleCompare} />

      {compareData && <ComparePanel data={compareData} />}

      <p className="text-xs text-muted">
        Selecting a run with “Open” switches the Timeline, Layer, Diff, and Spike views to that run.
        Check boxes to compare runs side by side.
      </p>
    </div>
  )
}

function ComparePanel({ data }) {
  const {
    runs: names,
    loss_series,
    divergence,
    config_diff,
    common_cause,
    concentration_series,
  } = data
  const colorByName = Object.fromEntries(
    names.map((name, i) => [name, RUN_PALETTE[i % RUN_PALETTE.length]])
  )

  const traces = names.map((name) => ({
    x: loss_series[name]?.map((row) => row.step) ?? [],
    y: loss_series[name]?.map((row) => row.loss) ?? [],
    type: 'scatter',
    mode: 'lines',
    name,
    line: { color: colorByName[name], width: 1.5 },
  }))

  const hasConcentration = names.some((name) => (concentration_series?.[name]?.length ?? 0) > 0)
  const concentrationTraces = names
    .filter((name) => concentration_series?.[name]?.length > 0)
    .map((name) => ({
      x: concentration_series[name].map((row) => row.step),
      y: concentration_series[name].map((row) => row.max_share),
      type: 'scatter',
      mode: 'lines',
      name,
      line: { color: colorByName[name], width: 1.5 },
    }))

  const shapes = divergence
    ? [
        {
          type: 'line',
          xref: 'x',
          x0: divergence.step,
          x1: divergence.step,
          yref: 'paper',
          y0: 0,
          y1: 1,
          line: { color: '#f87171', width: 1.5, dash: 'dash' },
        },
      ]
    : []

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <StatCard label="Compared runs" value={names.length} icon={GitCompare} />
        <StatCard
          label="Divergence point"
          value={divergence ? `Step ${divergence.step}` : 'No durable divergence'}
          icon={TrendingUp}
        />
        <StatCard label="Config differences" value={config_diff.length} icon={FolderKanban} />
      </div>

      <ChartCard
        title="Loss curves"
        description={
          divergence
            ? `Runs durably separate at step ${divergence.step} (gap sustained for ${divergence.min_run}+ steps).`
            : 'No durable separation found across the selected runs.'
        }
      >
        <Chart
          data={traces}
          layout={{ height: 340, shapes, legend: { orientation: 'h', y: -0.15 } }}
        />
      </ChartCard>

      {hasConcentration && (
        <ChartCard
          title="Routing / addressing concentration"
          description="Max expert / slot share per run. A run crossing 0.6–0.85 has a concentrated router or addressor — the architecture-aware early-warning signal."
        >
          <Chart
            data={concentrationTraces}
            layout={{
              height: 300,
              legend: { orientation: 'h', y: -0.15 },
              yaxis: { range: [0, 1] },
              shapes: [
                {
                  type: 'line',
                  xref: 'paper',
                  x0: 0,
                  x1: 1,
                  yref: 'y',
                  y0: 0.6,
                  y1: 0.6,
                  line: { color: '#f87171', width: 1, dash: 'dot' },
                },
                {
                  type: 'line',
                  xref: 'paper',
                  x0: 0,
                  x1: 1,
                  yref: 'y',
                  y0: 0.85,
                  y1: 0.85,
                  line: { color: '#f87171', width: 1, dash: 'dash' },
                },
              ],
            }}
          />
        </ChartCard>
      )}

      {common_cause.length > 0 && (
        <Card className="story-card">
          <CardHeader>
            <CardTitle>Common cause</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-xs leading-relaxed">
            {common_cause.map((cause) => {
              const isConcentration = cause.field === 'max routing concentration'
              return (
                <p key={cause.field} className="rounded-md bg-background p-2">
                  <span className="font-mono">{cause.field}</span>
                  {isConcentration ? (
                    <>
                      : every run with spikes concentrated (peak{' '}
                      <span className="font-semibold text-danger">
                        {JSON.stringify(cause.spiked_value)}
                      </span>
                      ), every stable run stayed diffuse (peak{' '}
                      <span className="font-semibold text-foreground">
                        {JSON.stringify(cause.stable_value)}
                      </span>
                      ).
                    </>
                  ) : (
                    <>
                      : every run with spikes has{' '}
                      <span className="font-semibold text-danger">
                        {JSON.stringify(cause.spiked_value)}
                      </span>
                      , every stable run has{' '}
                      <span className="font-semibold text-foreground">
                        {JSON.stringify(cause.stable_value)}
                      </span>
                      .
                    </>
                  )}
                </p>
              )
            })}
          </CardContent>
        </Card>
      )}

      {config_diff.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Config differences</CardTitle>
          </CardHeader>
          <CardContent className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-border text-muted">
                  <th className="py-2 pr-4 font-medium">Field</th>
                  {names.map((name) => (
                    <th key={name} className="py-2 pr-4 font-mono font-medium">
                      {name}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {config_diff.map((diff) => (
                  <tr key={diff.field} className="border-b border-border/50 last:border-0">
                    <td className="py-2 pr-4 font-mono">{diff.field}</td>
                    {names.map((name) => (
                      <td key={name} className="py-2 pr-4 font-mono">
                        {JSON.stringify(diff.values[name] ?? '—')}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
