import { useEffect, useMemo, useState } from 'react'
import { useRun } from '../RunContext.jsx'
import { Button } from '../components/ui/Button.jsx'
import { Panel } from '../components/ui/Panel.jsx'
import { StatStrip } from '../components/ui/StatStrip.jsx'
import { StatusDot } from '../components/ui/StatusDot.jsx'
import { Skeleton } from '../components/ui/Skeleton.jsx'
import ErrorMessage from '../components/ErrorMessage.jsx'
import LineChart from '../components/charts/LineChart.jsx'
import { seriesColor } from '../theme.js'
import { fetchCompare, fetchClusters, fetchCounterexample } from '../api.js'

const fmt = (v) => (v == null ? '—' : Number.isFinite(v) ? String(+v.toPrecision(4)) : 'non-finite')

/** Summaries carry no per-step data, so the index can only tell spiked from stable. */
function summaryStatus(run) {
  // NaN/inf arrive as null, so the server flags them in last_loss_finite.
  if (run.last_loss_finite === false) return ['diverged', 'Diverged']
  if ((run.spike_count ?? 0) > 0) return ['spiked', 'Spiked']
  return ['stable', 'Stable']
}

const series = (points = [], key = 'loss') => ({
  x: points.map((p) => p.step),
  y: points.map((p) => p[key]),
})

export default function Runs() {
  const { runs, loading, error, switchRun, activeRunName } = useRun()
  const [selected, setSelected] = useState([])
  const [compareData, setCompareData] = useState(null)
  const [comparing, setComparing] = useState(false)
  const [compareError, setCompareError] = useState(null)
  const [clusters, setClusters] = useState(null)
  const [clusterError, setClusterError] = useState(null)
  const [counterexample, setCounterexample] = useState(null)
  const [counterexampleLoading, setCounterexampleLoading] = useState(false)
  const [counterexampleError, setCounterexampleError] = useState(null)

  useEffect(() => {
    let cancelled = false
    fetchClusters()
      .then((data) => !cancelled && setClusters(data))
      .catch((err) => !cancelled && setClusterError(err?.message || 'Failed to cluster runs.'))
    return () => {
      cancelled = true
    }
  }, [])

  const sortedRuns = useMemo(
    () => [...runs].sort((a, b) => (b.start_time ?? '').localeCompare(a.start_time ?? '')),
    [runs]
  )
  const clusterOf = useMemo(() => {
    const map = {}
    for (const cluster of clusters?.clusters ?? []) {
      for (const name of cluster.runs) map[name] = cluster.label
    }
    for (const name of clusters?.unclustered ?? []) map[name] = 'no signal'
    return map
  }, [clusters])

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
      setCompareData(await fetchCompare(selected))
    } catch (err) {
      setCompareError(err?.message || 'Failed to compare runs.')
    } finally {
      setComparing(false)
    }
  }

  async function handleCounterexample(runName) {
    setCounterexampleLoading(true)
    setCounterexampleError(null)
    try {
      setCounterexample(await fetchCounterexample(runName))
    } catch (err) {
      setCounterexampleError(err?.message || 'Failed to find a counterexample run.')
    } finally {
      setCounterexampleLoading(false)
    }
  }

  const open = (name) => {
    switchRun(name)
    window.location.hash = '#overview'
  }

  if (loading) {
    return (
      <div className="space-y-3">
        <Skeleton className="h-12" />
        <Skeleton className="h-64" />
      </div>
    )
  }
  if (error) return <ErrorMessage message={`Failed to load runs: ${error}`} />
  if (sortedRuns.length === 0) {
    return (
      <div className="rounded-panel border border-border bg-surface px-6 py-10">
        <p>No runs in this directory yet</p>
        <p className="mt-1 text-sm text-muted">
          Point <code>trainscope ui</code> at a directory of runs, or try{' '}
          <code>trainscope demo</code>.
        </p>
      </div>
    )
  }

  const spiked = sortedRuns.filter((r) => (r.spike_count ?? 0) > 0).length

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Runs</h1>
        <p className="mt-1 text-sm text-muted">
          {sortedRuns.length} runs in this directory, {spiked} with spikes
        </p>
      </div>

      <Panel
        title="All runs"
        bodyClassName="overflow-x-auto"
        actions={
          <Button
            size="sm"
            variant="secondary"
            onClick={handleCompare}
            disabled={selected.length < 2 || comparing}
          >
            {comparing ? 'Comparing…' : `Compare (${selected.length})`}
          </Button>
        }
      >
        <table className="w-full min-w-[640px] text-sm">
          <thead>
            <tr className="text-left text-xs text-muted">
              <th className="w-10 px-4 py-2 font-normal">
                <span className="sr-only">Select</span>
              </th>
              <th className="py-2 pr-4 font-normal">Run</th>
              <th className="py-2 pr-4 font-normal">Status</th>
              <th className="py-2 pr-4 text-right font-normal">Steps</th>
              <th className="py-2 pr-4 text-right font-normal">Last loss</th>
              <th className="py-2 pr-4 font-normal">Cluster</th>
            </tr>
          </thead>
          <tbody>
            {sortedRuns.map((run) => {
              const [kind, word] = summaryStatus(run)
              const isActive = run.is_active || run.name === activeRunName
              return (
                <tr key={run.name} className="border-t border-border hover:bg-hover">
                  <td className="px-4 py-2.5">
                    <input
                      type="checkbox"
                      className="h-4 w-4 accent-[var(--ink)]"
                      checked={selected.includes(run.name)}
                      onChange={() => toggleSelect(run.name)}
                      aria-label={`Select run ${run.name}`}
                    />
                  </td>
                  <td className="py-2.5 pr-4">
                    <button
                      type="button"
                      onClick={() => open(run.name)}
                      className="font-medium text-ink hover:underline"
                    >
                      {run.name}
                    </button>
                    {isActive && <span className="ml-2 text-xs text-muted">current</span>}
                    <div className="text-xs text-muted">
                      {[run.model_name, run.detector?.name && `${run.detector.name} detector`]
                        .filter(Boolean)
                        .join(', ')}
                    </div>
                  </td>
                  <td className="py-2.5 pr-4">
                    <span className="flex items-center gap-2">
                      <StatusDot kind={kind} />
                      {word}
                      {(run.spike_count ?? 0) > 0 && (
                        <span className="text-xs text-muted">
                          {run.spike_count} {run.spike_count === 1 ? 'spike' : 'spikes'}
                        </span>
                      )}
                    </span>
                  </td>
                  <td className="py-2.5 pr-4 text-right">{run.n_global_rows ?? '—'}</td>
                  <td className="py-2.5 pr-4 text-right">{fmt(run.last_loss)}</td>
                  <td className="py-2.5 pr-4 text-muted">{clusterOf[run.name] ?? '—'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </Panel>

      <ErrorMessage message={compareError} onRetry={handleCompare} />
      {compareData && <ComparePanel data={compareData} />}

      {clusterError && <ErrorMessage message={clusterError} />}
      {clusters?.clusters?.length > 0 && (
        <Panel title="Run behavior clusters" bodyClassName="p-0">
          <p className="px-4 pt-3 text-sm text-muted">
            Runs grouped by which early-warning signal fired first. The typical lead is the median
            time between that signal&apos;s first crossing and the loss explosion.
          </p>
          {clusters.clusters.map((cluster) => (
            <div
              key={cluster.label}
              className="border-t border-border px-4 py-3 first-of-type:mt-3"
            >
              <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
                <span className="font-medium">{cluster.label}</span>
                <span className="text-xs text-muted">{cluster.n_runs} runs</span>
                {cluster.typical_lead_steps != null && (
                  <span className="text-xs text-red">~{cluster.typical_lead_steps}-step lead</span>
                )}
                {cluster.discriminant_traits?.map((trait) => (
                  <span key={trait.field} className="text-xs text-muted">
                    distinct: {trait.field}={trait.value}
                  </span>
                ))}
                <span className="ml-auto text-xs text-muted">
                  {cluster.fired_signals.join(', ')}
                </span>
              </div>
              <div className="mt-2 flex flex-wrap items-center gap-2">
                {cluster.runs.map((name) => (
                  <Button
                    key={name}
                    variant={name === activeRunName ? 'secondary' : 'ghost'}
                    size="sm"
                    onClick={() => open(name)}
                  >
                    {name}
                  </Button>
                ))}
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={counterexampleLoading}
                  onClick={() => handleCounterexample(cluster.runs[0])}
                  className="ml-auto"
                >
                  {counterexampleLoading ? 'Finding…' : 'Why this & vs. stable'}
                </Button>
              </div>
              {cluster.loss_band?.steps?.length > 0 && (
                <div className="mt-2">
                  <LineChart
                    x={cluster.loss_band.steps}
                    height={120}
                    series={[
                      { label: 'Median', values: cluster.loss_band.median, color: 'var(--ink)' },
                      {
                        label: 'Upper quartile',
                        values: cluster.loss_band.upper,
                        color: 'var(--subtle)',
                        width: 1,
                      },
                      {
                        label: 'Lower quartile',
                        values: cluster.loss_band.lower,
                        color: 'var(--subtle)',
                        width: 1,
                      },
                    ]}
                    ariaLabel={`Normalised loss band for ${cluster.label}`}
                  />
                </div>
              )}
              {cluster.crossing_steps?.length > 0 && (
                <p className="mt-2 flex flex-wrap gap-x-2 text-xs text-muted">
                  <span>first signal at</span>
                  {cluster.crossing_steps.map((step, i) => (
                    <span key={`${step}-${i}`} className="text-ink">
                      step {step}
                    </span>
                  ))}
                </p>
              )}
            </div>
          ))}
          {clusters.unclustered?.length > 0 && (
            <p className="border-t border-border px-4 py-3 text-xs text-muted">
              {clusters.unclustered.length} stable run(s), no early-warning signal fired:{' '}
              {clusters.unclustered.join(', ')}
            </p>
          )}
        </Panel>
      )}

      {counterexampleError && <ErrorMessage message={counterexampleError} />}
      {counterexample && <CounterexamplePanel data={counterexample} />}
    </div>
  )
}

function CounterexamplePanel({ data }) {
  const q = series(data.loss_series?.[data.query_run])
  const s = series(data.loss_series?.[data.counterexample_run])
  const x = q.x.length >= s.x.length ? q.x : s.x
  const align = (from) => x.map((step) => from.y[from.x.indexOf(step)] ?? null)
  return (
    <Panel title={`Why ${data.query_run} vs. ${data.counterexample_run}`}>
      <p className="text-sm text-muted">
        The nearest stable run by config distance ({data.config_distance}).{' '}
        {data.first_signal
          ? `First signal: ${data.first_signal} at step ${data.crossing_step}.`
          : 'No early-warning signal fired in the query run.'}
      </p>
      <div className="mt-3">
        <LineChart
          x={x}
          height={240}
          series={[
            { label: data.query_run, values: align(q), color: seriesColor(1) },
            { label: data.counterexample_run, values: align(s), color: seriesColor(0) },
          ]}
          ariaLabel="Loss of the query run and its nearest stable run"
        />
      </div>
      {data.config_diff?.length > 0 && (
        <table className="mt-3 w-full text-sm">
          <thead>
            <tr className="text-left text-xs text-muted">
              <th className="py-2 pr-4 font-normal">Config field</th>
              <th className="py-2 pr-4 font-normal">{data.query_run}</th>
              <th className="py-2 pr-4 font-normal">{data.counterexample_run}</th>
            </tr>
          </thead>
          <tbody>
            {data.config_diff.map((entry) => (
              <tr key={entry.field} className="border-t border-border">
                <td className="py-2 pr-4 text-muted">{entry.field}</td>
                <td className="py-2 pr-4">{String(entry.query_value)}</td>
                <td className="py-2 pr-4">{String(entry.stable_value)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Panel>
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
  const allSteps = [
    ...new Set(names.flatMap((n) => (loss_series[n] ?? []).map((r) => r.step))),
  ].sort((a, b) => a - b)
  const lossOf = (name) => {
    const byStep = new Map((loss_series[name] ?? []).map((r) => [r.step, r.loss]))
    return allSteps.map((step) => byStep.get(step) ?? null)
  }
  const concNames = names.filter((n) => concentration_series?.[n]?.length > 0)
  const concSteps = [
    ...new Set(concNames.flatMap((n) => concentration_series[n].map((r) => r.step))),
  ].sort((a, b) => a - b)
  const concOf = (name) => {
    const byStep = new Map(concentration_series[name].map((r) => [r.step, r.max_share]))
    return concSteps.map((step) => byStep.get(step) ?? null)
  }

  return (
    <div className="space-y-4">
      <StatStrip
        items={[
          { label: 'Compared runs', value: names.length },
          {
            label: 'Divergence point',
            value: divergence ? `Step ${divergence.step}` : 'No durable divergence',
          },
          { label: 'Config differences', value: config_diff.length },
          { label: 'Common causes', value: common_cause.length },
        ]}
      />

      <Panel title="Loss curves">
        <p className="text-sm text-muted">
          {divergence
            ? `Runs durably separate at step ${divergence.step} (gap sustained for ${divergence.min_run}+ steps).`
            : 'No durable separation found across the selected runs.'}
        </p>
        <div className="mt-3">
          <LineChart
            x={allSteps}
            height={300}
            series={names.map((name, i) => ({
              label: name,
              values: lossOf(name),
              color: seriesColor(i),
            }))}
            markers={
              divergence ? [{ at: divergence.step, label: 'divergence', kind: 'spike' }] : []
            }
            ariaLabel="Loss curves of the compared runs"
          />
        </div>
      </Panel>

      {concNames.length > 0 && (
        <Panel title="Routing / addressing concentration">
          <p className="text-sm text-muted">
            Max expert or slot share per run. Crossing 0.6–0.85 means a concentrated router or
            addressor, the architecture-aware early-warning signal.
          </p>
          <div className="mt-3">
            <LineChart
              x={concSteps}
              height={240}
              series={concNames.map((name) => ({
                label: name,
                values: concOf(name),
                color: seriesColor(names.indexOf(name)),
              }))}
              limit={[
                { value: 0.6, label: '0.6' },
                { value: 0.85, label: '0.85' },
              ]}
              ariaLabel="Routing concentration of the compared runs"
            />
          </div>
        </Panel>
      )}

      {common_cause.length > 0 && (
        <Panel title="Common cause">
          <div className="space-y-2 text-sm">
            {common_cause.map((cause) => (
              <p key={cause.field}>
                <span className="font-medium">{cause.field}</span>
                {cause.field === 'max routing concentration' ? (
                  <>
                    : every run with spikes concentrated (peak{' '}
                    <span className="text-red">{JSON.stringify(cause.spiked_value)}</span>), every
                    stable run stayed diffuse (peak {JSON.stringify(cause.stable_value)}).
                  </>
                ) : (
                  <>
                    : every run with spikes has{' '}
                    <span className="text-red">{JSON.stringify(cause.spiked_value)}</span>, every
                    stable run has {JSON.stringify(cause.stable_value)}.
                  </>
                )}
              </p>
            ))}
          </div>
        </Panel>
      )}

      {config_diff.length > 0 && (
        <Panel title="Config differences" bodyClassName="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="text-xs text-muted">
                <th className="px-4 py-2 font-normal">Field</th>
                {names.map((name) => (
                  <th key={name} className="px-4 py-2 font-normal">
                    {name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {config_diff.map((diff) => (
                <tr key={diff.field} className="border-t border-border">
                  <td className="px-4 py-2">{diff.field}</td>
                  {names.map((name) => (
                    <td key={name} className="px-4 py-2">
                      {JSON.stringify(diff.values[name] ?? '—')}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      )}
    </div>
  )
}
