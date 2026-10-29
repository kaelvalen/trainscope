import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { fetchLayerMatrix, fetchSignals } from '../api.js'
import { useRun } from '../RunContext.jsx'
import { CHART_COLORS } from '../theme.js'
import LayerHeatmap from '../components/charts/LayerHeatmap.jsx'
import LineChart from '../components/charts/LineChart.jsx'
import { lossYMax } from '../components/charts/chartOptions.js'
import { Button } from '../components/ui/Button.jsx'
import { Panel } from '../components/ui/Panel.jsx'
import { StatStrip } from '../components/ui/StatStrip.jsx'
import { StatusDot } from '../components/ui/StatusDot.jsx'
import { runStatus } from '../utils/runStatus.js'
import { spikeBands } from '../utils/spikeCluster.js'
import { computeWarmupBand } from '../utils/warmup.js'

const GUTTER = 132
const MATRIX_REFRESH_MS = 5000
const ZOOM_DEBOUNCE_MS = 250
const LIVE_WINDOW_MS = 30000
const NO_MARKERS = []
const SIGNAL_LABELS = {
  kurtosis: 'Activation kurtosis',
  grad_norm: 'Grad norm',
  concentration: 'Routing concentration',
  loss: 'Loss',
}

// Rows always carry these fields; null is how JSON delivers NaN/inf.
const fmt = (v) => (Number.isFinite(v) ? String(+v.toPrecision(4)) : 'non-finite')

function median(values) {
  const s = [...values].sort((a, b) => a - b)
  const m = Math.floor(s.length / 2)
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2
}

/** median + 3 × 1.4826 × MAD of the pre-spike grad norms (the same rule as first_crossing_step). */
function threeSigma(values) {
  const finite = values.filter(Number.isFinite)
  if (finite.length < 2) return null
  const center = median(finite)
  const mad = median(finite.map((v) => Math.abs(v - center)))
  return center + 3 * 1.4826 * mad
}

function spikeWindowText(events) {
  if (events.length === 0) return 'No spike'
  const first = events[0]
  const range =
    first.endStep > first.startStep ? `${first.startStep}–${first.endStep}` : `${first.startStep}`
  return events.length === 1 ? range : `${events.length} spikes, first ${range}`
}

/** "Training" only while steps keep arriving, not merely while the socket is open. */
function useGrowing(length, liveStatus) {
  const previous = useRef(length)
  const [lastGrowth, setLastGrowth] = useState(0)
  useEffect(() => {
    if (length > previous.current && previous.current > 0) setLastGrowth(Date.now())
    previous.current = length
  }, [length])
  return liveStatus === 'connected' && Date.now() - lastGrowth < LIVE_WINDOW_MS
}

/**
 * Layer matrix for the visible range. One request at a time: steps arriving
 * while it is in flight queue a single follow-up (throttled to
 * MATRIX_REFRESH_MS); a zoom change refetches after a short debounce. A result
 * is only discarded on unmount, never because newer steps arrived.
 */
function useLayerMatrix(length, range) {
  const [matrix, setMatrix] = useState(null)
  const s = useRef({ inflight: false, queued: null, last: 0, timer: null, mounted: true, range })
  s.current.range = range

  const fire = useCallback(() => {
    const st = s.current
    st.inflight = true
    st.last = Date.now()
    const r = st.range
    fetchLayerMatrix(r ? { start: r[0], end: r[1] } : undefined)
      .then((m) => st.mounted && setMatrix(m))
      .catch(() => st.mounted && setMatrix((m) => m ?? { layers: [] }))
      .finally(() => {
        st.inflight = false
        if (st.mounted && st.queued != null) {
          const immediate = st.queued
          st.queued = null
          schedule(immediate)
        }
      })
    // schedule is defined below and stable; listing it would create a cycle.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const schedule = useCallback(
    (immediate) => {
      const st = s.current
      if (st.inflight) {
        st.queued = st.queued || immediate
        return
      }
      clearTimeout(st.timer)
      const wait = immediate ? ZOOM_DEBOUNCE_MS : MATRIX_REFRESH_MS - (Date.now() - st.last)
      if (wait <= 0) fire()
      else st.timer = setTimeout(fire, wait)
    },
    [fire]
  )

  useEffect(() => {
    const st = s.current
    st.mounted = true
    return () => {
      st.mounted = false
      clearTimeout(st.timer)
    }
  }, [])
  useEffect(() => schedule(false), [length, schedule])
  const isFirst = useRef(true)
  useEffect(() => {
    if (isFirst.current) {
      isFirst.current = false
      return
    }
    schedule(true)
  }, [range?.[0], range?.[1], schedule]) // eslint-disable-line react-hooks/exhaustive-deps
  return matrix
}

export default function Overview() {
  const { meta, globalData, spikeEvents, runs = [], switchRun, liveStatus } = useRun()
  const runName = meta?.trainscope_config?.run_name || 'this run'
  const [signals, setSignals] = useState(null)
  const [xRange, setXRange] = useState(null)
  const [cursorStep, setCursorStep] = useState(null)
  const live = useGrowing(globalData.length, liveStatus)
  const matrix = useLayerMatrix(globalData.length, xRange)

  const spikeCount = spikeEvents.length
  useEffect(() => {
    let cancelled = false
    fetchSignals()
      .then((s) => !cancelled && setSignals(s))
      .catch(() => !cancelled && setSignals({ first_steps: {} }))
    return () => {
      cancelled = true
    }
  }, [spikeCount])

  const steps = useMemo(() => globalData.map((r) => r.step), [globalData])
  const losses = useMemo(() => globalData.map((r) => r.loss), [globalData])
  const gradNorms = useMemo(() => globalData.map((r) => r.grad_norm_before_clip), [globalData])
  const firstSpike = spikeEvents[0] ?? null
  const status = runStatus({ globalData, spikeEvents, live })

  const signalRows = useMemo(
    () =>
      Object.entries(signals?.first_steps ?? {})
        .sort((a, b) => a[1] - b[1])
        .map(([name, step]) => ({
          name,
          label: SIGNAL_LABELS[name] ?? name,
          step,
          lead: firstSpike && firstSpike.startStep - step >= 0 ? firstSpike.startStep - step : null,
        })),
    [signals, firstSpike]
  )
  const warnings = signalRows.filter((r) => r.name !== 'loss')
  const firstWarning = warnings.length ? warnings[0].step : null

  const firstStart = firstSpike?.startStep ?? null
  // Memoized so hovering (cursor state) does not re-sort or rebuild 100k-point arrays.
  const limit = useMemo(() => {
    const value = threeSigma(
      firstStart == null ? gradNorms : gradNorms.filter((_, i) => steps[i] < firstStart)
    )
    return value == null ? null : { value, label: '3σ' }
  }, [gradNorms, steps, firstStart])
  const yMax = useMemo(() => lossYMax(steps, losses, firstStart), [steps, losses, firstStart])
  const lossSeries = useMemo(
    () => [{ label: 'Loss', values: losses, color: CHART_COLORS.loss }],
    [losses]
  )
  const gradSeries = useMemo(
    () => [{ label: 'Grad norm', values: gradNorms, color: CHART_COLORS.gradNorm }],
    [gradNorms]
  )
  const domain = useMemo(() => [steps[0], steps[steps.length - 1]], [steps])
  const spikeWindow = useMemo(
    () => (firstSpike ? { from: firstSpike.startStep, to: firstSpike.endStep } : null),
    [firstSpike]
  )
  const warmup = useMemo(() => computeWarmupBand(meta, globalData), [meta, globalData])
  const bands = useMemo(
    () => [
      ...(warmup ? [{ from: warmup.startStep, to: warmup.endStep, kind: 'warmup' }] : []),
      ...spikeBands(spikeEvents),
    ],
    [warmup, spikeEvents]
  )
  const markers = useMemo(
    () =>
      firstWarning == null
        ? []
        : [{ at: firstWarning, label: `first warning, step ${firstWarning}`, kind: 'warning' }],
    [firstWarning]
  )

  if (globalData.length === 0) {
    return (
      <div className="rounded-panel border border-border bg-surface px-6 py-10">
        <p className="text-ink">Waiting for the first step from {runName}</p>
        <p className="mt-1 text-sm text-muted">
          Attach TrainScope to a training loop, or run <code>trainscope demo</code> to see an
          example.
        </p>
      </div>
    )
  }

  const last = globalData[globalData.length - 1]
  const chartProps = {
    x: steps,
    syncKey: 'overview',
    gutter: GUTTER,
    xRange,
    bands,
    markers,
    onSelectRange: ([a, b]) => setXRange([Math.round(a), Math.round(b)]),
    onResetRange: () => setXRange(null),
    onCursor: setCursorStep,
  }

  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_340px]">
      <div className="min-w-0 space-y-4">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <h1 className="truncate text-2xl font-semibold tracking-tight">{runName}</h1>
            <p className="mt-1 flex items-center gap-2 text-sm text-muted">
              <StatusDot kind={status.kind} />
              {status.detail}
            </p>
          </div>
          <div className="flex gap-2">
            {runs.length > 1 && (
              <Button variant="secondary" onClick={() => (window.location.hash = '#compare')}>
                Compare
              </Button>
            )}
            <Button onClick={() => (window.location.hash = '#replay')}>Open replay</Button>
          </div>
        </div>

        <StatStrip
          items={[
            { label: 'Final loss', value: fmt(last.loss) },
            { label: 'Grad norm', value: fmt(last.grad_norm_before_clip) },
            {
              label: 'First warning',
              value: firstWarning == null ? 'none' : `step ${firstWarning}`,
            },
            { label: 'Spike window', value: spikeWindowText(spikeEvents) },
          ]}
        />

        <Panel
          title="Loss and gradient norm"
          bodyClassName="px-2 pb-2 pt-3"
          actions={
            <>
              <span className="flex items-center gap-1.5">
                <i className="inline-block h-0.5 w-2.5 bg-ink" aria-hidden="true" />
                Loss
              </span>
              <span className="flex items-center gap-1.5">
                <i className="inline-block h-0.5 w-2.5 bg-blue" aria-hidden="true" />
                Grad norm
              </span>
              {xRange && (
                <button
                  type="button"
                  className="text-ink underline"
                  onClick={() => setXRange(null)}
                >
                  Reset zoom
                </button>
              )}
            </>
          }
        >
          <LineChart
            {...chartProps}
            series={lossSeries}
            height={200}
            xAxis={false}
            yMax={yMax}
            ariaLabel={`Loss over ${steps.length} steps`}
          />
          <LineChart
            {...chartProps}
            series={gradSeries}
            height={128}
            markers={NO_MARKERS}
            limit={limit}
            ariaLabel={`Gradient norm over ${steps.length} steps`}
          />
        </Panel>

        <Panel title="Gradient norm by layer" bodyClassName="px-2 pb-3 pt-3">
          <LayerHeatmap
            matrix={matrix}
            domain={domain}
            labelWidth={GUTTER}
            warningStep={firstWarning}
            spikeWindow={spikeWindow}
            xRange={xRange}
            cursorStep={cursorStep}
            onCellClick={(layer) => {
              window.location.hash = `#layers?layer=${encodeURIComponent(layer)}`
            }}
          />
        </Panel>
      </div>

      <aside className="space-y-4">
        <Panel title="Signals before the spike" bodyClassName="p-0">
          {signals == null ? (
            <p className="px-4 py-3 text-sm text-muted">Loading signals</p>
          ) : signalRows.length === 0 ? (
            <p className="px-4 py-3 text-sm text-muted">No signal crossed its threshold</p>
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-muted">
                  <th className="px-4 py-2 font-normal">Signal</th>
                  <th className="px-4 py-2 text-right font-normal">Crossed</th>
                  <th className="px-4 py-2 text-right font-normal">Lead</th>
                </tr>
              </thead>
              <tbody>
                {signalRows.map((row) => (
                  <tr key={row.name} className="border-t border-border">
                    <td className="px-4 py-2.5">{row.label}</td>
                    <td className="px-4 py-2.5 text-right">{row.step}</td>
                    <td
                      className={`px-4 py-2.5 text-right ${row.lead > 0 ? 'text-red' : 'text-muted'}`}
                    >
                      {row.lead ?? '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Panel>

        {runs.length > 1 && (
          <Panel title="Other runs" bodyClassName="p-0">
            <ul>
              {runs
                .filter((run) => run.name !== runName)
                .map((run) => (
                  <li key={run.name} className="border-t border-border first:border-t-0">
                    <button
                      type="button"
                      onClick={() => switchRun(run.name)}
                      className="flex min-h-[40px] w-full items-center gap-2 px-4 text-left text-sm hover:bg-hover"
                    >
                      <StatusDot kind={(run.spike_count ?? 0) > 0 ? 'spiked' : 'stable'} />
                      <span className="flex-1 truncate">{run.name}</span>
                      <span className="text-xs text-muted">
                        {(run.spike_count ?? 0) > 0
                          ? `${run.spike_count} ${run.spike_count === 1 ? 'spike' : 'spikes'}`
                          : `${run.n_global_rows ?? run.last_step ?? 0} steps`}
                      </span>
                    </button>
                  </li>
                ))}
            </ul>
          </Panel>
        )}
      </aside>
    </div>
  )
}
