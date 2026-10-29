import { readToken } from '../../theme.js'

const FONT = '11px "Geist Variable", system-ui, sans-serif'

/** Canvas needs real colors: resolve `var(--token)` through the stylesheet. */
export function resolveColor(color, fallback = '#737373') {
  if (typeof color !== 'string') return fallback
  if (!color.startsWith('var(')) return color
  return readToken(color.slice(4, -1).trim()) || fallback
}

export const toFinite = (values) => values.map((v) => (Number.isFinite(v) ? v : null))

/** 1.25x the largest finite loss before the first spike, so one explosion can't flatten the history. */
export function lossYMax(steps, loss, firstSpike) {
  if (firstSpike == null) return null
  let max = -Infinity
  steps.forEach((step, i) => {
    const v = loss[i]
    if (step < firstSpike && Number.isFinite(v)) max = Math.max(max, v)
  })
  return Number.isFinite(max) ? max * 1.25 : null
}

/** Where the line leaves the plot, labelled with the peak it reaches (or the first non-finite point). */
export function offScaleMarker(steps, values, yMax) {
  if (yMax == null) return null
  const i = values.findIndex((v) => (Number.isFinite(v) ? v > yMax : v != null))
  if (i < 0) return null
  if (!Number.isFinite(values[i])) return { at: steps[i], text: `non-finite at step ${steps[i]}` }
  let peak = i
  for (let j = i; j < values.length; j++) {
    if (Number.isFinite(values[j]) && values[j] > values[peak]) peak = j
  }
  return { at: steps[i], text: `${values[peak].toFixed(1)} at step ${steps[peak]}, off scale` }
}

function yRange(yMax, limit) {
  return (_u, min, max) => {
    if (min == null || max == null) return [0, yMax ?? 1]
    if (yMax != null) return [Math.min(min, yMax), yMax]
    const top = limit != null ? Math.max(max, limit.value * 1.1) : max
    const span = top - min || Math.abs(top) || 1
    return [min - span * 0.05, top + span * 0.05]
  }
}

/** Marker labels sit under the series so a line passing behind them stays unbroken. */
function drawMarkerLabels(u, markers) {
  const { ctx, bbox } = u
  const dpr = window.devicePixelRatio || 1
  ctx.save()
  ctx.font = FONT.replace('11px', `${11 * dpr}px`)
  ctx.textAlign = 'right'
  for (const m of markers) {
    if (!m.label) continue
    const x = u.valToPos(m.at, 'x', true)
    if (x < bbox.left || x > bbox.left + bbox.width) continue
    ctx.fillStyle = resolveColor(m.kind === 'spike' ? 'var(--red)' : 'var(--muted)')
    ctx.fillText(m.label, x - 5 * dpr, bbox.top + bbox.height - 6 * dpr)
  }
  ctx.restore()
}

function drawBands(u, bands) {
  const { ctx, bbox } = u
  ctx.save()
  for (const band of bands) {
    const x0 = Math.max(bbox.left, u.valToPos(band.from, 'x', true))
    const x1 = Math.min(bbox.left + bbox.width, u.valToPos(band.to, 'x', true))
    if (x1 <= x0) continue
    ctx.fillStyle = resolveColor(band.kind === 'spike' ? 'var(--red-band)' : 'var(--warm-band)')
    ctx.fillRect(x0, bbox.top, x1 - x0, bbox.height)
  }
  ctx.restore()
}

function labelText(ctx, text, x, y, color, dpr) {
  const w = ctx.measureText(text).width
  ctx.fillStyle = resolveColor('var(--surface)', '#ffffff')
  ctx.fillRect(x - w - 3 * dpr, y - 8 * dpr, w + 6 * dpr, 12 * dpr)
  ctx.fillStyle = color
  ctx.textAlign = 'right'
  ctx.fillText(text, x, y)
}

function drawOverlays(u, { markers = [], limit, offScale }) {
  const { ctx, bbox } = u
  const dpr = window.devicePixelRatio || 1
  ctx.save()
  ctx.font = FONT.replace('11px', `${11 * dpr}px`)
  ctx.lineWidth = dpr
  if (limit) {
    const y = u.valToPos(limit.value, 'y', true)
    ctx.strokeStyle = resolveColor('var(--subtle)')
    ctx.setLineDash([4 * dpr, 3 * dpr])
    ctx.beginPath()
    ctx.moveTo(bbox.left, y)
    ctx.lineTo(bbox.left + bbox.width, y)
    ctx.stroke()
  }
  for (const m of markers) {
    const x = u.valToPos(m.at, 'x', true)
    if (x < bbox.left || x > bbox.left + bbox.width) continue
    const color = resolveColor(m.kind === 'spike' ? 'var(--red)' : 'var(--muted)')
    ctx.strokeStyle = color
    ctx.setLineDash(m.kind === 'spike' ? [] : [3 * dpr, 3 * dpr])
    ctx.beginPath()
    ctx.moveTo(x, bbox.top)
    ctx.lineTo(x, bbox.top + bbox.height)
    ctx.stroke()
  }
  if (offScale) {
    const x = u.valToPos(offScale.at, 'x', true)
    ctx.setLineDash([])
    ctx.fillStyle = resolveColor('var(--red)')
    ctx.beginPath()
    ctx.moveTo(x, bbox.top + 2 * dpr)
    ctx.lineTo(x - 4 * dpr, bbox.top + 8 * dpr)
    ctx.lineTo(x + 4 * dpr, bbox.top + 8 * dpr)
    ctx.fill()
    labelText(ctx, offScale.text, x - 8 * dpr, bbox.top + 10 * dpr, ctx.fillStyle, dpr)
  }
  ctx.restore()
}

/**
 * Pure uPlot options for a line chart. `overlays()` returns the latest bands,
 * markers, limit and off-scale marker so they can change without rebuilding.
 */
export function buildLineOptions(props, { width }, overlays = () => props) {
  const { series, height, yMax = null, limit = null, xAxis = true, gutter = 48 } = props
  const muted = resolveColor('var(--muted)')
  const grid = resolveColor('var(--border)')
  const axis = {
    stroke: muted,
    font: FONT,
    grid: { stroke: grid, width: 1 },
    ticks: { show: false },
  }
  return {
    width,
    height,
    padding: [8, 10, xAxis ? 0 : 4, 0],
    legend: { show: series.length > 1 },
    scales: { x: { time: false }, y: { range: yRange(yMax, limit) } },
    axes: [
      { ...axis, show: xAxis, grid: { show: false }, size: 24 },
      { ...axis, size: gutter, values: (_u, ticks) => ticks.map((t) => +t.toPrecision(3)) },
    ],
    series: [
      { label: 'Step' },
      ...series.map((s) => ({
        label: s.label,
        stroke: resolveColor(s.color),
        width: s.width ?? 2,
        dash: s.dash,
        spanGaps: false,
        points: { show: Boolean(s.points), size: 6, fill: resolveColor(s.color) },
        ...(s.points ? { paths: () => null } : {}),
      })),
    ],
    hooks: {
      drawClear: [
        (u) => {
          drawBands(u, overlays().bands || [])
          drawMarkerLabels(u, overlays().markers || [])
        },
      ],
      draw: [(u) => drawOverlays(u, overlays())],
    },
  }
}
