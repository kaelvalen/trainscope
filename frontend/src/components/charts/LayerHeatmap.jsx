import { useEffect, useMemo, useRef, useState } from 'react'
import { readToken } from '../../theme.js'
import {
  firstCrossings,
  groupRows,
  heatmapSummary,
  sortRows,
  zColor,
} from '../../utils/layerMatrix.js'
import './charts.css'

const ROW = 16
const AXIS = 20

function hexToRgb(hex, fallback) {
  const m = /^#?([0-9a-f]{6})$/i.exec((hex || '').trim())
  if (!m) return fallback
  const n = parseInt(m[1], 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

/**
 * Layer × step heatmap on canvas. Columns use the same x→px mapping as the
 * line charts above it (same label gutter, right pad and `domain`/`xRange`),
 * so a step lines up vertically across the chart and the heatmap.
 */
export default function LayerHeatmap({
  matrix,
  domain,
  labelWidth = 132,
  plotPadRight = 10,
  warningStep = null,
  spikeWindow = null,
  xRange = null,
  cursorStep = null,
  onCellClick,
}) {
  const wrapRef = useRef(null)
  const canvasRef = useRef(null)
  const tipRef = useRef(null)
  const [width, setWidth] = useState(0)
  const [expanded, setExpanded] = useState(() => new Set())
  const [sortMode, setSortMode] = useState('model')

  const starts = useMemo(() => matrix?.bucket_starts ?? [], [matrix])
  const grouped = useMemo(() => (matrix ? groupRows(matrix, { expanded }) : []), [matrix, expanded])
  const crossings = useMemo(() => firstCrossings(grouped, starts), [grouped, starts])
  const rows = useMemo(() => sortRows(grouped, sortMode, crossings), [grouped, sortMode, crossings])

  const [x0, x1] = xRange ?? domain ?? [starts[0] ?? 0, starts[starts.length - 1] ?? 1]
  const plotW = Math.max(1, width - labelWidth - plotPadRight)
  const toPx = (step) => labelWidth + ((step - x0) / Math.max(1e-9, x1 - x0)) * plotW
  const bucketEnd = (i) => (i + 1 < starts.length ? starts[i + 1] : (domain?.[1] ?? starts[i] + 1))

  const hasRows = rows.length > 0
  // The canvas wrapper only exists once rows do, so observe it when it appears.
  useEffect(() => {
    const el = wrapRef.current
    if (!el) return undefined
    if (typeof ResizeObserver === 'undefined') {
      setWidth(el.clientWidth)
      return undefined
    }
    const ro = new ResizeObserver(([entry]) => setWidth(Math.floor(entry.contentRect.width)))
    ro.observe(el)
    return () => ro.disconnect()
  }, [hasRows])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !width || rows.length === 0) return
    const dpr = window.devicePixelRatio || 1
    const height = rows.length * ROW + AXIS
    canvas.width = width * dpr
    canvas.height = height * dpr
    canvas.style.width = `${width}px`
    canvas.style.height = `${height}px`
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, width, height)

    const surface = hexToRgb(readToken('--surface'), [255, 255, 255])
    const heat = hexToRgb(readToken('--heat'), [155, 28, 34])
    const muted = readToken('--muted') || '#6b6b6b'
    const ink = readToken('--ink') || '#171717'
    const red = readToken('--red') || '#e5484d'
    const font = '11px "Geist Variable", system-ui, sans-serif'
    ctx.font = font
    ctx.textBaseline = 'middle'

    rows.forEach((row, r) => {
      const y = r * ROW
      ctx.fillStyle = muted
      ctx.textAlign = 'right'
      const prefix = row.isGroup ? (expanded.has(row.key) ? '▾ ' : '▸ ') : ''
      ctx.fillText(`${prefix}${row.label}`, labelWidth - 10, y + ROW / 2, labelWidth - 14)
      starts.forEach((start, b) => {
        const a = Math.max(labelWidth, toPx(start))
        const e = Math.min(labelWidth + plotW, toPx(bucketEnd(b)))
        if (e <= a) return
        ctx.fillStyle = zColor(row.z[b], surface, heat)
        ctx.fillRect(a, y + 1, Math.max(1, e - a - (e - a > 3 ? 1 : 0)), ROW - 2)
      })
    })

    const bottom = rows.length * ROW
    ctx.lineWidth = 1
    if (warningStep != null) {
      const x = toPx(warningStep)
      ctx.strokeStyle = muted
      ctx.setLineDash([3, 3])
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, bottom)
      ctx.stroke()
    }
    ctx.setLineDash([])
    if (spikeWindow) {
      const a = Math.max(labelWidth, toPx(spikeWindow.from))
      const e = Math.min(labelWidth + plotW, toPx(spikeWindow.to))
      if (e > a) {
        ctx.strokeStyle = red
        ctx.strokeRect(a + 0.5, 0.5, e - a - 1, bottom - 1)
      }
    }
    if (cursorStep != null && cursorStep >= x0 && cursorStep <= x1) {
      const x = toPx(cursorStep)
      ctx.strokeStyle = ink
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, bottom)
      ctx.stroke()
    }

    ctx.fillStyle = muted
    ctx.textAlign = 'center'
    const ticks = 5
    for (let i = 0; i < ticks; i++) {
      const step = Math.round(x0 + ((x1 - x0) * i) / (ticks - 1))
      ctx.fillText(String(step), toPx(step), bottom + AXIS / 2 + 2)
    }
  })

  if (!matrix) return <p className="px-2 text-sm text-muted">Loading layer metrics</p>
  if (matrix.layers.length === 0) {
    return <p className="px-2 text-sm text-muted">No layer metrics recorded for this run</p>
  }

  const hit = (event) => {
    const rect = canvasRef.current.getBoundingClientRect()
    const px = event.clientX - rect.left
    const r = Math.floor((event.clientY - rect.top) / ROW)
    if (r < 0 || r >= rows.length) return null
    const row = rows[r]
    if (px < labelWidth) return { row, bucket: null }
    const step = x0 + ((px - labelWidth) / plotW) * (x1 - x0)
    let b = -1
    for (let i = 0; i < starts.length; i++) if (starts[i] <= step) b = i
    return b < 0 ? { row, bucket: null } : { row, bucket: b, px, y: r * ROW }
  }

  const onMove = (event) => {
    const tip = tipRef.current
    const h = hit(event)
    if (!tip) return
    if (!h || h.bucket == null) {
      tip.hidden = true
      return
    }
    const v = h.row.values[h.bucket]
    const z = h.row.z[h.bucket]
    const end = bucketEnd(h.bucket) - 1
    const range = end > starts[h.bucket] ? `${starts[h.bucket]}–${end}` : `${starts[h.bucket]}`
    tip.innerHTML = `<div class="chart-tooltip-step">Step ${range}</div><div>${h.row.label}</div><div>value <b>${v == null ? 'n/a' : +v.toPrecision(4)}</b></div><div>z <b>${z == null ? 'n/a' : z.toFixed(1)}</b></div>`
    tip.hidden = false
    const flip = h.px > width - 180
    tip.style.left = `${flip ? h.px - 12 : h.px + 12}px`
    tip.style.transform = flip ? 'translateX(-100%)' : 'none'
    tip.style.top = `${h.y + ROW}px`
  }

  const onClick = (event) => {
    const h = hit(event)
    if (!h) return
    if (h.row.isGroup) {
      setExpanded((prev) => {
        const next = new Set(prev)
        if (next.has(h.row.key)) next.delete(h.row.key)
        else next.add(h.row.key)
        return next
      })
      return
    }
    if (h.bucket != null) onCellClick?.(h.row.key, starts[h.bucket])
  }

  return (
    <div className="layer-heatmap">
      <div className="flex justify-end pb-2">
        <button
          type="button"
          className="heatmap-toggle"
          aria-pressed={sortMode === 'first'}
          onClick={() => setSortMode((m) => (m === 'first' ? 'model' : 'first'))}
        >
          Sort by first crossing
        </button>
      </div>
      <div ref={wrapRef} className="relative">
        <canvas
          ref={canvasRef}
          role="img"
          aria-label={heatmapSummary(rows, crossings)}
          onMouseMove={onMove}
          onMouseLeave={() => tipRef.current && (tipRef.current.hidden = true)}
          onClick={onClick}
          style={{ cursor: 'pointer', display: 'block' }}
        />
        <div ref={tipRef} className="chart-tooltip" hidden />
      </div>
    </div>
  )
}
