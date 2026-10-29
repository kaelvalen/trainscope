import { useEffect, useMemo, useRef, useState } from 'react'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import './charts.css'
import { buildLineOptions, offScaleMarker, toFinite } from './chartOptions.js'

/**
 * Line chart on uPlot. Bands, markers, the threshold line and the off-scale
 * marker are drawn on the canvas; the parent owns the visible x range
 * (`xRange`) and receives drag selections through `onSelectRange`.
 */
export default function LineChart(props) {
  const { x, series, height = 200, yMax = null, syncKey, xRange, ariaLabel } = props
  const wrapRef = useRef(null)
  const plotRef = useRef(null)
  const tipRef = useRef(null)
  const latest = useRef(props)
  const [width, setWidth] = useState(0)

  const offScale = useMemo(
    () => (yMax != null && series[0] ? offScaleMarker(x, series[0].values, yMax) : null),
    [x, series, yMax]
  )
  latest.current = { ...props, offScale }

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
  }, [])

  const structure = series.map((s) => `${s.label}|${s.color}|${s.points ? 1 : 0}`).join(',')

  useEffect(() => {
    if (!width || !wrapRef.current) return undefined
    const opts = buildLineOptions(latest.current, { width }, () => latest.current)
    opts.cursor = {
      sync: syncKey ? { key: syncKey } : undefined,
      drag: { x: true, y: false, setScale: false },
      points: { size: 6 },
    }
    opts.hooks.setSelect = [
      (u) => {
        const { left, width: w } = u.select
        if (w > 2) {
          latest.current.onSelectRange?.([u.posToVal(left, 'x'), u.posToVal(left + w, 'x')])
        }
        u.setSelect({ left: 0, width: 0, top: 0, height: 0 }, false)
      },
    ]
    opts.hooks.setCursor = [(u) => updateTooltip(u, tipRef.current, latest.current)]
    const data = [latest.current.x, ...latest.current.series.map((s) => toFinite(s.values))]
    const plot = new uPlot(opts, data, wrapRef.current)
    plotRef.current = plot
    const range = latest.current.xRange
    if (range) plot.setScale('x', { min: range[0], max: range[1] })
    const reset = () => latest.current.onResetRange?.()
    plot.over.addEventListener('dblclick', reset)
    return () => {
      plot.over.removeEventListener('dblclick', reset)
      plot.destroy()
      plotRef.current = null
    }
  }, [width, height, structure, syncKey])

  const limitKey = JSON.stringify(props.limit ?? null)
  useEffect(() => {
    const plot = plotRef.current
    if (!plot) return
    // Re-range y on every data/clamp change (a live spike sets yMax), then restore the zoom.
    plot.setData([x, ...series.map((s) => toFinite(s.values))])
    if (xRange) plot.setScale('x', { min: xRange[0], max: xRange[1] })
  }, [x, series, xRange, yMax, limitKey])

  useEffect(() => {
    plotRef.current?.redraw(false)
  }, [props.bands, props.markers, props.limit, offScale])

  return (
    <div className="line-chart" role="img" aria-label={ariaLabel}>
      <div ref={wrapRef} />
      <div ref={tipRef} className="chart-tooltip" hidden />
    </div>
  )
}

function updateTooltip(u, tip, props) {
  const idx = u.cursor.idx
  props.onCursor?.(idx == null ? null : props.x[idx])
  if (!tip) return
  if (idx == null) {
    tip.hidden = true
    return
  }
  const rows = props.series
    .map((s) => {
      const v = s.values[idx]
      const shown = Number.isFinite(v) ? +v.toPrecision(4) : 'n/a'
      return `<div><span style="color:${u.series[props.series.indexOf(s) + 1].stroke()}">●</span> ${s.label} <b>${shown}</b></div>`
    })
    .join('')
  tip.innerHTML = `<div class="chart-tooltip-step">Step ${props.x[idx]}</div>${rows}`
  tip.hidden = false
  const left = u.valToPos(props.x[idx], 'x') + u.bbox.left / (window.devicePixelRatio || 1)
  const flip = left > u.width - 160
  tip.style.left = `${flip ? left - 12 : left + 12}px`
  tip.style.transform = flip ? 'translateX(-100%)' : 'none'
  tip.style.top = '8px'
}
