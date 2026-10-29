import { useEffect, useRef, useState } from 'react'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import './charts.css'
import { resolveColor, toFinite } from './chartOptions.js'

const FONT = '11px "Geist Variable", system-ui, sans-serif'

/** Bar chart on uPlot. `x` may be labels (categories) or numbers (histogram bin starts). */
export default function BarChart({ x, series, height = 220, ariaLabel, gutter = 48 }) {
  const wrapRef = useRef(null)
  const [width, setWidth] = useState(0)

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

  useEffect(() => {
    if (!width || !wrapRef.current) return undefined
    const index = x.map((_, i) => i)
    const label = (i) => {
      const v = x[i]
      return typeof v === 'number' ? +v.toPrecision(3) : String(v ?? '')
    }
    const muted = resolveColor('var(--muted)')
    const bars = uPlot.paths.bars({ size: [0.8, 48], gap: 2 })
    const plot = new uPlot(
      {
        width,
        height,
        padding: [8, 10, 0, 0],
        legend: { show: series.length > 1, live: false },
        cursor: { drag: { x: false, y: false }, points: { show: false } },
        scales: { x: { time: false, range: (_u, min, max) => [min - 0.5, max + 0.5] } },
        axes: [
          {
            stroke: muted,
            font: FONT,
            grid: { show: false },
            ticks: { show: false },
            size: 28,
            values: (_u, ticks) => ticks.map((t) => (Number.isInteger(t) ? label(t) : '')),
          },
          {
            stroke: muted,
            font: FONT,
            size: gutter,
            grid: { stroke: resolveColor('var(--border)'), width: 1 },
            ticks: { show: false },
          },
        ],
        series: [
          { label: 'Bin' },
          ...series.map((s) => ({
            label: s.label,
            fill: resolveColor(s.color),
            stroke: resolveColor(s.color),
            width: 0,
            paths: bars,
            points: { show: false },
          })),
        ],
      },
      [index, ...series.map((s) => toFinite(s.values))],
      wrapRef.current
    )
    return () => plot.destroy()
  }, [width, height, x, series, gutter])

  return (
    <div className="line-chart" role="img" aria-label={ariaLabel}>
      <div ref={wrapRef} />
    </div>
  )
}
