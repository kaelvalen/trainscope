import { lazy, Suspense } from 'react'
import { DARK_LAYOUT, PLOT_CONFIG } from '../theme.js'
import { cn } from '../utils.js'
import { Skeleton } from './ui/Skeleton.jsx'

/**
 * Consistent Plotly wrapper for the TrainScope dark theme.
 *
 * Accepts the same `data`, `layout`, and `config` props as react-plotly.js but
 * merges in the shared dark layout defaults so individual views stay small.
 *
 * Plotly is loaded lazily (~4.9MB raw / ~1.5MB gzipped), so it is split into
 * its own chunk and only fetched when the first chart actually renders; the
 * app shell, nav, and data loading do not block on it.
 */
const Plot = lazy(() => import('react-plotly.js'))

export default function Chart({ data, layout = {}, config = {}, className, ...rest }) {
  const mergedLayout = {
    ...DARK_LAYOUT,
    ...layout,
    autosize: true,
    font: { ...DARK_LAYOUT.font, ...layout.font },
    margin: { ...DARK_LAYOUT.margin, ...layout.margin },
    xaxis: { ...DARK_LAYOUT.xaxis, ...layout.xaxis },
    yaxis: { ...DARK_LAYOUT.yaxis, ...layout.yaxis },
  }

  const mergedConfig = { ...PLOT_CONFIG, ...config }

  return (
    <div className={cn('chart-frame w-full', className)}>
      <Suspense
        fallback={
          <Skeleton
            className="w-full"
            style={layout.height ? { height: layout.height } : undefined}
          />
        }
      >
        <Plot
          data={data}
          layout={mergedLayout}
          config={mergedConfig}
          style={{ width: '100%' }}
          useResizeHandler
          {...rest}
        />
      </Suspense>
    </div>
  )
}
