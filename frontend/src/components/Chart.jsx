import Plot from 'react-plotly.js'
import { DARK_LAYOUT, PLOT_CONFIG } from '../theme.js'

/**
 * Consistent Plotly wrapper for the TrainScope dark theme.
 *
 * Accepts the same `data`, `layout`, and `config` props as react-plotly.js but
 * merges in the shared dark layout defaults so individual views stay small.
 */
export default function Chart({ data, layout = {}, config = {}, style = {}, ...rest }) {
  const mergedLayout = {
    ...DARK_LAYOUT,
    ...layout,
    font: { ...DARK_LAYOUT.font, ...layout.font },
    margin: { ...DARK_LAYOUT.margin, ...layout.margin },
    xaxis: { ...DARK_LAYOUT.xaxis, ...layout.xaxis },
    yaxis: { ...DARK_LAYOUT.yaxis, ...layout.yaxis },
  }

  const mergedConfig = { ...PLOT_CONFIG, ...config }

  return (
    <Plot
      data={data}
      layout={mergedLayout}
      config={mergedConfig}
      style={{ width: '100%', ...style }}
      useResizeHandler
      {...rest}
    />
  )
}
