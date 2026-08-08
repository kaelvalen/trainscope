/**
 * Shared theme constants for TrainScope.
 *
 * Keeping colors, spacing, and Plotly layout defaults in one place makes the
 * views easier to maintain and keeps the dark UI consistent.
 */

export const COLORS = {
  bg: '#080A0C',
  panel: '#0F1115',
  border: '#23272F',
  text: '#e8edf5',
  muted: '#7E8795',
  accent: '#4BDDC0',
  success: '#47C28B',
  warning: '#F7AC3B',
  danger: '#F4525D',
  dangerBg: '#471518',
  purple: '#A07CDE',
  button: '#1D7C69',
}

/**
 * Semantic chart color palette. Views should import these instead of
 * hard-coding hex values so the theme stays consistent.
 */
export const CHART_COLORS = {
  loss: COLORS.accent,
  gradNorm: COLORS.success,
  weight: COLORS.purple,
  kurtosis: COLORS.warning,
  spike: COLORS.danger,
  muted: COLORS.muted,
  diffTop: COLORS.danger,
  diffRest: COLORS.accent,
}

export const SPACING = {
  xs: '4px',
  sm: '8px',
  md: '16px',
  lg: '24px',
  xl: '40px',
}

export const DARK_LAYOUT = {
  paper_bgcolor: COLORS.panel,
  plot_bgcolor: COLORS.bg,
  font: { color: COLORS.text, size: 11, family: "'JetBrains Mono', ui-monospace, monospace" },
  margin: { l: 58, r: 24, t: 24, b: 42 },
  colorway: [COLORS.accent, COLORS.success, COLORS.purple, COLORS.warning, COLORS.danger],
  xaxis: { gridcolor: COLORS.border, zerolinecolor: COLORS.border },
  yaxis: { gridcolor: COLORS.border, zerolinecolor: COLORS.border },
  hovermode: 'x unified',
  dragmode: 'zoom',
  hoverlabel: {
    bgcolor: COLORS.panel,
    bordercolor: COLORS.border,
    font: { color: COLORS.text, family: "'JetBrains Mono', ui-monospace, monospace" },
  },
}

export const PLOT_CONFIG = {
  displayModeBar: true,
  scrollZoom: true,
  // Strip Plotly's noisy default toolbar down to zoom/pan/reset — a 2-axis
  // time series doesn't need lasso/select/spikelines/hover-toggle controls.
  modeBarButtonsToRemove: [
    'lasso2d',
    'select2d',
    'autoScale2d',
    'toggleSpikelines',
    'hoverClosestCartesian',
    'hoverCompareCartesian',
  ],
  displaylogo: false,
  responsive: true,
}

/**
 * Default CSS-in-JS helpers used by several views.
 * Prefer CSS classes from styles/global.css for layout concerns; these are
 * kept for one-off dynamic needs.
 */
export const STYLES = {
  panel: {
    background: COLORS.panel,
    border: `1px solid ${COLORS.border}`,
    borderRadius: '4px',
    padding: SPACING.md,
  },
  input: {
    background: COLORS.panel,
    border: `1px solid ${COLORS.border}`,
    color: COLORS.text,
    padding: '6px 12px',
    borderRadius: '4px',
    fontSize: '13px',
  },
  select: {
    background: COLORS.panel,
    border: `1px solid ${COLORS.border}`,
    color: COLORS.text,
    padding: '6px 12px',
    borderRadius: '4px',
    fontSize: '13px',
  },
  button: {
    background: COLORS.button,
    color: '#fff',
    border: 'none',
    borderRadius: '4px',
    padding: '7px 18px',
    fontSize: '13px',
    fontWeight: 600,
    cursor: 'pointer',
  },
  buttonDisabled: {
    opacity: 0.6,
    cursor: 'not-allowed',
  },
  label: {
    fontSize: '13px',
    color: COLORS.muted,
  },
  mutedText: {
    color: COLORS.muted,
    fontSize: '13px',
  },
  centered: {
    padding: SPACING.xl,
    textAlign: 'center',
    color: COLORS.muted,
  },
}

export function truncateLayerName(name, maxLen = 30) {
  if (!name || name.length <= maxLen) return name || ''
  return '…' + name.slice(-(maxLen - 1))
}

export function spikeShape(step, options = {}) {
  const { color = 'rgba(252, 129, 129, 0.6)', width = 1.5, dash = 'dot' } = options
  return {
    type: 'line',
    x0: step,
    x1: step,
    yref: 'paper',
    y0: 0,
    y1: 1,
    line: { color, width, dash },
  }
}

export function scrubLineShape(step, options = {}) {
  const { color = COLORS.accent, width = 1.5 } = options
  return {
    type: 'line',
    x0: step,
    x1: step,
    yref: 'paper',
    y0: 0,
    y1: 1,
    line: { color, width },
  }
}

/**
 * Plotly band shape covering the detector's warmup window (no anomaly scores
 * are produced until the detector has seen `min_observations` loss values).
 */
export function warmupBandShape(startStep, endStep) {
  return {
    type: 'rect',
    xref: 'x',
    x0: startStep,
    x1: endStep,
    yref: 'paper',
    y0: 0,
    y1: 1,
    fillcolor: 'rgba(126, 135, 149, 0.07)',
    line: { color: 'rgba(126, 135, 149, 0.35)', width: 1, dash: 'dot' },
  }
}

/** Plotly annotation labelling the detector warmup band. */
export function warmupAnnotation(startStep, endStep) {
  return {
    x: startStep + Math.max(1, (endStep - startStep) * 0.02),
    y: 0.98,
    xref: 'x',
    yref: 'paper',
    text: 'detector warming up — spikes not yet reported',
    showarrow: false,
    yanchor: 'top',
    font: { color: COLORS.muted, size: 10 },
    bgcolor: 'rgba(15, 17, 21, 0.85)',
    bordercolor: COLORS.border,
    borderwidth: 1,
    borderpad: 4,
  }
}
