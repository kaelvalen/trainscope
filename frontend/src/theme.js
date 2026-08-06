/**
 * Shared theme constants for TrainScope.
 *
 * Keeping colors, spacing, and Plotly layout defaults in one place makes the
 * views easier to maintain and keeps the dark UI consistent.
 */

export const COLORS = {
  bg: '#0d111a',
  panel: '#131a26',
  border: '#293241',
  text: '#e8edf5',
  muted: '#8290a5',
  accent: '#56d5e8',
  success: '#63d8a2',
  warning: '#f6bb67',
  danger: '#fb7d87',
  dangerBg: '#4a232b',
  purple: '#b99af7',
  button: '#0f7180',
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
  font: { color: COLORS.text, size: 11, family: 'system-ui, sans-serif' },
  margin: { l: 58, r: 24, t: 24, b: 42 },
  colorway: [COLORS.accent, COLORS.success, COLORS.purple, COLORS.warning, COLORS.danger],
  xaxis: { gridcolor: COLORS.border, zerolinecolor: COLORS.border },
  yaxis: { gridcolor: COLORS.border, zerolinecolor: COLORS.border },
  hovermode: 'x unified',
  hoverlabel: { bgcolor: COLORS.panel, bordercolor: COLORS.border, font: { color: COLORS.text } },
}

export const PLOT_CONFIG = {
  displayModeBar: false,
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
    borderRadius: '12px',
    padding: SPACING.md,
  },
  input: {
    background: COLORS.panel,
    border: `1px solid ${COLORS.border}`,
    color: COLORS.text,
    padding: '6px 12px',
    borderRadius: '8px',
    fontSize: '13px',
  },
  select: {
    background: COLORS.panel,
    border: `1px solid ${COLORS.border}`,
    color: COLORS.text,
    padding: '6px 12px',
    borderRadius: '8px',
    fontSize: '13px',
  },
  button: {
    background: COLORS.button,
    color: '#fff',
    border: 'none',
    borderRadius: '8px',
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
