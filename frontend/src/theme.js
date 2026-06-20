/**
 * Shared theme constants for TrainScope.
 *
 * Keeping colors, spacing, and Plotly layout defaults in one place makes the
 * views easier to maintain and keeps the dark UI consistent.
 */

export const COLORS = {
  bg: '#0f1117',
  panel: '#1a1f2e',
  border: '#2d3748',
  text: '#e2e8f0',
  muted: '#718096',
  accent: '#63b3ed',
  success: '#68d391',
  warning: '#f6ad55',
  danger: '#fc8181',
  dangerBg: '#742a2a',
  purple: '#b794f4',
  button: '#2b6cb0',
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
  font: { color: COLORS.text, size: 12 },
  margin: { l: 60, r: 20, t: 40, b: 40 },
  xaxis: { gridcolor: COLORS.border, zerolinecolor: COLORS.border },
  yaxis: { gridcolor: COLORS.border, zerolinecolor: COLORS.border },
  hovermode: 'closest',
}

export const PLOT_CONFIG = {
  displayModeBar: false,
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
    borderRadius: '6px',
    padding: SPACING.md,
  },
  input: {
    background: COLORS.panel,
    border: `1px solid ${COLORS.border}`,
    color: COLORS.text,
    padding: '6px 12px',
    borderRadius: '6px',
    fontSize: '13px',
  },
  select: {
    background: COLORS.panel,
    border: `1px solid ${COLORS.border}`,
    color: COLORS.text,
    padding: '6px 12px',
    borderRadius: '6px',
    fontSize: '13px',
  },
  button: {
    background: COLORS.button,
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
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
