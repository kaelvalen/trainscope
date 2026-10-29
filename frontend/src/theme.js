/**
 * Chart color bindings. Colors live as CSS variables in styles/tokens.css
 * (light and dark); views refer to them by meaning, never by hex.
 */

export const CHART_COLORS = {
  loss: 'var(--ink)',
  gradNorm: 'var(--blue)',
  weight: 'var(--muted)',
  kurtosis: 'var(--amber)',
  spike: 'var(--red)',
}

/**
 * Categorical series colors in fixed order (never cycled). Series past the
 * seventh share --subtle and rely on the legend; views with many entities
 * should fold the tail into "Other" instead.
 */
export const SERIES_COLORS = Array.from({ length: 7 }, (_, i) => `var(--series-${i + 1})`)
export const seriesColor = (i) => SERIES_COLORS[i] ?? 'var(--subtle)'

/** Resolve a CSS custom property to its value; canvas drawing needs real colors. */
export function readToken(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}

export function truncateLayerName(name, maxLen = 30) {
  if (!name || name.length <= maxLen) return name || ''
  return '…' + name.slice(-(maxLen - 1))
}
