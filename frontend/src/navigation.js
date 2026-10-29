export const NAV_ITEMS = [
  { id: 'overview', label: 'Overview', shortcut: '1' },
  { id: 'layers', label: 'Layers', shortcut: '2' },
  { id: 'spikes', label: 'Spikes', shortcut: '3' },
  { id: 'routing', label: 'Routing', shortcut: '4' },
  { id: 'compare', label: 'Compare', shortcut: '5' },
  { id: 'replay', label: 'Replay', shortcut: '6' },
]

// View ids from 1.9.0, kept so existing links still land on the same view.
export const HASH_ALIASES = { timeline: 'overview', diff: 'compare', moe: 'routing' }

/**
 * Map a location hash to a view. `#runs` is the runs index (not a tab); an
 * unknown or hidden tab falls back to the overview. A `?query` after the id
 * is preserved in the canonical hash but never part of the view name.
 */
export function resolveTab(hash, visibleIds) {
  const raw = (hash || '').replace(/^#/, '')
  const [id, query] = raw.split('?')
  const suffix = query ? `?${query}` : ''
  if (id === 'runs') return { view: 'runs', canonical: '#runs' }
  const view = HASH_ALIASES[id] ?? id
  if (visibleIds.includes(view)) return { view, canonical: `#${view}${suffix}` }
  return { view: 'overview', canonical: '#overview' }
}
