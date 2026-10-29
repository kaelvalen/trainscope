import { Suspense, lazy, useCallback, useEffect, useMemo, useState } from 'react'
import { useRun } from './RunContext.jsx'
import ErrorMessage from './components/ErrorMessage.jsx'
import ErrorBoundary from './components/ErrorBoundary.jsx'
import { Tabs } from './components/shell/Tabs.jsx'
import { TopBar } from './components/shell/TopBar.jsx'
import { Skeleton } from './components/ui/Skeleton.jsx'
import useKeyboardShortcuts from './hooks/useKeyboardShortcuts.js'
import { NAV_ITEMS, resolveTab } from './navigation.js'

const Runs = lazy(() => import('./views/Runs.jsx'))
const Overview = lazy(() => import('./views/Timeline.jsx'))
const LayerDrilldown = lazy(() => import('./views/LayerDrilldown.jsx'))
const ExpertUtilization = lazy(() => import('./views/ExpertUtilization.jsx'))
const DiffView = lazy(() => import('./views/DiffView.jsx'))
const SpikeInspector = lazy(() => import('./views/SpikeInspector.jsx'))
const Replay = lazy(() => import('./views/Replay.jsx'))

const VIEW_COMPONENTS = {
  runs: Runs,
  overview: Overview,
  layers: LayerDrilldown,
  spikes: SpikeInspector,
  routing: ExpertUtilization,
  compare: DiffView,
  replay: Replay,
}

function LoadingSkeleton() {
  return (
    <div className="space-y-4" aria-label="Loading run data">
      <Skeleton className="h-16" />
      <Skeleton className="h-80" />
      <Skeleton className="h-56" />
    </div>
  )
}

export default function App() {
  const { meta, moeData, runs, switchRun, loading, error, refresh, isReady, liveStatus } = useRun()

  const visible = useMemo(
    () =>
      NAV_ITEMS.filter(
        (item) =>
          (item.id !== 'routing' || (moeData?.length ?? 0) > 0) &&
          (item.id !== 'compare' || (runs?.length ?? 0) > 1)
      ),
    [moeData, runs]
  )
  const visibleIds = visible.map((item) => item.id)
  const [hash, setHash] = useState(() => window.location.hash)
  const { view, canonical } = resolveTab(hash, visibleIds)

  useEffect(() => {
    const onHash = () => setHash(window.location.hash)
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  // Tabs like Routing only become visible once run data arrives; don't rewrite deep links before then.
  useEffect(() => {
    if (isReady && window.location.hash !== canonical) {
      window.history.replaceState(null, '', canonical)
    }
  }, [canonical, isReady])

  const go = useCallback((id) => {
    window.history.pushState(null, '', `#${id}`)
    setHash(`#${id}`)
  }, [])

  const shortcuts = Object.fromEntries(visible.map((item, i) => [String(i + 1), () => go(item.id)]))
  useKeyboardShortcuts({ ...shortcuts, r: () => refresh() }, [visibleIds.join(','), refresh, go])

  const ActiveComponent = VIEW_COMPONENTS[view]

  return (
    <div className="min-h-screen bg-page text-ink">
      <TopBar
        runName={meta?.trainscope_config?.run_name}
        runs={runs}
        switchRun={(name) => {
          switchRun(name)
          go('overview')
        }}
        liveStatus={liveStatus}
        onOpenRuns={() => go('runs')}
      />
      <Tabs items={visible} active={view} onSelect={go} />

      <main className="mx-auto max-w-[1360px] px-4 py-6 sm:px-8 sm:py-8" aria-busy={loading}>
        {loading && <LoadingSkeleton />}
        {!loading && error && <ErrorMessage message={error} onRetry={refresh} />}
        {!loading && !error && isReady && (
          <ErrorBoundary>
            <Suspense fallback={<LoadingSkeleton />}>
              <ActiveComponent />
            </Suspense>
          </ErrorBoundary>
        )}
      </main>
    </div>
  )
}
