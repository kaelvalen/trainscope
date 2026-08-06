import { useState } from 'react'
import { useRun } from './RunContext.jsx'
import Timeline from './views/Timeline.jsx'
import LayerDrilldown from './views/LayerDrilldown.jsx'
import DiffView from './views/DiffView.jsx'
import SpikeInspector from './views/SpikeInspector.jsx'
import ErrorMessage from './components/ErrorMessage.jsx'
import ErrorBoundary from './components/ErrorBoundary.jsx'
import { Navbar } from './components/ui/Navbar.jsx'
import { Sidebar } from './components/ui/Sidebar.jsx'
import { Skeleton } from './components/ui/Skeleton.jsx'
import { PageHeader } from './components/ui/PageHeader.jsx'
import useKeyboardShortcuts from './hooks/useKeyboardShortcuts.js'
import { NAV_ITEMS } from './navigation.js'

const VIEW_COMPONENTS = {
  timeline: Timeline,
  layers: LayerDrilldown,
  diff: DiffView,
  spikes: SpikeInspector,
}

function LoadingSkeleton() {
  return (
    <div className="space-y-6" aria-label="Loading run data">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {Array.from({ length: 4 }, (_, index) => (
          <Skeleton key={index} className="h-32" />
        ))}
      </div>
      <Skeleton className="h-80" />
      <Skeleton className="h-72" />
    </div>
  )
}

export default function App() {
  const [activeTab, setActiveTab] = useState(0)
  const [mobileOpen, setMobileOpen] = useState(false)
  const { meta, spikes, spikeEvents, loading, error, refresh, isReady, liveStatus } = useRun()
  const activeView = NAV_ITEMS[activeTab] || NAV_ITEMS[0]
  const ActiveComponent = VIEW_COMPONENTS[activeView.id]

  useKeyboardShortcuts(
    {
      1: () => setActiveTab(0),
      2: () => setActiveTab(1),
      3: () => setActiveTab(2),
      4: () => setActiveTab(3),
      ArrowLeft: () => setActiveTab((i) => Math.max(0, i - 1)),
      ArrowRight: () => setActiveTab((i) => Math.min(NAV_ITEMS.length - 1, i + 1)),
      r: () => refresh(),
    },
    [refresh]
  )

  return (
    <div className="app-shell text-foreground">
      <Sidebar
        activeIndex={activeTab}
        onChange={setActiveTab}
        mobileOpen={mobileOpen}
        onClose={() => setMobileOpen(false)}
        liveStatus={liveStatus}
        runName={meta?.trainscope_config?.run_name}
      />

      <div className="app-main">
        <Navbar
          runName={meta?.trainscope_config?.run_name}
          spikeCount={spikes.length}
          spikeEvents={spikeEvents}
          liveStatus={liveStatus}
          onMenuClick={() => setMobileOpen(true)}
          onRefresh={refresh}
          refreshing={loading}
          sectionLabel={activeView.label}
        />

        <main className="app-content" aria-busy={loading}>
          <div className="content-container">
            <PageHeader
              eyebrow={activeView.eyebrow}
              title={activeView.label}
              description={activeView.description}
              icon={activeView.icon}
            />
            {loading && <LoadingSkeleton />}
            {!loading && error && <ErrorMessage message={error} onRetry={refresh} />}
            {!loading && !error && isReady && (
              <ErrorBoundary>
                <ActiveComponent />
              </ErrorBoundary>
            )}
          </div>
        </main>
      </div>
    </div>
  )
}
