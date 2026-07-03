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
import useKeyboardShortcuts from './hooks/useKeyboardShortcuts.js'

const TABS = [
  { label: 'Timeline', shortcut: '1', component: Timeline },
  { label: 'Layer Drill-down', shortcut: '2', component: LayerDrilldown },
  { label: 'Diff View', shortcut: '3', component: DiffView },
  { label: 'Spike Inspector', shortcut: '4', component: SpikeInspector },
]

function LoadingSkeleton() {
  return (
    <div className="space-y-4">
      <Skeleton className="h-8 w-48" />
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Skeleton className="h-24" />
        <Skeleton className="h-24" />
        <Skeleton className="h-24" />
        <Skeleton className="h-24" />
      </div>
      <Skeleton className="h-64" />
      <Skeleton className="h-64" />
    </div>
  )
}

export default function App() {
  const [activeTab, setActiveTab] = useState(0)
  const [mobileOpen, setMobileOpen] = useState(false)
  const { meta, spikes, loading, error, refresh, isReady, liveStatus } = useRun()

  useKeyboardShortcuts(
    {
      1: () => setActiveTab(0),
      2: () => setActiveTab(1),
      3: () => setActiveTab(2),
      4: () => setActiveTab(3),
      ArrowLeft: () => setActiveTab((i) => Math.max(0, i - 1)),
      ArrowRight: () => setActiveTab((i) => Math.min(TABS.length - 1, i + 1)),
      r: () => refresh(),
    },
    [refresh]
  )

  const ActiveComponent = TABS[activeTab].component

  return (
    <div className="flex h-screen bg-background text-foreground">
      <Sidebar
        activeIndex={activeTab}
        onChange={setActiveTab}
        mobileOpen={mobileOpen}
        onClose={() => setMobileOpen(false)}
      />

      <div className="flex min-w-0 flex-1 flex-col">
        <Navbar
          runName={meta?.trainscope_config?.run_name}
          spikeCount={spikes.length}
          liveStatus={liveStatus}
          onMenuClick={() => setMobileOpen(true)}
        />

        <main className="flex-1 overflow-auto p-4 lg:p-6">
          {loading && <LoadingSkeleton />}
          {!loading && error && <ErrorMessage message={error} onRetry={refresh} />}
          {!loading && !error && isReady && (
            <ErrorBoundary>
              <ActiveComponent />
            </ErrorBoundary>
          )}
        </main>
      </div>
    </div>
  )
}
