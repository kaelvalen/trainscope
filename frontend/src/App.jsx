import { useState } from 'react'
import { useRun } from './RunContext.jsx'
import Timeline from './views/Timeline.jsx'
import LayerDrilldown from './views/LayerDrilldown.jsx'
import DiffView from './views/DiffView.jsx'
import SpikeInspector from './views/SpikeInspector.jsx'
import LoadingSpinner from './components/LoadingSpinner.jsx'
import ErrorMessage from './components/ErrorMessage.jsx'
import ErrorBoundary from './components/ErrorBoundary.jsx'
import KeyboardShortcutsHelp from './components/KeyboardShortcutsHelp.jsx'
import useKeyboardShortcuts from './hooks/useKeyboardShortcuts.js'

const TABS = [
  { label: 'Timeline', shortcut: '1', component: Timeline },
  { label: 'Layer Drill-down', shortcut: '2', component: LayerDrilldown },
  { label: 'Diff View', shortcut: '3', component: DiffView },
  { label: 'Spike Inspector', shortcut: '4', component: SpikeInspector },
]

export default function App() {
  const [activeTab, setActiveTab] = useState(0)
  const { meta, spikes, loading, error, refresh, isReady } = useRun()

  useKeyboardShortcuts(
    {
      1: () => setActiveTab(0),
      2: () => setActiveTab(1),
      3: () => setActiveTab(2),
      4: () => setActiveTab(3),
      ArrowLeft: () => setActiveTab((i) => Math.max(0, i - 1)),
      ArrowRight: () => setActiveTab((i) => Math.min(TABS.length - 1, i + 1)),
    },
    []
  )

  const ActiveComponent = TABS[activeTab].component
  const spikeCount = spikes.length

  return (
    <div className="ts-app">
      <header className="ts-header">
        <span className="ts-title">TrainScope</span>
        {meta && (
          <span className="ts-meta">
            Run:{' '}
            <strong style={{ color: 'var(--text)' }}>
              {meta.trainscope_config?.run_name || '—'}
            </strong>
          </span>
        )}
        {spikeCount > 0 && (
          <span className="ts-spike-tag">
            {spikeCount} spike{spikeCount !== 1 ? 's' : ''}
          </span>
        )}
        <div className="ts-header-actions">
          <KeyboardShortcutsHelp />
        </div>
      </header>

      <nav className="ts-tabs" role="tablist" aria-label="View tabs">
        {TABS.map((tab, i) => (
          <button
            key={tab.label}
            role="tab"
            aria-selected={activeTab === i}
            className={`ts-tab ${activeTab === i ? 'ts-tab-active' : ''}`}
            onClick={() => setActiveTab(i)}
            title={`${tab.label} (press ${tab.shortcut})`}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <main className="ts-content">
        {loading && <LoadingSpinner message="Loading run data…" />}
        {!loading && error && <ErrorMessage message={error} onRetry={refresh} />}
        {!loading && !error && isReady && (
          <ErrorBoundary>
            <ActiveComponent />
          </ErrorBoundary>
        )}
      </main>
    </div>
  )
}
