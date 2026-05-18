import { useState, useEffect } from 'react'
import { fetchMeta, fetchSpikes } from './api.js'
import Timeline from './views/Timeline.jsx'
import LayerDrilldown from './views/LayerDrilldown.jsx'
import DiffView from './views/DiffView.jsx'
import SpikeInspector from './views/SpikeInspector.jsx'

const TABS = ['Timeline', 'Layer Drill-down', 'Diff View', 'Spike Inspector']

const styles = {
  app: {
    minHeight: '100vh',
    background: '#0f1117',
    color: '#e2e8f0',
  },
  header: {
    background: '#1a1f2e',
    borderBottom: '1px solid #2d3748',
    padding: '16px 24px',
    display: 'flex',
    alignItems: 'center',
    gap: '24px',
  },
  title: {
    fontSize: '20px',
    fontWeight: 700,
    color: '#63b3ed',
    letterSpacing: '-0.5px',
  },
  meta: {
    fontSize: '13px',
    color: '#718096',
  },
  spikeTag: {
    background: '#742a2a',
    color: '#fc8181',
    borderRadius: '4px',
    padding: '2px 8px',
    fontSize: '12px',
    fontWeight: 600,
  },
  tabs: {
    display: 'flex',
    gap: '0',
    borderBottom: '1px solid #2d3748',
    background: '#1a1f2e',
    padding: '0 24px',
  },
  tab: {
    padding: '10px 20px',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: 500,
    borderBottom: '2px solid transparent',
    color: '#718096',
    background: 'none',
    border: 'none',
    borderBottom: '2px solid transparent',
    transition: 'color 0.15s',
  },
  tabActive: {
    color: '#63b3ed',
    borderBottom: '2px solid #63b3ed',
  },
  content: {
    padding: '24px',
  },
}

export default function App() {
  const [activeTab, setActiveTab] = useState(0)
  const [meta, setMeta] = useState(null)
  const [spikeCount, setSpikeCount] = useState(0)

  useEffect(() => {
    fetchMeta()
      .then(setMeta)
      .catch(() => {})
    fetchSpikes()
      .then(spikes => setSpikeCount(spikes.length))
      .catch(() => {})
  }, [])

  return (
    <div style={styles.app}>
      <div style={styles.header}>
        <span style={styles.title}>LossWatch</span>
        {meta && (
          <span style={styles.meta}>
            Run: <strong style={{ color: '#e2e8f0' }}>{meta.losswatch_config?.run_name || '—'}</strong>
          </span>
        )}
        {spikeCount > 0 && (
          <span style={styles.spikeTag}>{spikeCount} spike{spikeCount !== 1 ? 's' : ''}</span>
        )}
      </div>

      <div style={styles.tabs}>
        {TABS.map((tab, i) => (
          <button
            key={tab}
            style={{
              ...styles.tab,
              ...(activeTab === i ? styles.tabActive : {}),
            }}
            onClick={() => setActiveTab(i)}
          >
            {tab}
          </button>
        ))}
      </div>

      <div style={styles.content}>
        {activeTab === 0 && <Timeline />}
        {activeTab === 1 && <LayerDrilldown />}
        {activeTab === 2 && <DiffView />}
        {activeTab === 3 && <SpikeInspector />}
      </div>
    </div>
  )
}
