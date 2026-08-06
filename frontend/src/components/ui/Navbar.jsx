import { Activity, Menu, RefreshCw } from 'lucide-react'
import { LiveIndicator } from './LiveIndicator.jsx'
import { Badge } from './Badge.jsx'
import { Button } from './Button.jsx'
import KeyboardShortcutsHelp from '../KeyboardShortcutsHelp.jsx'

export function Navbar({
  runName,
  spikeCount,
  spikeEvents = [],
  liveStatus,
  onMenuClick,
  onRefresh,
  refreshing = false,
  sectionLabel,
}) {
  const eventCount = spikeEvents.length
  let badgeLabel = ''
  if (eventCount === 1) {
    const singleEv = spikeEvents[0]
    if (singleEv.earlyWarningWindow > 0) {
      badgeLabel = `1 event (${singleEv.earlyWarningWindow}-step early warning)`
    } else {
      badgeLabel = '1 spike event'
    }
  } else if (eventCount > 1) {
    badgeLabel = `${eventCount} spike events`
  } else if (spikeCount > 0) {
    badgeLabel = `${spikeCount} spike${spikeCount !== 1 ? 's' : ''}`
  }

  return (
    <header className="topbar">
      <Button
        variant="ghost"
        size="icon"
        className="topbar-menu lg:hidden"
        onClick={onMenuClick}
        aria-label="Open navigation"
      >
        <Menu className="h-4 w-4" />
      </Button>

      <div className="topbar-mobile-brand lg:hidden">
        <span className="brand-mark brand-mark--small" aria-hidden="true">
          <Activity className="h-3.5 w-3.5" />
        </span>
        <span className="brand-name">TrainScope</span>
      </div>

      <div className="topbar-context hidden min-w-0 lg:flex">
        <span className="topbar-context__eyebrow">Active run</span>
        <span className="topbar-context__run" title={runName || undefined}>
          {runName || 'Waiting for run data'}
        </span>
      </div>

      <div className="topbar-section hidden md:flex">
        <span className="topbar-section__dot" aria-hidden="true" />
        {sectionLabel}
      </div>

      <div className="topbar-actions">
        {badgeLabel && (
          <Badge variant="danger" className="hidden sm:inline-flex">
            {badgeLabel}
          </Badge>
        )}
        <LiveIndicator status={liveStatus} />
        <Button
          variant="ghost"
          size="icon"
          onClick={onRefresh}
          disabled={refreshing}
          title="Refresh run data (r)"
          aria-label="Refresh run data"
        >
          <RefreshCw className={refreshing ? 'h-4 w-4 animate-spin' : 'h-4 w-4'} />
        </Button>
        <KeyboardShortcutsHelp />
      </div>
    </header>
  )
}
