import { Activity, Menu } from 'lucide-react'
import { LiveIndicator } from './LiveIndicator.jsx'
import { Badge } from './Badge.jsx'
import { Button } from './Button.jsx'
import KeyboardShortcutsHelp from '../KeyboardShortcutsHelp.jsx'

export function Navbar({ runName, spikeCount, liveStatus, onMenuClick }) {
  return (
    <header className="flex h-14 items-center gap-4 border-b border-border bg-panel px-4 lg:px-6">
      <Button
        variant="ghost"
        size="icon"
        className="lg:hidden"
        onClick={onMenuClick}
        aria-label="Open navigation"
      >
        <Menu className="h-4 w-4" />
      </Button>

      <div className="flex items-center gap-2">
        <Activity className="h-5 w-5 text-accent" />
        <span className="text-lg font-bold tracking-tight text-foreground">TrainScope</span>
      </div>

      {runName && (
        <div className="hidden text-sm text-muted sm:block">
          Run: <span className="font-medium text-foreground">{runName}</span>
        </div>
      )}

      <div className="ml-auto flex items-center gap-3">
        {spikeCount > 0 && (
          <Badge variant="danger">
            {spikeCount} spike{spikeCount !== 1 ? 's' : ''}
          </Badge>
        )}
        <LiveIndicator status={liveStatus} />
        <KeyboardShortcutsHelp />
      </div>
    </header>
  )
}
