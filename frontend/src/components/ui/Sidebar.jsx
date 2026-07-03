import { LineChart, Layers, GitCompare, Zap, X, Menu } from 'lucide-react'
import { Button } from './Button.jsx'
import { cn } from '../../utils.js'

const NAV_ITEMS = [
  { label: 'Timeline', shortcut: '1', icon: LineChart },
  { label: 'Layer Drill-down', shortcut: '2', icon: Layers },
  { label: 'Diff View', shortcut: '3', icon: GitCompare },
  { label: 'Spike Inspector', shortcut: '4', icon: Zap },
]

export function Sidebar({ activeIndex, onChange, mobileOpen, onClose }) {
  return (
    <>
      {/* Mobile overlay */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/60 lg:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <aside
        className={cn(
          'fixed inset-y-0 left-0 z-50 w-64 transform border-r border-border bg-panel transition-transform lg:static lg:translate-x-0',
          mobileOpen ? 'translate-x-0' : '-translate-x-full'
        )}
      >
        <div className="flex h-14 items-center justify-between border-b border-border px-4 lg:hidden">
          <span className="font-semibold text-foreground">Navigation</span>
          <Button variant="ghost" size="icon" onClick={onClose} aria-label="Close navigation">
            <X className="h-4 w-4" />
          </Button>
        </div>

        <nav className="flex flex-col gap-1 p-3" aria-label="Main navigation">
          {NAV_ITEMS.map((item, index) => {
            const Icon = item.icon
            const isActive = activeIndex === index
            return (
              <button
                key={item.label}
                onClick={() => {
                  onChange(index)
                  onClose?.()
                }}
                className={cn(
                  'flex w-full items-center gap-3 rounded-md px-3 py-2.5 text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-accent/10 text-accent'
                    : 'text-muted hover:bg-muted/10 hover:text-foreground'
                )}
                aria-current={isActive ? 'page' : undefined}
                title={`${item.label} (press ${item.shortcut})`}
              >
                <Icon className="h-4 w-4" />
                <span className="flex-1 text-left">{item.label}</span>
                <kbd className="rounded border border-border bg-background px-1.5 py-0.5 text-xs text-muted">
                  {item.shortcut}
                </kbd>
              </button>
            )
          })}
        </nav>
      </aside>
    </>
  )
}

export function MobileMenuButton({ onClick }) {
  return (
    <Button variant="ghost" size="icon" onClick={onClick} aria-label="Open navigation">
      <Menu className="h-4 w-4" />
    </Button>
  )
}
