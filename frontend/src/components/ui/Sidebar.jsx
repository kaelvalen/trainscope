import { Activity, Menu, X } from 'lucide-react'
import { Button } from './Button.jsx'
import { LiveIndicator } from './LiveIndicator.jsx'
import { cn } from '../../utils.js'
import { NAV_ITEMS } from '../../navigation.js'

export function Sidebar({ activeIndex, onChange, mobileOpen, onClose, liveStatus, runName }) {
  return (
    <>
      {mobileOpen && (
        <div
          className="sidebar-overlay fixed inset-0 z-40 lg:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <aside
        className={cn(
          'app-sidebar fixed inset-y-0 left-0 z-50 w-64 transform transition-transform lg:static lg:translate-x-0',
          mobileOpen ? 'translate-x-0' : '-translate-x-full'
        )}
      >
        <div className="sidebar-brand">
          <div className="brand-mark" aria-hidden="true">
            <Activity className="h-4 w-4" />
          </div>
          <div className="min-w-0">
            <div className="brand-name">TrainScope</div>
            <div className="brand-caption">Training observability</div>
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="ml-auto lg:hidden"
            onClick={onClose}
            aria-label="Close navigation"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        <div className="sidebar-section-label">Workspace</div>
        <nav className="sidebar-nav" aria-label="Main navigation">
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
                className={cn('sidebar-nav-item', isActive && 'is-active')}
                aria-current={isActive ? 'page' : undefined}
                title={`${item.label} (press ${item.shortcut})`}
              >
                <span className="sidebar-nav-icon" aria-hidden="true">
                  <Icon className="h-4 w-4" />
                </span>
                <span className="flex-1 text-left">{item.label}</span>
                <kbd className="sidebar-key">{item.shortcut}</kbd>
              </button>
            )
          })}
        </nav>

        <div className="sidebar-footer">
          <div className="sidebar-status">
            <div className="flex items-center justify-between gap-2">
              <span className="sidebar-status__label">Connection</span>
              <LiveIndicator status={liveStatus} compact />
            </div>
            <p className="sidebar-status__run" title={runName || undefined}>
              {runName || 'Waiting for run data'}
            </p>
          </div>
          <p className="sidebar-hint">
            Press <kbd>?</kbd> for shortcuts
          </p>
        </div>
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
