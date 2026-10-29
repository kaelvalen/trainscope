import { useEffect, useRef, useState } from 'react'
import KeyboardShortcutsHelp from '../KeyboardShortcutsHelp.jsx'
import { LiveIndicator } from '../ui/LiveIndicator.jsx'

const runIsRed = (run) =>
  (run.spike_count ?? 0) > 0 || (run.last_loss != null && !Number.isFinite(run.last_loss))

/** Logo, `trainscope / <run> ▾` breadcrumb with a run switcher, run count and live state. */
export function TopBar({ runName, runs = [], switchRun, liveStatus, onOpenRuns }) {
  const [open, setOpen] = useState(false)
  const menuRef = useRef(null)

  useEffect(() => {
    if (!open) return undefined
    const close = (event) => {
      if (event.type === 'keydown' && event.key !== 'Escape') return
      if (event.type === 'mousedown' && menuRef.current?.contains(event.target)) return
      setOpen(false)
    }
    document.addEventListener('mousedown', close)
    document.addEventListener('keydown', close)
    return () => {
      document.removeEventListener('mousedown', close)
      document.removeEventListener('keydown', close)
    }
  }, [open])

  const multi = runs.length > 1

  return (
    <header className="bg-surface">
      <div className="mx-auto flex h-14 max-w-[1360px] items-center gap-3 px-4 sm:px-8">
        <span aria-hidden="true" className="relative h-[22px] w-[22px] rounded-md bg-ink">
          <span className="absolute bottom-[6px] left-[5px] right-[5px] h-2 -skew-x-12 border-b-2 border-l-2 border-surface" />
        </span>
        <nav aria-label="Breadcrumb" className="flex min-w-0 items-center gap-2 text-sm">
          {multi ? (
            <a
              href="#runs"
              className="text-muted hover:text-ink"
              onClick={(event) => {
                event.preventDefault()
                onOpenRuns()
              }}
            >
              trainscope
            </a>
          ) : (
            <span className="text-muted">trainscope</span>
          )}
          <span aria-hidden="true" className="text-lg text-border-strong">
            /
          </span>
          <div className="relative" ref={menuRef}>
            <button
              type="button"
              disabled={!multi}
              aria-haspopup="listbox"
              aria-expanded={open}
              onClick={() => setOpen((o) => !o)}
              className="flex h-8 max-w-[40vw] items-center gap-1 truncate rounded-control px-1.5 font-medium text-ink enabled:hover:bg-hover"
            >
              <span className="truncate">{runName || 'Run'}</span>
              {multi && <span className="text-muted">▾</span>}
            </button>
            {open && (
              <ul
                role="listbox"
                aria-label="Runs"
                className="absolute left-0 top-10 z-50 min-w-[240px] rounded-panel border border-border bg-surface p-1 shadow-[0_8px_24px_rgba(0,0,0,0.08)]"
              >
                {runs.map((run) => (
                  <li key={run.name} role="option" aria-selected={run.name === runName}>
                    <button
                      type="button"
                      onClick={() => {
                        setOpen(false)
                        switchRun(run.name)
                      }}
                      className="flex h-9 w-full items-center gap-2 rounded-md px-2 text-left text-sm hover:bg-hover"
                    >
                      <span
                        aria-hidden="true"
                        className={`h-2 w-2 rounded-full ${runIsRed(run) ? 'bg-red' : 'bg-subtle'}`}
                      />
                      <span className="flex-1 truncate">{run.name}</span>
                      {run.name === runName && <span className="text-xs text-muted">current</span>}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
          {multi && (
            <span className="rounded-full border border-border px-2 py-0.5 text-xs text-muted">
              {runs.length} runs
            </span>
          )}
        </nav>
        <div className="ml-auto flex items-center gap-3">
          <LiveIndicator status={liveStatus} />
          <KeyboardShortcutsHelp />
        </div>
      </div>
    </header>
  )
}
