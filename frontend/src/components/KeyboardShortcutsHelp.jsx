import { useState } from 'react'
import { Button } from './ui/Button.jsx'
import useKeyboardShortcuts from '../hooks/useKeyboardShortcuts.js'

const SHORTCUTS = [
  { keys: ['1', '…', '6'], description: 'Switch views' },
  { keys: ['?'], description: 'Toggle this help' },
  { keys: ['r'], description: 'Refresh run data' },
]

export default function KeyboardShortcutsHelp() {
  const [open, setOpen] = useState(false)

  useKeyboardShortcuts({ '?': () => setOpen((o) => !o), Escape: () => setOpen(false) }, [])

  return (
    <>
      <Button
        variant="ghost"
        size="sm"
        onClick={() => setOpen((o) => !o)}
        title="Keyboard shortcuts (?)"
        aria-label="Keyboard shortcuts"
      >
        <kbd className="rounded border border-border px-1 text-[11px] leading-4">?</kbd>
        <span className="hidden sm:inline">Shortcuts</span>
      </Button>

      {open && (
        <div
          className="fixed inset-0 z-[1000] flex items-center justify-center bg-black/30 p-4"
          onClick={() => setOpen(false)}
          role="dialog"
          aria-modal="true"
          aria-labelledby="keyboard-help-title"
        >
          <div
            className="w-full max-w-sm rounded-panel border border-border bg-surface shadow-[0_16px_40px_rgba(0,0,0,0.12)]"
            onClick={(e) => e.stopPropagation()}
          >
            <h2
              id="keyboard-help-title"
              className="border-b border-border px-4 py-3 text-sm font-medium"
            >
              Keyboard shortcuts
            </h2>
            <ul className="divide-y divide-border px-4">
              {SHORTCUTS.map(({ keys, description }) => (
                <li key={description} className="flex items-center justify-between py-2.5 text-sm">
                  <span className="text-muted">{description}</span>
                  <span className="flex items-center gap-1">
                    {keys.map((k) => (
                      <kbd key={k} className="rounded border border-border px-1.5 py-0.5 text-xs">
                        {k}
                      </kbd>
                    ))}
                  </span>
                </li>
              ))}
            </ul>
            <div className="border-t border-border p-3">
              <Button variant="secondary" className="w-full" onClick={() => setOpen(false)}>
                Close
              </Button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
