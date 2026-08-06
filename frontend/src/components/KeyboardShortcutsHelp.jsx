import { useState } from 'react'
import { Keyboard, X } from 'lucide-react'
import { Button } from './ui/Button.jsx'
import { Card, CardContent, CardHeader, CardTitle } from './ui/Card.jsx'
import useKeyboardShortcuts from '../hooks/useKeyboardShortcuts.js'

const SHORTCUTS = [
  { keys: ['1', '2', '3', '4'], description: 'Switch views' },
  { keys: ['←', '→'], description: 'Previous / next view' },
  { keys: ['?'], description: 'Toggle this help' },
  { keys: ['r'], description: 'Refresh run data' },
]

export default function KeyboardShortcutsHelp() {
  const [open, setOpen] = useState(false)

  useKeyboardShortcuts({ '?': () => setOpen((o) => !o) }, [])

  return (
    <>
      <Button
        variant="ghost"
        size="sm"
        onClick={() => setOpen((o) => !o)}
        title="Keyboard shortcuts (?)"
        aria-label="Keyboard shortcuts"
      >
        <Keyboard className="h-3.5 w-3.5" />
        <span className="hidden sm:inline">Shortcuts</span>
      </Button>

      {open && (
        <div
          className="fixed inset-0 z-[1000] flex items-center justify-center bg-black/60 p-4"
          onClick={() => setOpen(false)}
          role="dialog"
          aria-modal="true"
          aria-labelledby="keyboard-help-title"
        >
          <Card className="w-full max-w-sm" onClick={(e) => e.stopPropagation()}>
            <CardHeader>
              <CardTitle id="keyboard-help-title">Keyboard shortcuts</CardTitle>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setOpen(false)}
                aria-label="Close keyboard shortcuts"
              >
                <X className="h-4 w-4" />
              </Button>
            </CardHeader>
            <CardContent>
              <ul className="divide-y divide-border">
                {SHORTCUTS.map(({ keys, description }) => (
                  <li
                    key={description}
                    className="flex items-center justify-between py-2.5 text-sm"
                  >
                    <span className="text-muted">{description}</span>
                    <span className="flex items-center gap-1">
                      {keys.map((k) => (
                        <kbd
                          key={k}
                          className="rounded border border-border bg-background px-1.5 py-0.5 text-xs font-mono text-foreground"
                        >
                          {k}
                        </kbd>
                      ))}
                    </span>
                  </li>
                ))}
              </ul>
              <Button variant="primary" className="mt-4 w-full" onClick={() => setOpen(false)}>
                Close
              </Button>
            </CardContent>
          </Card>
        </div>
      )}
    </>
  )
}
