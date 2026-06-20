import { useState } from 'react'
import { COLORS } from '../theme.js'
import useKeyboardShortcuts from '../hooks/useKeyboardShortcuts.js'

const SHORTCUTS = [
  { keys: ['1', '2', '3', '4'], description: 'Switch tabs' },
  { keys: ['←', '→'], description: 'Previous / next tab' },
  { keys: ['?'], description: 'Toggle this help' },
]

export default function KeyboardShortcutsHelp() {
  const [open, setOpen] = useState(false)

  useKeyboardShortcuts({ '?': () => setOpen((o) => !o) }, [])

  return (
    <>
      <button
        onClick={() => setOpen((o) => !o)}
        className="ts-button ts-button-ghost"
        title="Keyboard shortcuts (?)"
        aria-label="Keyboard shortcuts"
      >
        ?
      </button>
      {open && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.6)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            padding: '16px',
          }}
          onClick={() => setOpen(false)}
          role="dialog"
          aria-modal="true"
          aria-labelledby="keyboard-help-title"
        >
          <div
            style={{
              background: COLORS.panel,
              border: `1px solid ${COLORS.border}`,
              borderRadius: '8px',
              padding: '24px',
              minWidth: '280px',
              maxWidth: '400px',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <h2
              id="keyboard-help-title"
              style={{ fontSize: '16px', marginBottom: '16px', color: COLORS.text }}
            >
              Keyboard shortcuts
            </h2>
            <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
              {SHORTCUTS.map(({ keys, description }) => (
                <li
                  key={description}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    gap: '16px',
                    padding: '8px 0',
                    borderBottom: `1px solid ${COLORS.border}`,
                    fontSize: '13px',
                    color: COLORS.muted,
                  }}
                >
                  <span>{description}</span>
                  <span style={{ display: 'flex', gap: '4px' }}>
                    {keys.map((k) => (
                      <kbd
                        key={k}
                        style={{
                          background: COLORS.bg,
                          border: `1px solid ${COLORS.border}`,
                          borderRadius: '4px',
                          padding: '2px 6px',
                          color: COLORS.text,
                          fontFamily: 'monospace',
                        }}
                      >
                        {k}
                      </kbd>
                    ))}
                  </span>
                </li>
              ))}
            </ul>
            <button
              onClick={() => setOpen(false)}
              className="ts-button"
              style={{ marginTop: '16px', width: '100%' }}
            >
              Close
            </button>
          </div>
        </div>
      )}
    </>
  )
}
