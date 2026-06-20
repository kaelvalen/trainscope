import { useEffect, useRef } from 'react'

/**
 * Attach global keyboard shortcuts.
 *
 * @param {Object} handlers - map from key name to callback
 * @param {string[]} deps - effect dependencies
 *
 * Handlers are matched against `e.key`. Combinations can be expressed with a
 * dash, e.g. `"?"` or `"?"` (no modifier required for the help key).
 * Modifiers are ignored unless the key string starts with "ctrl+", "meta+",
 * "shift+", or "alt+".
 */
export default function useKeyboardShortcuts(handlers, deps = []) {
  const handlersRef = useRef(handlers)
  handlersRef.current = handlers

  useEffect(() => {
    function onKeyDown(e) {
      // Ignore shortcuts when typing in inputs/textareas/selects.
      const tag = e.target?.tagName?.toLowerCase()
      if (tag === 'input' || tag === 'textarea' || tag === 'select') return

      const entries = Object.entries(handlersRef.current)
      for (const [combo, handler] of entries) {
        const parts = combo.toLowerCase().split('+')
        const key = parts[parts.length - 1]
        const wantsCtrl = parts.includes('ctrl')
        const wantsMeta = parts.includes('meta')
        const wantsShift = parts.includes('shift')
        const wantsAlt = parts.includes('alt')

        const keyMatches = e.key.toLowerCase() === key || e.key === combo
        const modifiersMatch =
          e.ctrlKey === wantsCtrl &&
          e.metaKey === wantsMeta &&
          e.shiftKey === wantsShift &&
          e.altKey === wantsAlt

        if (keyMatches && modifiersMatch) {
          e.preventDefault()
          handler()
          return
        }
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
}
