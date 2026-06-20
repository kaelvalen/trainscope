import { COLORS } from '../theme.js'

/**
 * Consistent placeholder for empty datasets.
 */
export default function EmptyState({ children, icon = '—' }) {
  return (
    <div
      style={{
        padding: '40px',
        textAlign: 'center',
        color: COLORS.muted,
        background: COLORS.panel,
        border: `1px dashed ${COLORS.border}`,
        borderRadius: '6px',
      }}
      role="status"
    >
      <div style={{ fontSize: '24px', marginBottom: '8px' }}>{icon}</div>
      <div style={{ fontSize: '14px' }}>{children}</div>
    </div>
  )
}
