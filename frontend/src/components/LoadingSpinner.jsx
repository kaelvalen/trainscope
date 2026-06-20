import { COLORS } from '../theme.js'

export default function LoadingSpinner({ message = 'Loading…' }) {
  return (
    <div
      style={{
        color: COLORS.muted,
        padding: '40px',
        textAlign: 'center',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '12px',
      }}
      role="status"
      aria-live="polite"
    >
      <div
        className="ts-spinner"
        style={{
          width: '24px',
          height: '24px',
          border: `3px solid ${COLORS.border}`,
          borderTopColor: COLORS.accent,
          borderRadius: '50%',
        }}
      />
      <span style={{ fontSize: '14px' }}>{message}</span>
      <style>{`
        @keyframes ts-spin {
          to { transform: rotate(360deg); }
        }
        .ts-spinner {
          animation: ts-spin 0.8s linear infinite;
        }
      `}</style>
    </div>
  )
}
