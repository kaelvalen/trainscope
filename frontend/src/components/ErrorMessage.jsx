import { COLORS } from '../theme.js'

export default function ErrorMessage({ message, onRetry }) {
  if (!message) return null
  return (
    <div
      style={{
        color: COLORS.danger,
        padding: '16px',
        background: `${COLORS.dangerBg}33`,
        border: `1px solid ${COLORS.dangerBg}`,
        borderRadius: '6px',
        marginBottom: '16px',
        fontSize: '13px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '12px',
        flexWrap: 'wrap',
      }}
      role="alert"
    >
      <span>{message}</span>
      {onRetry && (
        <button
          onClick={onRetry}
          style={{
            background: COLORS.dangerBg,
            color: COLORS.danger,
            border: 'none',
            borderRadius: '4px',
            padding: '4px 10px',
            cursor: 'pointer',
            fontSize: '12px',
            fontWeight: 600,
          }}
        >
          Retry
        </button>
      )}
    </div>
  )
}
