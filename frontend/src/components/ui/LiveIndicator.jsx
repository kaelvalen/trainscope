import { cn } from '../../utils.js'

export function LiveIndicator({ status, compact = false }) {
  const isConnected = status === 'connected'
  const isConnecting = status === 'connecting'

  return (
    <span className={cn('inline-flex items-center gap-1.5 text-xs text-muted', compact && 'gap-1')}>
      <span
        aria-hidden="true"
        className={cn(
          'h-1.5 w-1.5 rounded-full',
          isConnected ? 'bg-blue' : isConnecting ? 'bg-amber' : 'bg-subtle'
        )}
      />
      <span>{isConnected ? 'Live' : isConnecting ? 'Connecting' : 'Offline'}</span>
    </span>
  )
}
