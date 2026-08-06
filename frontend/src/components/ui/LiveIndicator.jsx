import { Badge } from './Badge.jsx'
import { cn } from '../../utils.js'

export function LiveIndicator({ status, compact = false }) {
  const isConnected = status === 'connected'
  const isConnecting = status === 'connecting'

  return (
    <Badge
      variant={isConnected ? 'success' : isConnecting ? 'warning' : 'default'}
      className={cn('live-indicator', compact && 'live-indicator--compact')}
    >
      <span
        className={cn('live-indicator__dot', isConnected && 'is-connected')}
        aria-hidden="true"
      />
      <span>{isConnected ? 'Live' : isConnecting ? 'Connecting' : 'Offline'}</span>
    </Badge>
  )
}
