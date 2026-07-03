import { Radio } from 'lucide-react'
import { Badge } from './Badge.jsx'
import { cn } from '../../utils.js'

export function LiveIndicator({ status }) {
  const isConnected = status === 'connected'
  const isConnecting = status === 'connecting'

  return (
    <Badge
      variant={isConnected ? 'success' : isConnecting ? 'warning' : 'default'}
      className="gap-1.5"
    >
      <Radio
        className={cn('h-3 w-3', isConnected && 'animate-pulse fill-current')}
        aria-hidden="true"
      />
      <span>{isConnected ? 'Live' : isConnecting ? 'Connecting' : 'Offline'}</span>
    </Badge>
  )
}
