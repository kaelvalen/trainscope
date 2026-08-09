import { cn } from '../../utils.js'

export function Skeleton({ className, style }) {
  return <div className={cn('skeleton rounded-lg bg-muted/20', className)} style={style} />
}
