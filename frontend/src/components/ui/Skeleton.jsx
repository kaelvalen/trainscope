import { cn } from '../../utils.js'

export function Skeleton({ className }) {
  return <div className={cn('animate-pulse rounded-md bg-muted/20', className)} />
}
