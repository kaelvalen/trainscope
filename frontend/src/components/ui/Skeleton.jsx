import { cn } from '../../utils.js'

export function Skeleton({ className, style }) {
  return (
    <div
      className={cn('animate-pulse rounded-panel bg-hover motion-reduce:animate-none', className)}
      style={style}
    />
  )
}
