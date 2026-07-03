import { cn } from '../../utils.js'

const variants = {
  default: 'bg-muted/15 text-muted',
  success: 'bg-success/15 text-success',
  warning: 'bg-warning/15 text-warning',
  danger: 'bg-danger-bg/30 text-danger',
  accent: 'bg-accent/15 text-accent',
}

export function Badge({ variant = 'default', className, children }) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium',
        variants[variant],
        className
      )}
    >
      {children}
    </span>
  )
}
