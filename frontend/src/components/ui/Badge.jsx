import { cn } from '../../utils.js'

const variants = {
  default: 'status-badge--default',
  success: 'status-badge--success',
  warning: 'status-badge--warning',
  danger: 'status-badge--danger',
  accent: 'status-badge--accent',
}

export function Badge({ variant = 'default', className, children }) {
  return (
    <span
      className={cn(
        'status-badge inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-semibold',
        variants[variant],
        className
      )}
    >
      {children}
    </span>
  )
}
