import { cn } from '../../utils.js'

const variants = {
  default: 'text-muted',
  success: 'text-ink',
  warning: 'text-amber',
  danger: 'text-red',
  accent: 'text-blue',
}

export function Badge({ variant = 'default', className, children }) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border border-border px-2 py-0.5 text-xs',
        variants[variant] ?? variants.default,
        className
      )}
    >
      {children}
    </span>
  )
}
