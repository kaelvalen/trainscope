import { cn } from '../../utils.js'

export function Card({ className, children, ...props }) {
  return (
    <div
      className={cn(
        'surface-card rounded-xl border border-border bg-panel p-5 shadow-sm',
        className
      )}
      {...props}
    >
      {children}
    </div>
  )
}

export function CardHeader({ className, children, ...props }) {
  return (
    <div className={cn('mb-4 flex items-center justify-between gap-3', className)} {...props}>
      {children}
    </div>
  )
}

export function CardTitle({ className, children, ...props }) {
  return (
    <h3 className={cn('text-sm font-semibold text-foreground', className)} {...props}>
      {children}
    </h3>
  )
}

export function CardContent({ className, children, ...props }) {
  return (
    <div className={cn('text-sm text-muted', className)} {...props}>
      {children}
    </div>
  )
}
