import { cn } from '../../utils.js'

export function Card({ className, children, ...props }) {
  return (
    <div
      className={cn('rounded-lg border border-border bg-panel p-4 shadow-sm', className)}
      {...props}
    >
      {children}
    </div>
  )
}

export function CardHeader({ className, children }) {
  return <div className={cn('mb-3 flex items-center justify-between', className)}>{children}</div>
}

export function CardTitle({ className, children }) {
  return <h3 className={cn('text-sm font-semibold text-foreground', className)}>{children}</h3>
}

export function CardContent({ className, children }) {
  return <div className={cn('text-sm text-muted', className)}>{children}</div>
}
