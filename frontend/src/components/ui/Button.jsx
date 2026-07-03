import { forwardRef } from 'react'
import { cn } from '../../utils.js'

const variants = {
  primary: 'bg-button text-white hover:brightness-110 focus-visible:ring-accent',
  ghost: 'bg-transparent border border-border text-muted hover:border-accent hover:text-foreground',
  danger: 'bg-danger-bg text-danger hover:brightness-110 focus-visible:ring-danger',
  muted: 'bg-muted/10 text-muted hover:bg-muted/20 hover:text-foreground',
}

const sizes = {
  sm: 'px-2.5 py-1 text-xs',
  md: 'px-4 py-2 text-sm',
  lg: 'px-5 py-2.5 text-base',
  icon: 'p-2',
}

export const Button = forwardRef(function Button(
  { variant = 'primary', size = 'md', className, children, ...props },
  ref
) {
  return (
    <button
      ref={ref}
      className={cn(
        'inline-flex items-center justify-center gap-2 rounded-md font-semibold transition-colors',
        'focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-offset-background',
        'disabled:cursor-not-allowed disabled:opacity-50',
        variants[variant],
        sizes[size],
        className
      )}
      {...props}
    >
      {children}
    </button>
  )
})
