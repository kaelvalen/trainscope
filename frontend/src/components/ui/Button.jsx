import { forwardRef } from 'react'
import { cn } from '../../utils.js'

const variants = {
  primary: 'ui-button--primary focus-visible:ring-accent',
  ghost: 'ui-button--ghost focus-visible:ring-accent',
  danger: 'ui-button--danger focus-visible:ring-danger',
  muted: 'ui-button--muted focus-visible:ring-accent',
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
        'ui-button inline-flex items-center justify-center gap-2 rounded-lg font-semibold transition-colors',
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
