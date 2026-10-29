import { forwardRef } from 'react'
import { cn } from '../../utils.js'

const variants = {
  primary: 'border-ink bg-ink text-surface hover:opacity-90',
  secondary: 'border-border bg-surface text-ink hover:bg-hover',
  ghost: 'border-transparent bg-transparent text-muted hover:bg-hover hover:text-ink',
  danger: 'border-border bg-surface text-red hover:bg-hover',
}
// Older call sites still pass `muted`; it reads as secondary.
variants.muted = variants.secondary

const sizes = {
  sm: 'h-7 px-2.5 text-xs',
  md: 'h-8 px-3 text-sm',
  lg: 'h-9 px-4 text-sm',
  icon: 'h-8 w-8',
}

/** 28–36px tall; a pseudo-element keeps the hit area at 40px or more. */
export const Button = forwardRef(function Button(
  { variant = 'primary', size = 'md', className, children, ...props },
  ref
) {
  return (
    <button
      ref={ref}
      className={cn(
        'relative inline-flex items-center justify-center gap-2 rounded-control border font-medium',
        'transition-[background-color,color,opacity] duration-150',
        "before:absolute before:inset-x-0 before:-inset-y-1.5 before:content-['']",
        'disabled:cursor-not-allowed disabled:opacity-50',
        variants[variant] ?? variants.primary,
        sizes[size] ?? sizes.md,
        className
      )}
      {...props}
    >
      {children}
    </button>
  )
})
