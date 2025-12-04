import * as React from 'react'
import { cva } from 'class-variance-authority'
import { cn } from '@/lib/utils'

const badgeVariants = cva(
  'inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
  {
    variants: {
      variant: {
        default: 'border-transparent bg-primary text-primary-foreground',
        secondary: 'border-transparent bg-secondary text-secondary-foreground',
        destructive: 'border-transparent bg-destructive text-destructive-foreground',
        outline: 'text-foreground',
        success: 'border-emerald-500/30 bg-emerald-500/20 text-emerald-400',
        warning: 'border-amber-500/30 bg-amber-500/20 text-amber-400',
        danger: 'border-red-500/30 bg-red-500/20 text-red-400',
        info: 'border-blue-500/30 bg-blue-500/20 text-blue-400',
        purple: 'border-purple-500/30 bg-purple-500/20 text-purple-400',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  }
)

function Badge({ className, variant, ...props }) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  )
}

export { Badge, badgeVariants }
