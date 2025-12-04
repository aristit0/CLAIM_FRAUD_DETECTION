import * as React from 'react'
import { Slot } from '@radix-ui/react-slot'
import { cva } from 'class-variance-authority'
import { cn } from '@/lib/utils'

const buttonVariants = cva(
  'inline-flex items-center justify-center rounded-xl text-sm font-semibold transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-purple-500 disabled:opacity-50',
  {
    variants: {
      variant: {
        default: 'bg-gradient-to-r from-purple-600 to-cyan-500 text-white hover:opacity-90 glow-purple',
        destructive: 'bg-gradient-to-r from-red-600 to-pink-600 text-white hover:opacity-90 glow-red',
        success: 'bg-gradient-to-r from-emerald-600 to-teal-500 text-white hover:opacity-90 glow-green',
        warning: 'bg-gradient-to-r from-amber-500 to-orange-500 text-white hover:opacity-90 glow-orange',
        outline: 'border border-white/20 bg-white/5 hover:bg-white/10',
        ghost: 'hover:bg-white/10',
      },
      size: {
        default: 'h-11 px-5',
        sm: 'h-9 px-4 text-xs',
        lg: 'h-14 px-8 text-base',
        icon: 'h-10 w-10',
      },
    },
    defaultVariants: { variant: 'default', size: 'default' },
  }
)

export const Button = React.forwardRef(({ className, variant, size, asChild = false, ...props }, ref) => {
  const Comp = asChild ? Slot : 'button'
  return <Comp className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />
})
Button.displayName = 'Button'
