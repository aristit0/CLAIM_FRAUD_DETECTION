import * as React from 'react'
import * as ProgressPrimitive from '@radix-ui/react-progress'
import * as DialogPrimitive from '@radix-ui/react-dialog'
import { X } from 'lucide-react'
import { cn } from '@/lib/utils'

// Progress
export const Progress = React.forwardRef(({ className, value, indicatorClass, ...props }, ref) => (
  <ProgressPrimitive.Root ref={ref} className={cn('relative h-3 w-full overflow-hidden rounded-full bg-white/10', className)} {...props}>
    <ProgressPrimitive.Indicator className={cn('h-full transition-all duration-500 rounded-full', indicatorClass || 'bg-gradient-to-r from-purple-500 to-cyan-500')} style={{ width: `${value || 0}%` }} />
  </ProgressPrimitive.Root>
))
Progress.displayName = 'Progress'

// Badge
export const Badge = ({ className, variant = 'default', ...props }) => {
  const variants = {
    default: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
    success: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
    warning: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
    danger: 'bg-red-500/20 text-red-400 border-red-500/30',
    info: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30',
  }
  return <span className={cn('inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold border', variants[variant], className)} {...props} />
}

// Input
export const Input = React.forwardRef(({ className, ...props }, ref) => (
  <input ref={ref} className={cn('w-full h-12 px-4 rounded-xl bg-white/5 border border-white/10 text-white placeholder:text-white/40 focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all', className)} {...props} />
))
Input.displayName = 'Input'

// Dialog
export const Dialog = DialogPrimitive.Root
export const DialogTrigger = DialogPrimitive.Trigger
export const DialogPortal = DialogPrimitive.Portal
export const DialogOverlay = React.forwardRef(({ className, ...props }, ref) => (
  <DialogPrimitive.Overlay ref={ref} className={cn('fixed inset-0 z-50 bg-black/80 backdrop-blur-sm', className)} {...props} />
))
export const DialogContent = React.forwardRef(({ className, children, ...props }, ref) => (
  <DialogPortal>
    <DialogOverlay />
    <DialogPrimitive.Content ref={ref} className={cn('fixed left-1/2 top-1/2 z-50 w-full max-w-lg -translate-x-1/2 -translate-y-1/2 glass-strong rounded-2xl p-6', className)} {...props}>
      {children}
      <DialogPrimitive.Close className="absolute right-4 top-4 rounded-lg p-1 hover:bg-white/10">
        <X className="h-4 w-4" />
      </DialogPrimitive.Close>
    </DialogPrimitive.Content>
  </DialogPortal>
))
export const DialogHeader = ({ className, ...props }) => <div className={cn('flex flex-col space-y-2 text-center sm:text-left', className)} {...props} />
export const DialogTitle = React.forwardRef(({ className, ...props }, ref) => (
  <DialogPrimitive.Title ref={ref} className={cn('text-lg font-bold', className)} {...props} />
))
export const DialogDescription = React.forwardRef(({ className, ...props }, ref) => (
  <DialogPrimitive.Description ref={ref} className={cn('text-sm text-white/60', className)} {...props} />
))
export const DialogFooter = ({ className, ...props }) => <div className={cn('flex justify-end gap-3 mt-6', className)} {...props} />
