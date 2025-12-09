import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva } from "class-variance-authority"
import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-lg text-sm font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-gradient-to-r from-cyan-500 to-cyan-400 text-slate-900 hover:from-cyan-400 hover:to-cyan-300 shadow-lg shadow-cyan-500/25 btn-glow",
        destructive: "bg-gradient-to-r from-red-600 to-red-500 text-white hover:from-red-500 hover:to-red-400 shadow-lg shadow-red-500/25",
        outline: "border border-cyan-500/30 bg-transparent text-cyan-400 hover:bg-cyan-500/10 hover:border-cyan-400/50",
        secondary: "bg-slate-800 text-slate-100 hover:bg-slate-700 border border-slate-700",
        ghost: "text-slate-300 hover:bg-slate-800 hover:text-white",
        link: "text-cyan-400 underline-offset-4 hover:underline",
        success: "bg-gradient-to-r from-emerald-500 to-green-400 text-slate-900 hover:from-emerald-400 hover:to-green-300 shadow-lg shadow-emerald-500/25",
        warning: "bg-gradient-to-r from-amber-500 to-yellow-400 text-slate-900 hover:from-amber-400 hover:to-yellow-300 shadow-lg shadow-amber-500/25",
        neon: "bg-transparent border-2 border-cyan-400 text-cyan-400 hover:bg-cyan-400/10 hover:shadow-[0_0_20px_rgba(0,245,255,0.3)]",
        purple: "bg-gradient-to-r from-purple-600 to-fuchsia-500 text-white hover:from-purple-500 hover:to-fuchsia-400 shadow-lg shadow-purple-500/25",
      },
      size: {
        default: "h-10 px-5 py-2",
        sm: "h-8 rounded-md px-3 text-xs",
        lg: "h-12 rounded-lg px-8 text-base",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

const Button = React.forwardRef(({ className, variant, size, asChild = false, ...props }, ref) => {
  const Comp = asChild ? Slot : "button"
  return (
    <Comp
      className={cn(buttonVariants({ variant, size, className }))}
      ref={ref}
      {...props}
    />
  )
})
Button.displayName = "Button"

export { Button, buttonVariants }
