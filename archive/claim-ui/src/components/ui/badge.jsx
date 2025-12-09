import * as React from "react"
import { cva } from "class-variance-authority"
import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex items-center rounded-full px-3 py-1 text-xs font-medium transition-colors",
  {
    variants: {
      variant: {
        default: "bg-slate-800 text-slate-300 border border-slate-700",
        pending: "bg-amber-500/20 text-amber-400 border border-amber-500/30",
        approved: "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30",
        rejected: "bg-red-500/20 text-red-400 border border-red-500/30",
        processing: "bg-cyan-500/20 text-cyan-400 border border-cyan-500/30",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

function Badge({ className, variant, ...props }) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  )
}

export { Badge, badgeVariants }
