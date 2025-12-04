import { clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export const cn = (...inputs) => twMerge(clsx(inputs))

export const formatRupiah = (n) => {
  if (n == null) return '-'
  return new Intl.NumberFormat('id-ID', { style: 'currency', currency: 'IDR', minimumFractionDigits: 0 }).format(n)
}

export const formatDate = (d) => d ? new Date(d).toLocaleDateString('id-ID', { year: 'numeric', month: 'short', day: 'numeric' }) : '-'

export const getRiskClass = (level) => {
  const l = level?.toUpperCase() || ''
  if (l.includes('HIGH')) return 'risk-high glow-red'
  if (l.includes('MODERATE') || l.includes('MEDIUM')) return 'risk-moderate glow-orange'
  return 'risk-low glow-green'
}

export const getStatusBadge = (status) => {
  const s = status?.toLowerCase() || ''
  if (s === 'approved') return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
  if (s === 'declined') return 'bg-red-500/20 text-red-400 border-red-500/30'
  if (s === 'pending') return 'bg-amber-500/20 text-amber-400 border-amber-500/30'
  return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
}
