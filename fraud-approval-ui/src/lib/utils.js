import { clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs) {
  return twMerge(clsx(inputs))
}

export function formatRupiah(n) {
  if (n === null || n === undefined) return '-'
  return new Intl.NumberFormat('id-ID', {
    style: 'currency',
    currency: 'IDR',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(n)
}

export function formatDate(date) {
  if (!date) return '-'
  return new Date(date).toLocaleDateString('id-ID', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  })
}

export function getRiskColor(riskLevel) {
  switch (riskLevel?.toUpperCase()) {
    case 'HIGH RISK':
      return 'risk-badge-high'
    case 'MODERATE RISK':
      return 'risk-badge-moderate'
    case 'LOW RISK':
      return 'risk-badge-low'
    default:
      return 'bg-gray-600'
  }
}

export function getStatusColor(status) {
  switch (status?.toLowerCase()) {
    case 'approved':
      return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
    case 'declined':
      return 'bg-red-500/20 text-red-400 border-red-500/30'
    case 'manual_review':
      return 'bg-amber-500/20 text-amber-400 border-amber-500/30'
    case 'approved_partial':
      return 'bg-blue-500/20 text-blue-400 border-blue-500/30'
    case 'request_docs':
      return 'bg-purple-500/20 text-purple-400 border-purple-500/30'
    default:
      return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
  }
}
