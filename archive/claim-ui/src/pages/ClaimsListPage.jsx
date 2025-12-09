import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  List, Search, ChevronLeft, ChevronRight, 
  RefreshCw, Eye, Clock, User, DollarSign,
  FileText, Filter
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { formatRupiah, formatDateTime } from '@/lib/utils'

// Mock data for claims
const generateMockClaims = () => {
  const names = ['Budi Santoso', 'Rina Melati', 'Maya Pratiwi', 'Joko Susanto', 'Siti Rahayu', 
                 'Ahmad Hidayat', 'Dewi Kusuma', 'Bambang Wijaya', 'Ratna Sari', 'Eko Prasetyo']
  const statuses = ['pending', 'approved', 'rejected', 'processing']
  
  return Array.from({ length: 47 }, (_, i) => ({
    claim_id: 1000 + i,
    patient_name: names[Math.floor(Math.random() * names.length)],
    patient_nik: `317409${String(Math.floor(Math.random() * 100000000)).padStart(8, '0')}`,
    total_claim_amount: Math.floor(Math.random() * 2000000) + 100000,
    status: statuses[Math.floor(Math.random() * statuses.length)],
    created_at: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString(),
  }))
}

const MOCK_CLAIMS = generateMockClaims()

export default function ClaimsListPage() {
  const [claims, setClaims] = useState([])
  const [page, setPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)
  const [loading, setLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const limit = 10

  useEffect(() => {
    loadClaims()
  }, [page])

  const loadClaims = () => {
    setLoading(true)
    // Simulate API call
    setTimeout(() => {
      const filtered = MOCK_CLAIMS.filter(c => 
        c.patient_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        c.patient_nik.includes(searchTerm) ||
        c.claim_id.toString().includes(searchTerm)
      )
      const start = (page - 1) * limit
      const end = start + limit
      setClaims(filtered.slice(start, end))
      setTotalPages(Math.ceil(filtered.length / limit))
      setLoading(false)
    }, 500)
  }

  const handleSearch = () => {
    setPage(1)
    loadClaims()
  }

  const getStatusVariant = (status) => {
    switch (status) {
      case 'approved': return 'approved'
      case 'rejected': return 'rejected'
      case 'processing': return 'processing'
      default: return 'pending'
    }
  }

  const stats = {
    total: MOCK_CLAIMS.length,
    pending: MOCK_CLAIMS.filter(c => c.status === 'pending').length,
    approved: MOCK_CLAIMS.filter(c => c.status === 'approved').length,
    totalAmount: MOCK_CLAIMS.reduce((sum, c) => sum + c.total_claim_amount, 0)
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Header */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-3xl font-bold text-white flex items-center gap-3">
          <List className="w-8 h-8 text-cyan-400" />
          Claims Dashboard
        </h1>
        <p className="text-slate-400 mt-1">View and manage submitted claims</p>
      </motion.div>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <Card className="border-slate-800 bg-slate-900/50">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-slate-500 uppercase tracking-wide">Total Claims</p>
                <p className="text-2xl font-bold text-white">{stats.total}</p>
              </div>
              <div className="w-10 h-10 rounded-lg bg-cyan-500/20 flex items-center justify-center">
                <FileText className="w-5 h-5 text-cyan-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-slate-800 bg-slate-900/50">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-slate-500 uppercase tracking-wide">Pending</p>
                <p className="text-2xl font-bold text-amber-400">{stats.pending}</p>
              </div>
              <div className="w-10 h-10 rounded-lg bg-amber-500/20 flex items-center justify-center">
                <Clock className="w-5 h-5 text-amber-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-slate-800 bg-slate-900/50">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-slate-500 uppercase tracking-wide">Approved</p>
                <p className="text-2xl font-bold text-emerald-400">{stats.approved}</p>
              </div>
              <div className="w-10 h-10 rounded-lg bg-emerald-500/20 flex items-center justify-center">
                <User className="w-5 h-5 text-emerald-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-slate-800 bg-slate-900/50">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-slate-500 uppercase tracking-wide">Total Value</p>
                <p className="text-lg font-bold text-white">{formatRupiah(stats.totalAmount)}</p>
              </div>
              <div className="w-10 h-10 rounded-lg bg-purple-500/20 flex items-center justify-center">
                <DollarSign className="w-5 h-5 text-purple-400" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Search & Filter */}
      <Card className="border-slate-800 bg-slate-900/50 mb-6">
        <CardContent className="p-4">
          <div className="flex gap-3">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
              <Input
                placeholder="Search by name, NIK, or claim ID..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                className="pl-10"
              />
            </div>
            <Button variant="outline" onClick={handleSearch} className="gap-2">
              <Filter className="w-4 h-4" />
              Filter
            </Button>
            <Button variant="ghost" onClick={loadClaims} className="gap-2">
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Claims Table */}
      <Card className="border-slate-800 bg-slate-900/50">
        <CardHeader className="border-b border-slate-800">
          <CardTitle className="text-sm font-medium text-slate-400">
            Showing {claims.length} of {MOCK_CLAIMS.length} claims
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {loading ? (
            <div className="p-8 flex items-center justify-center">
              <div className="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin" />
            </div>
          ) : claims.length === 0 ? (
            <div className="p-8 text-center text-slate-500">
              No claims found
            </div>
          ) : (
            <div className="divide-y divide-slate-800">
              {claims.map((claim, idx) => (
                <motion.div
                  key={claim.claim_id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.05 }}
                  className="p-4 hover:bg-slate-800/30 transition-colors cursor-pointer table-row-hover"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500/20 to-purple-500/20 flex items-center justify-center">
                        <span className="text-sm font-mono text-cyan-400">#{claim.claim_id}</span>
                      </div>
                      <div>
                        <p className="font-medium text-white">{claim.patient_name}</p>
                        <p className="text-xs text-slate-500 font-mono">{claim.patient_nik}</p>
                      </div>
                    </div>

                    <div className="flex items-center gap-6">
                      <div className="text-right">
                        <p className="font-semibold text-white">{formatRupiah(claim.total_claim_amount)}</p>
                        <p className="text-xs text-slate-500">{formatDateTime(claim.created_at)}</p>
                      </div>
                      <Badge variant={getStatusVariant(claim.status)}>
                        {claim.status}
                      </Badge>
                      <Button variant="ghost" size="icon" className="text-slate-400 hover:text-cyan-400">
                        <Eye className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Pagination */}
      <div className="flex items-center justify-between mt-6">
        <p className="text-sm text-slate-500">
          Page {page} of {totalPages}
        </p>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setPage(p => Math.max(1, p - 1))}
            disabled={page === 1}
            className="gap-1"
          >
            <ChevronLeft className="w-4 h-4" />
            Previous
          </Button>

          <div className="flex gap-1">
            {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
              let pageNum
              if (totalPages <= 5) {
                pageNum = i + 1
              } else if (page <= 3) {
                pageNum = i + 1
              } else if (page >= totalPages - 2) {
                pageNum = totalPages - 4 + i
              } else {
                pageNum = page - 2 + i
              }
              return (
                <Button
                  key={pageNum}
                  variant={page === pageNum ? "default" : "ghost"}
                  size="sm"
                  onClick={() => setPage(pageNum)}
                  className={`w-8 ${page === pageNum ? '' : 'text-slate-400'}`}
                >
                  {pageNum}
                </Button>
              )
            })}
          </div>

          <Button
            variant="outline"
            size="sm"
            onClick={() => setPage(p => Math.min(totalPages, p + 1))}
            disabled={page === totalPages}
            className="gap-1"
          >
            Next
            <ChevronRight className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </div>
  )
}
