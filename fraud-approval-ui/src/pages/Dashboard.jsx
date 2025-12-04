import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Search,
  LogOut,
  Eye,
  AlertTriangle,
  CheckCircle2,
  Clock,
  TrendingUp,
  Users,
  FileText,
  Zap,
  Filter,
  Loader2,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { formatRupiah, formatDate, getStatusColor } from '@/lib/utils'
import { API_BASE } from '@/config'

const stats = [
  { label: 'Pending Review', value: '127', icon: Clock, color: 'from-amber-500 to-orange-500', trend: '+12%' },
  { label: 'High Risk', value: '23', icon: AlertTriangle, color: 'from-red-500 to-pink-500', trend: '+5%' },
  { label: 'Approved Today', value: '45', icon: CheckCircle2, color: 'from-emerald-500 to-teal-500', trend: '+18%' },
  { label: 'Total Claims', value: '1,234', icon: FileText, color: 'from-violet-500 to-purple-500', trend: '+8%' },
]

export default function Dashboard() {
  const navigate = useNavigate()
  const [claims, setClaims] = useState([])
  const [searchQuery, setSearchQuery] = useState('')
  const [loading, setLoading] = useState(true)
  const [page, setPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)

  useEffect(() => {
    const isAuth = localStorage.getItem('isAuthenticated')
    if (!isAuth) {
      navigate('/')
      return
    }
    fetchClaims()
  }, [navigate, page])

  const fetchClaims = async () => {
    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/claims?page=${page}&limit=20`)
      const data = await res.json()
      setClaims(data.claims || [])
      setTotalPages(data.total_pages || 1)
    } catch (err) {
      console.error('Failed to fetch claims:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleLogout = () => {
    localStorage.removeItem('isAuthenticated')
    localStorage.removeItem('user')
    navigate('/')
  }

  const filteredClaims = claims.filter(
    (c) =>
      c.patient_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      c.claim_id.toString().includes(searchQuery)
  )

  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.1 },
    },
  }

  const item = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 },
  }

  return (
    <div className="min-h-screen pb-10">
      {/* Header */}
      <header className="glass-card sticky top-0 z-50 border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-600 to-cyan-500 flex items-center justify-center">
              <Zap className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold font-display gradient-text">Fraud Approval</h1>
              <p className="text-xs text-muted-foreground">AI-Powered Review System</p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <Badge variant="purple" className="hidden md:flex gap-2 py-1.5">
              <Users className="w-3 h-3" />
              {localStorage.getItem('user')}
            </Badge>
            <Button variant="ghost" size="icon" onClick={handleLogout}>
              <LogOut className="w-5 h-5" />
            </Button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 pt-8">
        {/* Stats Grid */}
        <motion.div
          variants={container}
          initial="hidden"
          animate="show"
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8"
        >
          {stats.map((stat, i) => (
            <motion.div key={stat.label} variants={item}>
              <Card className="glass-card overflow-hidden group hover:scale-[1.02] transition-transform">
                <CardContent className="p-5">
                  <div className="flex items-start justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground mb-1">{stat.label}</p>
                      <p className="text-3xl font-bold font-display">{stat.value}</p>
                    </div>
                    <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${stat.color} flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform`}>
                      <stat.icon className="w-6 h-6 text-white" />
                    </div>
                  </div>
                  <div className="flex items-center gap-1 mt-3 text-xs">
                    <TrendingUp className="w-3 h-3 text-emerald-400" />
                    <span className="text-emerald-400">{stat.trend}</span>
                    <span className="text-muted-foreground">vs last week</span>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </motion.div>

        {/* Search & Filter */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="flex flex-col sm:flex-row gap-4 mb-6"
        >
          <div className="relative flex-1">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
            <Input
              placeholder="Search by claim ID or patient name..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-12"
            />
          </div>
          <Button variant="outline" className="gap-2">
            <Filter className="w-4 h-4" />
            Filter
          </Button>
        </motion.div>

        {/* Claims Table */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <Card className="glass-card overflow-hidden">
            <CardHeader className="border-b border-white/10">
              <CardTitle className="text-lg font-display flex items-center gap-2">
                <FileText className="w-5 h-5 text-violet-400" />
                Pending Claims
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-white/10 bg-white/[0.02]">
                      <th className="text-left p-4 text-xs font-semibold text-muted-foreground uppercase tracking-wider">ID</th>
                      <th className="text-left p-4 text-xs font-semibold text-muted-foreground uppercase tracking-wider">Patient</th>
                      <th className="text-left p-4 text-xs font-semibold text-muted-foreground uppercase tracking-wider hidden md:table-cell">Date</th>
                      <th className="text-left p-4 text-xs font-semibold text-muted-foreground uppercase tracking-wider hidden lg:table-cell">Type</th>
                      <th className="text-left p-4 text-xs font-semibold text-muted-foreground uppercase tracking-wider hidden lg:table-cell">Department</th>
                      <th className="text-left p-4 text-xs font-semibold text-muted-foreground uppercase tracking-wider">Amount</th>
                      <th className="text-left p-4 text-xs font-semibold text-muted-foreground uppercase tracking-wider">Status</th>
                      <th className="text-left p-4 text-xs font-semibold text-muted-foreground uppercase tracking-wider">Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {loading ? (
                      <tr>
                        <td colSpan="8" className="p-12 text-center">
                          <Loader2 className="w-8 h-8 mx-auto animate-spin text-violet-400" />
                          <p className="text-muted-foreground mt-2">Loading claims...</p>
                        </td>
                      </tr>
                    ) : (
                    <AnimatePresence>
                      {filteredClaims.map((claim, index) => (
                        <motion.tr
                          key={claim.claim_id}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: 20 }}
                          transition={{ delay: index * 0.05 }}
                          className="border-b border-white/5 table-row-hover cursor-pointer"
                          onClick={() => navigate(`/review/${claim.claim_id}`)}
                        >
                          <td className="p-4">
                            <span className="font-mono text-sm text-violet-400">#{claim.claim_id}</span>
                          </td>
                          <td className="p-4">
                            <span className="font-medium">{claim.patient_name}</span>
                          </td>
                          <td className="p-4 hidden md:table-cell">
                            <span className="text-sm text-muted-foreground">{formatDate(claim.visit_date)}</span>
                          </td>
                          <td className="p-4 hidden lg:table-cell">
                            <Badge variant="secondary" className="font-normal">
                              {claim.visit_type}
                            </Badge>
                          </td>
                          <td className="p-4 hidden lg:table-cell">
                            <span className="text-sm">{claim.department}</span>
                          </td>
                          <td className="p-4">
                            <span className="font-semibold text-emerald-400">{formatRupiah(claim.total_claim_amount)}</span>
                          </td>
                          <td className="p-4">
                            <Badge className={getStatusColor(claim.status)}>
                              {claim.status}
                            </Badge>
                          </td>
                          <td className="p-4">
                            <Button
                              variant="ghost"
                              size="sm"
                              className="gap-2"
                              onClick={(e) => {
                                e.stopPropagation()
                                navigate(`/review/${claim.claim_id}`)
                              }}
                            >
                              <Eye className="w-4 h-4" />
                              Review
                            </Button>
                          </td>
                        </motion.tr>
                      ))}
                    </AnimatePresence>
                    )}
                  </tbody>
                </table>
              </div>

              {filteredClaims.length === 0 && (
                <div className="p-12 text-center">
                  <Search className="w-12 h-12 mx-auto text-muted-foreground mb-4" />
                  <p className="text-muted-foreground">No claims found</p>
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>
      </main>
    </div>
  )
}
