import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Search, LogOut, Eye, AlertTriangle, CheckCircle2, Clock, FileText, TrendingUp, Zap, BarChart3, PieChart, Activity } from 'lucide-react'
import { PieChart as RePieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts'
import { Button } from '@/components/ui/button'
import { Input, Badge } from '@/components/ui/components'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { formatRupiah, formatDate, getStatusBadge } from '@/lib/utils'
import { API_BASE } from '@/config'

const COLORS = ['#a855f7', '#06b6d4', '#ec4899', '#22c55e', '#f97316']

export default function Dashboard() {
  const navigate = useNavigate()
  const [claims, setClaims] = useState([])
  const [stats, setStats] = useState({})
  const [search, setSearch] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!localStorage.getItem('auth')) { navigate('/'); return }
    fetchData()
  }, [navigate])

  const fetchData = async () => {
    try {
      const [claimsRes, statsRes] = await Promise.all([
        fetch(`${API_BASE}/claims`),
        fetch(`${API_BASE}/stats`)
      ])
      const claimsData = await claimsRes.json()
      const statsData = await statsRes.json()
      setClaims(claimsData.claims || [])
      setStats(statsData)
    } catch (e) { console.error(e) }
    setLoading(false)
  }

  const filtered = claims.filter(c => 
    c.patient_name?.toLowerCase().includes(search.toLowerCase()) || 
    c.claim_id?.toString().includes(search)
  )

  const pieData = [
    { name: 'Pending', value: stats.pending || 0 },
    { name: 'Approved', value: stats.approved || 0 },
    { name: 'Declined', value: stats.declined || 0 },
  ]

  const barData = stats.by_department || []
  const trendData = stats.daily_trend || []

  return (
    <div className="min-h-screen pb-10">
      {/* Header */}
      <header className="glass sticky top-0 z-50 border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-600 to-cyan-500 flex items-center justify-center">
              <Zap className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold font-display gradient-text">Fraud Approval</h1>
              <p className="text-xs text-white/50">Dashboard</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Badge>{localStorage.getItem('user')}</Badge>
            <Button variant="ghost" size="icon" onClick={() => { localStorage.clear(); navigate('/') }}>
              <LogOut className="w-5 h-5" />
            </Button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 pt-6 space-y-6">
        {/* Stats Cards */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {[
            { label: 'Pending', value: stats.pending || 0, icon: Clock, gradient: 'from-amber-500 to-orange-500', glow: 'glow-orange' },
            { label: 'High Risk', value: stats.high_risk || 0, icon: AlertTriangle, gradient: 'from-red-500 to-pink-500', glow: 'glow-red' },
            { label: 'Approved', value: stats.approved || 0, icon: CheckCircle2, gradient: 'from-emerald-500 to-teal-500', glow: 'glow-green' },
            { label: 'Total', value: stats.total || 0, icon: FileText, gradient: 'from-purple-500 to-cyan-500', glow: 'glow-purple' },
          ].map((s, i) => (
            <motion.div key={s.label} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.1 }}>
              <Card className="stat-card hover-lift" style={{ '--card-accent': `linear-gradient(90deg, var(--tw-gradient-stops))` }}>
                <CardContent className="p-5">
                  <div className="flex items-start justify-between">
                    <div>
                      <p className="text-sm text-white/60">{s.label}</p>
                      <p className="text-3xl font-bold font-display mt-1">{s.value.toLocaleString()}</p>
                    </div>
                    <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${s.gradient} flex items-center justify-center ${s.glow}`}>
                      <s.icon className="w-6 h-6 text-white" />
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>

        {/* Charts Row */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Pie Chart */}
          <Card className="hover-lift">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-base">
                <PieChart className="w-5 h-5 text-purple-400" /> Status Distribution
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={200}>
                <RePieChart>
                  <Pie data={pieData} cx="50%" cy="50%" innerRadius={50} outerRadius={80} paddingAngle={5} dataKey="value">
                    {pieData.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
                  </Pie>
                  <Tooltip contentStyle={{ background: 'rgba(0,0,0,0.8)', border: 'none', borderRadius: 8 }} />
                </RePieChart>
              </ResponsiveContainer>
              <div className="flex justify-center gap-4 mt-2">
                {pieData.map((d, i) => (
                  <div key={d.name} className="flex items-center gap-1 text-xs">
                    <div className="w-3 h-3 rounded-full" style={{ background: COLORS[i] }} />
                    <span className="text-white/60">{d.name}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Bar Chart */}
          <Card className="hover-lift">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-base">
                <BarChart3 className="w-5 h-5 text-cyan-400" /> By Department
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={barData} layout="vertical">
                  <XAxis type="number" hide />
                  <YAxis type="category" dataKey="name" width={80} tick={{ fill: '#fff', fontSize: 10 }} />
                  <Tooltip contentStyle={{ background: 'rgba(0,0,0,0.8)', border: 'none', borderRadius: 8 }} />
                  <Bar dataKey="count" fill="url(#barGradient)" radius={[0, 4, 4, 0]} />
                  <defs>
                    <linearGradient id="barGradient" x1="0" y1="0" x2="1" y2="0">
                      <stop offset="0%" stopColor="#a855f7" />
                      <stop offset="100%" stopColor="#06b6d4" />
                    </linearGradient>
                  </defs>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Trend Chart */}
          <Card className="hover-lift">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-base">
                <Activity className="w-5 h-5 text-pink-400" /> Daily Trend
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={220}>
                <AreaChart data={trendData}>
                  <XAxis dataKey="date" tick={{ fill: '#fff', fontSize: 10 }} />
                  <YAxis hide />
                  <Tooltip contentStyle={{ background: 'rgba(0,0,0,0.8)', border: 'none', borderRadius: 8 }} />
                  <defs>
                    <linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#ec4899" stopOpacity={0.5} />
                      <stop offset="100%" stopColor="#ec4899" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <Area type="monotone" dataKey="count" stroke="#ec4899" fill="url(#areaGradient)" strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>

        {/* Search */}
        <div className="relative">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-white/40" />
          <Input placeholder="Cari claim ID atau nama pasien..." value={search} onChange={(e) => setSearch(e.target.value)} className="pl-12" />
        </div>

        {/* Claims Table */}
        <Card>
          <CardHeader className="border-b border-white/10">
            <CardTitle className="flex items-center gap-2">
              <FileText className="w-5 h-5 text-purple-400" /> Pending Claims
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            {loading ? (
              <div className="p-12 text-center">
                <div className="w-8 h-8 mx-auto border-4 border-purple-500 border-t-transparent rounded-full animate-spin" />
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-white/10 bg-white/[0.02]">
                      {['ID', 'Pasien', 'Tanggal', 'Tipe', 'Departemen', 'Total', 'Status', 'Aksi'].map(h => (
                        <th key={h} className="text-left p-4 text-xs font-semibold text-white/50 uppercase">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {filtered.map((c, i) => (
                      <motion.tr key={c.claim_id} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: i * 0.03 }}
                        className="border-b border-white/5 hover:bg-white/5 cursor-pointer transition-colors"
                        onClick={() => navigate(`/review/${c.claim_id}`)}>
                        <td className="p-4 font-mono text-sm text-purple-400">#{c.claim_id}</td>
                        <td className="p-4 font-medium">{c.patient_name}</td>
                        <td className="p-4 text-white/60 text-sm">{formatDate(c.visit_date)}</td>
                        <td className="p-4"><Badge variant="info">{c.visit_type}</Badge></td>
                        <td className="p-4 text-sm">{c.department}</td>
                        <td className="p-4 font-semibold text-emerald-400">{formatRupiah(c.total_claim_amount)}</td>
                        <td className="p-4"><Badge className={getStatusBadge(c.status)}>{c.status}</Badge></td>
                        <td className="p-4">
                          <Button variant="ghost" size="sm" onClick={(e) => { e.stopPropagation(); navigate(`/review/${c.claim_id}`) }}>
                            <Eye className="w-4 h-4 mr-1" /> Review
                          </Button>
                        </td>
                      </motion.tr>
                    ))}
                  </tbody>
                </table>
                {filtered.length === 0 && <div className="p-12 text-center text-white/40">No claims found</div>}
              </div>
            )}
          </CardContent>
        </Card>
      </main>
    </div>
  )
}
