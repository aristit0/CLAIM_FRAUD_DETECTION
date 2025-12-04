import { useState, useEffect } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ArrowLeft, AlertTriangle, CheckCircle2, XCircle, Eye, Pill, Stethoscope, Heart, User, Calendar, Phone, MapPin, CreditCard, Activity, Sparkles, ThumbsUp, ThumbsDown, Zap } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Progress, Badge, Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from '@/components/ui/components'
import { formatRupiah, formatDate, getRiskClass, cn } from '@/lib/utils'
import { API_BASE } from '@/config'

export default function Review() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [dialog, setDialog] = useState({ open: false, action: null })
  const [processing, setProcessing] = useState(false)

  useEffect(() => {
    if (!localStorage.getItem('auth')) { navigate('/'); return }
    fetchData()
  }, [id, navigate])

  const fetchData = async () => {
    try {
      const [claimRes, scoreRes] = await Promise.all([
        fetch(`${API_BASE}/claim/${id}`),
        fetch(`${API_BASE}/score/${id}`)
      ])
      const claim = await claimRes.json()
      const score = await scoreRes.json()
      setData({ ...claim, ...score })
    } catch (e) { console.error(e) }
    setLoading(false)
  }

  const handleAction = async (action) => {
    setProcessing(true)
    try {
      await fetch(`${API_BASE}/update_status/${id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: action })
      })
      navigate('/dashboard')
    } catch (e) { alert('Error') }
    setProcessing(false)
  }

  if (loading) return <div className="min-h-screen flex items-center justify-center"><div className="w-12 h-12 border-4 border-purple-500 border-t-transparent rounded-full animate-spin" /></div>
  if (!data) return <div className="min-h-screen flex items-center justify-center">Not found</div>

  const { header, model_output: m, diagnosis, procedures, drugs, vitamins, ai_explanation } = data

  return (
    <div className="min-h-screen pb-32">
      {/* Header */}
      <header className="glass sticky top-0 z-50 border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="icon" onClick={() => navigate('/dashboard')}><ArrowLeft className="w-5 h-5" /></Button>
            <div>
              <h1 className="text-xl font-bold font-display">Claim #{id}</h1>
              <p className="text-xs text-white/50">{header?.patient_name}</p>
            </div>
          </div>
          <div className={cn('px-4 py-2 rounded-xl font-bold text-sm text-white', getRiskClass(m?.risk_level))}>
            {m?.risk_level || 'UNKNOWN'}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 pt-6 space-y-6">
        {/* AI Analysis */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
          <Card className="glass-strong overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-purple-600/10 via-transparent to-cyan-500/10" />
            <CardContent className="p-6 relative space-y-6">
              {/* Score Cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                  { label: 'Fraud Score', value: `${((m?.fraud_score || 0) * 100).toFixed(0)}%`, color: (m?.fraud_score || 0) >= 0.7 ? 'text-red-400' : (m?.fraud_score || 0) >= 0.4 ? 'text-amber-400' : 'text-emerald-400' },
                  { label: 'Confidence', value: `${((m?.confidence || 0) * 100).toFixed(0)}%`, color: 'text-purple-400' },
                  { label: 'Mismatches', value: m?.features?.mismatch_count || 0, color: (m?.features?.mismatch_count || 0) > 0 ? 'text-red-400' : 'text-emerald-400' },
                  { label: 'Cost Anomaly', value: `${m?.features?.cost_anomaly || 0}/4`, color: (m?.features?.cost_anomaly || 0) >= 3 ? 'text-red-400' : 'text-amber-400' },
                ].map(s => (
                  <div key={s.label} className="stat-card glass p-4 rounded-xl">
                    <p className="text-xs text-white/50">{s.label}</p>
                    <p className={cn('text-3xl font-bold font-display', s.color)}>{s.value}</p>
                  </div>
                ))}
              </div>

              {/* Clinical Compatibility */}
              <div className="glass rounded-xl p-4">
                <h3 className="text-sm font-semibold mb-3 flex items-center gap-2"><Activity className="w-4 h-4 text-cyan-400" /> Clinical Compatibility</h3>
                <div className="grid grid-cols-3 gap-3">
                  {[
                    { key: 'procedure_compatible', label: 'Procedure', icon: Stethoscope },
                    { key: 'drug_compatible', label: 'Drug', icon: Pill },
                    { key: 'vitamin_compatible', label: 'Vitamin', icon: Heart },
                  ].map(({ key, label, icon: Icon }) => {
                    const ok = m?.clinical_compatibility?.[key]
                    return (
                      <div key={key} className={cn('flex items-center gap-2 px-3 py-2 rounded-lg', ok ? 'bg-emerald-500/10' : 'bg-red-500/10')}>
                        <Icon className={cn('w-4 h-4', ok ? 'text-emerald-400' : 'text-red-400')} />
                        <span className="text-sm">{label}</span>
                        {ok ? <CheckCircle2 className="w-4 h-4 text-emerald-400 ml-auto" /> : <XCircle className="w-4 h-4 text-red-400 ml-auto" />}
                      </div>
                    )
                  })}
                </div>
              </div>

              {/* AI Explanation */}
              <div className="bg-gradient-to-r from-purple-500/10 to-cyan-500/10 rounded-xl p-4 border border-purple-500/20">
                <h3 className="text-sm font-semibold mb-2 flex items-center gap-2"><Sparkles className="w-4 h-4 text-purple-400" /> AI Analysis</h3>
                <p className="text-sm text-white/70 whitespace-pre-line">{ai_explanation || m?.explanation}</p>
              </div>

              {/* Risk Factors */}
              {m?.top_risk_factors?.length > 0 && (
                <div>
                  <h3 className="text-sm font-semibold mb-3 flex items-center gap-2"><AlertTriangle className="w-4 h-4 text-amber-400" /> Risk Factors</h3>
                  <div className="space-y-2">
                    {m.top_risk_factors.slice(0, 3).map((f, i) => (
                      <div key={i} className="glass rounded-lg p-3">
                        <div className="flex justify-between mb-1">
                          <span className="text-sm">{f.interpretation}</span>
                          <span className="text-xs text-white/50">{(f.importance * 100).toFixed(0)}%</span>
                        </div>
                        <Progress value={f.importance * 100} indicatorClass="bg-gradient-to-r from-red-500 to-amber-500" />
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>

        {/* Patient & Visit Info */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card>
            <CardHeader><CardTitle className="flex items-center gap-2"><User className="w-5 h-5 text-purple-400" /> Patient Info</CardTitle></CardHeader>
            <CardContent className="space-y-3 text-sm">
              {[
                ['Name', header?.patient_name],
                ['NIK', <span className="font-mono text-purple-400">{header?.patient_nik}</span>],
                ['Gender', header?.patient_gender === 'M' ? 'Male' : 'Female'],
                ['DOB', formatDate(header?.patient_dob)],
                ['Phone', header?.patient_phone],
              ].map(([l, v]) => <div key={l} className="flex justify-between border-b border-white/5 pb-2"><span className="text-white/50">{l}</span><span>{v}</span></div>)}
              <div><span className="text-white/50 flex items-center gap-1"><MapPin className="w-3 h-3" /> Address</span><p className="mt-1">{header?.patient_address}</p></div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader><CardTitle className="flex items-center gap-2"><Calendar className="w-5 h-5 text-cyan-400" /> Visit Info</CardTitle></CardHeader>
            <CardContent className="space-y-3 text-sm">
              {[
                ['Date', formatDate(header?.visit_date)],
                ['Type', <Badge variant="info">{header?.visit_type}</Badge>],
                ['Department', header?.department],
                ['Doctor', header?.doctor_name],
                ['Status', <Badge className={header?.status === 'approved' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-amber-500/20 text-amber-400'}>{header?.status}</Badge>],
              ].map(([l, v]) => <div key={l} className="flex justify-between items-center border-b border-white/5 pb-2"><span className="text-white/50">{l}</span><span>{v}</span></div>)}
              <div className="flex justify-between items-center pt-2">
                <span className="text-white/50 flex items-center gap-1"><CreditCard className="w-3 h-3" /> Total</span>
                <span className="text-xl font-bold text-emerald-400">{formatRupiah(header?.total_claim_amount)}</span>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Medical Details */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Diagnosis */}
          <Card>
            <CardHeader className="pb-3"><CardTitle className="text-base">Diagnosis</CardTitle></CardHeader>
            <CardContent className="space-y-2 max-h-64 overflow-y-auto">
              {diagnosis?.map((d, i) => (
                <div key={i} className="glass rounded-lg p-3">
                  <p className="font-mono text-sm text-purple-400">{d.icd10_code}</p>
                  <p className="text-xs text-white/60 mt-1">{d.icd10_description}</p>
                  {d.is_primary && <Badge variant="default" className="mt-2 text-[10px]">Primary</Badge>}
                </div>
              ))}
            </CardContent>
          </Card>
          {/* Procedures */}
          <Card>
            <CardHeader className="pb-3"><CardTitle className="text-base">Procedures</CardTitle></CardHeader>
            <CardContent className="space-y-2 max-h-64 overflow-y-auto">
              {procedures?.map((p, i) => (
                <div key={i} className="glass rounded-lg p-3">
                  <p className="font-mono text-sm text-cyan-400">{p.icd9_code}</p>
                  <p className="text-xs text-white/60 mt-1">{p.icd9_description}</p>
                  <p className="text-sm text-emerald-400 mt-2">{formatRupiah(p.cost)}</p>
                </div>
              ))}
              <div className="flex justify-between pt-2 border-t border-white/10">
                <span className="text-white/50">Total</span>
                <span className="font-bold text-emerald-400">{formatRupiah(header?.total_procedure_cost)}</span>
              </div>
            </CardContent>
          </Card>
          {/* Drugs */}
          <Card>
            <CardHeader className="pb-3"><CardTitle className="text-base">Drugs & Vitamins</CardTitle></CardHeader>
            <CardContent className="space-y-2 max-h-64 overflow-y-auto">
              {drugs?.map((d, i) => (
                <div key={i} className="glass rounded-lg p-3">
                  <p className="font-mono text-sm text-pink-400">{d.drug_code}</p>
                  <p className="text-xs text-white/60 mt-1">{d.drug_name}</p>
                  <p className="text-sm text-emerald-400 mt-2">{formatRupiah(d.cost)}</p>
                </div>
              ))}
              {vitamins?.map((v, i) => (
                <div key={i} className="glass rounded-lg p-3">
                  <p className="text-sm text-amber-400">{v.vitamin_name}</p>
                  <p className="text-sm text-emerald-400 mt-2">{formatRupiah(v.cost)}</p>
                </div>
              ))}
              <div className="flex justify-between pt-2 border-t border-white/10">
                <span className="text-white/50">Total</span>
                <span className="font-bold text-emerald-400">{formatRupiah((header?.total_drug_cost || 0) + (header?.total_vitamin_cost || 0))}</span>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>

      {/* Action Bar */}
      <div className="fixed bottom-0 left-0 right-0 glass-strong border-t border-white/10 p-4">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div><p className="font-semibold">Decision</p><p className="text-xs text-white/50">Review carefully before deciding</p></div>
          <div className="flex gap-3">
            <Button variant="success" size="lg" onClick={() => setDialog({ open: true, action: 'approved' })}><ThumbsUp className="w-4 h-4 mr-2" /> Approve</Button>
            <Button variant="destructive" size="lg" onClick={() => setDialog({ open: true, action: 'declined' })}><ThumbsDown className="w-4 h-4 mr-2" /> Decline</Button>
            <Button variant="warning" size="lg" onClick={() => setDialog({ open: true, action: 'manual_review' })}><Eye className="w-4 h-4 mr-2" /> Manual Review</Button>
          </div>
        </div>
      </div>

      {/* Confirm Dialog */}
      <Dialog open={dialog.open} onOpenChange={(o) => setDialog({ ...dialog, open: o })}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{dialog.action === 'approved' ? 'Approve' : dialog.action === 'declined' ? 'Decline' : 'Manual Review'} Claim?</DialogTitle>
            <DialogDescription>This action will update the claim status.</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setDialog({ open: false, action: null })}>Cancel</Button>
            <Button onClick={() => handleAction(dialog.action)} disabled={processing}>
              {processing ? <Zap className="w-4 h-4 animate-spin" /> : 'Confirm'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
