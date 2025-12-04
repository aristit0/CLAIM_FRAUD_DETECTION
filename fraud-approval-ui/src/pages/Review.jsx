import { useState, useEffect } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  ArrowLeft,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Eye,
  FileSearch,
  Pill,
  Stethoscope,
  Heart,
  User,
  Calendar,
  Phone,
  MapPin,
  CreditCard,
  Activity,
  Sparkles,
  ThumbsUp,
  ThumbsDown,
  Clock,
  AlertCircle,
  Zap,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { Separator } from '@/components/ui/separator'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog'
import { formatRupiah, formatDate, getRiskColor, getStatusColor, cn } from '@/lib/utils'
import { API_BASE } from '@/config'

const compatibilityItems = [
  { key: 'procedure_compatible', label: 'Procedure', icon: Stethoscope },
  { key: 'drug_compatible', label: 'Drug', icon: Pill },
  { key: 'vitamin_compatible', label: 'Vitamin', icon: Heart },
]

export default function ReviewPage() {
  const { claimId } = useParams()
  const navigate = useNavigate()
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [actionDialog, setActionDialog] = useState({ open: false, action: null })
  const [processing, setProcessing] = useState(false)

  useEffect(() => {
    const isAuth = localStorage.getItem('isAuthenticated')
    if (!isAuth) {
      navigate('/')
      return
    }

    // Fetch claim detail from backend
    const fetchClaim = async () => {
      try {
        // Fetch score from scoring backend
        const scoreRes = await fetch(`${API_BASE}/api/score/${claimId}`)
        const scoreData = await scoreRes.json()
        
        // Fetch claim details
        const claimRes = await fetch(`${API_BASE}/api/claim/${claimId}`)
        const claimData = await claimRes.json()
        
        setData({
          claim_id: parseInt(claimId),
          header: claimData.header || {},
          model_output: scoreData.model_output || {},
          ai_explanation: scoreData.ai_explanation,
          diagnosis: claimData.diagnosis || [],
          procedures: claimData.procedures || [],
          drugs: claimData.drugs || [],
          vitamins: claimData.vitamins || [],
        })
      } catch (err) {
        console.error('Failed to fetch claim:', err)
      } finally {
        setLoading(false)
      }
    }
    
    fetchClaim()
  }, [claimId, navigate])

  const handleAction = async (action) => {
    setProcessing(true)
    try {
      await fetch(`${API_BASE}/api/update_status/${claimId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: action }),
      })
      setActionDialog({ open: false, action: null })
      navigate('/dashboard')
    } catch (err) {
      console.error('Failed to update status:', err)
      alert('Failed to update status')
    } finally {
      setProcessing(false)
    }
  }

  const actionConfig = {
    approved: {
      title: 'Approve Claim',
      description: 'Are you sure you want to approve this claim? This action will authorize payment.',
      icon: CheckCircle2,
      color: 'bg-emerald-600 hover:bg-emerald-700',
      iconColor: 'text-emerald-400',
    },
    declined: {
      title: 'Decline Claim',
      description: 'Are you sure you want to decline this claim? The claimant will be notified.',
      icon: XCircle,
      color: 'bg-red-600 hover:bg-red-700',
      iconColor: 'text-red-400',
    },
    manual_review: {
      title: 'Request Manual Review',
      description: 'This will escalate the claim for manual investigation by the review team.',
      icon: Eye,
      color: 'bg-blue-600 hover:bg-blue-700',
      iconColor: 'text-blue-400',
    },
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
          className="w-16 h-16 rounded-full border-4 border-violet-500 border-t-transparent"
        />
      </div>
    )
  }

  if (!data) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p>Claim not found</p>
      </div>
    )
  }

  const { header, model_output: m, diagnosis, procedures, drugs, vitamins } = data

  return (
    <div className="min-h-screen pb-32">
      {/* Header */}
      <header className="glass-card sticky top-0 z-50 border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button variant="ghost" size="icon" onClick={() => navigate('/dashboard')}>
              <ArrowLeft className="w-5 h-5" />
            </Button>
            <div>
              <h1 className="text-xl font-bold font-display">Claim #{claimId}</h1>
              <p className="text-xs text-muted-foreground">{header.patient_name}</p>
            </div>
          </div>
          <div className={cn('px-4 py-2 rounded-xl font-bold text-sm', getRiskColor(m.risk_level))}>
            {m.risk_level}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 pt-6 space-y-6">
        {/* AI Analysis Banner */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <Card className="glass-card-elevated overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-violet-600/10 via-transparent to-cyan-500/10" />
            <CardContent className="p-6 relative">
              {/* Score Cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                {/* Fraud Score */}
                <div className="stat-card" style={{ '--accent-color': m.fraud_score >= 0.7 ? '#ef4444' : m.fraud_score >= 0.4 ? '#f59e0b' : '#10b981' }}>
                  <p className="text-xs text-muted-foreground mb-1">Fraud Score</p>
                  <p className={cn(
                    'text-4xl font-bold font-display',
                    m.fraud_score >= 0.7 ? 'text-red-400' : m.fraud_score >= 0.4 ? 'text-amber-400' : 'text-emerald-400'
                  )}>
                    {(m.fraud_score * 100).toFixed(0)}%
                  </p>
                  <Progress
                    value={m.fraud_score * 100}
                    className="mt-2 h-2"
                    indicatorClassName={cn(
                      m.fraud_score >= 0.7 ? 'bg-red-500' : m.fraud_score >= 0.4 ? 'bg-amber-500' : 'bg-emerald-500'
                    )}
                  />
                </div>

                {/* Confidence */}
                <div className="stat-card" style={{ '--accent-color': '#8b5cf6' }}>
                  <p className="text-xs text-muted-foreground mb-1">Confidence</p>
                  <p className="text-4xl font-bold font-display text-violet-400">
                    {(m.confidence * 100).toFixed(0)}%
                  </p>
                  <Progress value={m.confidence * 100} className="mt-2 h-2" indicatorClassName="bg-violet-500" />
                </div>

                {/* Mismatches */}
                <div className="stat-card" style={{ '--accent-color': m.features.mismatch_count > 0 ? '#ef4444' : '#10b981' }}>
                  <p className="text-xs text-muted-foreground mb-1">Clinical Mismatches</p>
                  <p className={cn(
                    'text-4xl font-bold font-display',
                    m.features.mismatch_count > 0 ? 'text-red-400' : 'text-emerald-400'
                  )}>
                    {m.features.mismatch_count}
                  </p>
                  <p className="text-xs text-muted-foreground mt-2">Incompatibilities</p>
                </div>

                {/* Cost Anomaly */}
                <div className="stat-card" style={{ '--accent-color': m.features.cost_anomaly >= 3 ? '#ef4444' : '#f59e0b' }}>
                  <p className="text-xs text-muted-foreground mb-1">Cost Anomaly</p>
                  <p className={cn(
                    'text-4xl font-bold font-display',
                    m.features.cost_anomaly >= 3 ? 'text-red-400' : m.features.cost_anomaly >= 2 ? 'text-amber-400' : 'text-emerald-400'
                  )}>
                    {m.features.cost_anomaly}/4
                  </p>
                  <Progress value={m.features.cost_anomaly * 25} className="mt-2 h-2" indicatorClassName="bg-amber-500" />
                </div>
              </div>

              {/* Clinical Compatibility */}
              <div className="bg-white/5 rounded-xl p-4 mb-6">
                <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
                  <Activity className="w-4 h-4 text-cyan-400" />
                  Clinical Compatibility
                </h3>
                <div className="grid grid-cols-3 gap-3">
                  {compatibilityItems.map(({ key, label, icon: Icon }) => {
                    const isCompatible = m.clinical_compatibility[key]
                    return (
                      <div
                        key={key}
                        className={cn(
                          'flex items-center gap-2 px-3 py-2 rounded-lg transition-colors',
                          isCompatible ? 'bg-emerald-500/10' : 'bg-red-500/10'
                        )}
                      >
                        <Icon className={cn('w-4 h-4', isCompatible ? 'text-emerald-400' : 'text-red-400')} />
                        <span className="text-sm">{label}</span>
                        {isCompatible ? (
                          <CheckCircle2 className="w-4 h-4 text-emerald-400 ml-auto" />
                        ) : (
                          <XCircle className="w-4 h-4 text-red-400 ml-auto" />
                        )}
                      </div>
                    )
                  })}
                </div>
              </div>

              {/* AI Explanation */}
              <div className="bg-gradient-to-r from-violet-500/10 to-cyan-500/10 rounded-xl p-4 border border-violet-500/20">
                <h3 className="text-sm font-semibold mb-2 flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-violet-400" />
                  AI Analysis
                </h3>
                <p className="text-sm text-muted-foreground leading-relaxed">{m.explanation}</p>
              </div>

              {/* Risk Factors */}
              {m.top_risk_factors && m.top_risk_factors.length > 0 && (
                <div className="mt-6">
                  <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 text-amber-400" />
                    Top Risk Factors
                  </h3>
                  <div className="space-y-2">
                    {m.top_risk_factors.slice(0, 3).map((factor, idx) => (
                      <div key={idx} className="bg-white/5 rounded-lg p-3">
                        <div className="flex justify-between items-center mb-1">
                          <span className="text-sm font-medium">{factor.interpretation}</span>
                          <span className="text-xs text-muted-foreground">
                            Impact: {(factor.importance * 100).toFixed(0)}%
                          </span>
                        </div>
                        <Progress
                          value={factor.importance * 100}
                          className="h-1.5"
                          indicatorClassName="bg-gradient-to-r from-red-500 to-amber-500"
                        />
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
          <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }}>
            <Card className="glass-card h-full">
              <CardHeader>
                <CardTitle className="text-lg font-display flex items-center gap-2">
                  <User className="w-5 h-5 text-violet-400" />
                  Patient Information
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Name</span>
                  <span className="font-medium">{header.patient_name}</span>
                </div>
                <Separator className="opacity-50" />
                <div className="flex justify-between">
                  <span className="text-muted-foreground">NIK</span>
                  <span className="font-mono text-sm text-violet-400">{header.patient_nik}</span>
                </div>
                <Separator className="opacity-50" />
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Gender</span>
                  <span>{header.patient_gender}</span>
                </div>
                <Separator className="opacity-50" />
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Date of Birth</span>
                  <span>{formatDate(header.patient_dob)}</span>
                </div>
                <Separator className="opacity-50" />
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground flex items-center gap-1">
                    <Phone className="w-3 h-3" /> Phone
                  </span>
                  <span>{header.patient_phone}</span>
                </div>
                <Separator className="opacity-50" />
                <div>
                  <span className="text-muted-foreground flex items-center gap-1 mb-1">
                    <MapPin className="w-3 h-3" /> Address
                  </span>
                  <p className="text-sm">{header.patient_address}</p>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 }}>
            <Card className="glass-card h-full">
              <CardHeader>
                <CardTitle className="text-lg font-display flex items-center gap-2">
                  <Calendar className="w-5 h-5 text-cyan-400" />
                  Visit Information
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Visit Date</span>
                  <span className="font-medium">{formatDate(header.visit_date)}</span>
                </div>
                <Separator className="opacity-50" />
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Visit Type</span>
                  <Badge variant="info">{header.visit_type}</Badge>
                </div>
                <Separator className="opacity-50" />
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Department</span>
                  <span>{header.department}</span>
                </div>
                <Separator className="opacity-50" />
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Attending Doctor</span>
                  <span>{header.doctor_name}</span>
                </div>
                <Separator className="opacity-50" />
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Current Status</span>
                  <Badge className={getStatusColor(header.status)}>{header.status}</Badge>
                </div>
                <Separator className="opacity-50" />
                <div className="flex justify-between items-center pt-2">
                  <span className="text-muted-foreground flex items-center gap-1">
                    <CreditCard className="w-3 h-3" /> Total Claim
                  </span>
                  <span className="text-xl font-bold text-emerald-400">{formatRupiah(header.total_claim_amount)}</span>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </div>

        {/* Medical Details Tabs */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
          <Card className="glass-card">
            <CardContent className="p-6">
              <Tabs defaultValue="diagnosis" className="w-full">
                <TabsList className="w-full md:w-auto">
                  <TabsTrigger value="diagnosis">Diagnosis</TabsTrigger>
                  <TabsTrigger value="procedures">Procedures</TabsTrigger>
                  <TabsTrigger value="drugs">Drugs</TabsTrigger>
                  <TabsTrigger value="vitamins">Vitamins</TabsTrigger>
                </TabsList>

                <TabsContent value="diagnosis">
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b border-white/10">
                          <th className="text-left p-3 text-xs font-semibold text-muted-foreground uppercase">ICD-10</th>
                          <th className="text-left p-3 text-xs font-semibold text-muted-foreground uppercase">Description</th>
                          <th className="text-left p-3 text-xs font-semibold text-muted-foreground uppercase">Type</th>
                        </tr>
                      </thead>
                      <tbody>
                        {diagnosis.map((d, i) => (
                          <tr key={i} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                            <td className="p-3 font-mono text-violet-400">{d.icd10_code}</td>
                            <td className="p-3">{d.icd10_description}</td>
                            <td className="p-3">
                              <Badge variant={d.is_primary ? 'default' : 'secondary'}>
                                {d.is_primary ? 'Primary' : 'Secondary'}
                              </Badge>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </TabsContent>

                <TabsContent value="procedures">
                  <div className="grid gap-3">
                    {procedures.map((p, i) => (
                      <div key={i} className="bg-white/5 rounded-xl p-4 flex justify-between items-center">
                        <div>
                          <p className="font-mono text-sm text-blue-400">{p.icd9_code}</p>
                          <p className="text-sm text-muted-foreground mt-1">{p.icd9_description}</p>
                        </div>
                        <span className="font-semibold text-emerald-400">{formatRupiah(p.cost)}</span>
                      </div>
                    ))}
                    <div className="flex justify-between items-center pt-3 border-t border-white/10">
                      <span className="font-semibold">Total Procedure Cost</span>
                      <span className="text-lg font-bold text-emerald-400">{formatRupiah(header.total_procedure_cost)}</span>
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="drugs">
                  <div className="grid gap-3">
                    {drugs.map((d, i) => (
                      <div key={i} className="bg-white/5 rounded-xl p-4 flex justify-between items-center">
                        <div>
                          <p className="font-mono text-sm text-purple-400">{d.drug_code}</p>
                          <p className="text-sm text-muted-foreground mt-1">{d.drug_name}</p>
                        </div>
                        <span className="font-semibold text-emerald-400">{formatRupiah(d.cost)}</span>
                      </div>
                    ))}
                    <div className="flex justify-between items-center pt-3 border-t border-white/10">
                      <span className="font-semibold">Total Drug Cost</span>
                      <span className="text-lg font-bold text-emerald-400">{formatRupiah(header.total_drug_cost)}</span>
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="vitamins">
                  <div className="grid gap-3">
                    {vitamins.map((v, i) => (
                      <div key={i} className="bg-white/5 rounded-xl p-4 flex justify-between items-center">
                        <div>
                          <p className="text-sm text-amber-400">{v.vitamin_name}</p>
                        </div>
                        <span className="font-semibold text-emerald-400">{formatRupiah(v.cost)}</span>
                      </div>
                    ))}
                    <div className="flex justify-between items-center pt-3 border-t border-white/10">
                      <span className="font-semibold">Total Vitamin Cost</span>
                      <span className="text-lg font-bold text-emerald-400">{formatRupiah(header.total_vitamin_cost)}</span>
                    </div>
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </motion.div>
      </main>

      {/* Floating Action Bar */}
      <motion.div
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="fixed bottom-0 left-0 right-0 glass-card-elevated border-t border-white/10 p-4"
      >
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="text-center sm:text-left">
            <p className="text-sm font-semibold">Reviewer Decision</p>
            <p className="text-xs text-muted-foreground">Make your final decision for this claim</p>
          </div>
          <div className="flex gap-3">
            <Button
              variant="success"
              size="lg"
              className="gap-2"
              onClick={() => setActionDialog({ open: true, action: 'approved' })}
            >
              <ThumbsUp className="w-4 h-4" />
              Approve
            </Button>
            <Button
              variant="destructive"
              size="lg"
              className="gap-2"
              onClick={() => setActionDialog({ open: true, action: 'declined' })}
            >
              <ThumbsDown className="w-4 h-4" />
              Decline
            </Button>
            <Button
              variant="outline"
              size="lg"
              className="gap-2"
              onClick={() => setActionDialog({ open: true, action: 'manual_review' })}
            >
              <Eye className="w-4 h-4" />
              Manual Review
            </Button>
          </div>
        </div>
      </motion.div>

      {/* Action Confirmation Dialog */}
      <Dialog open={actionDialog.open} onOpenChange={(open) => setActionDialog({ ...actionDialog, open })}>
        <DialogContent>
          {actionDialog.action && (
            <>
              <DialogHeader>
                <div className="mx-auto mb-4">
                  <div className={cn(
                    'w-16 h-16 rounded-full flex items-center justify-center',
                    actionConfig[actionDialog.action].color
                  )}>
                    {(() => {
                      const Icon = actionConfig[actionDialog.action].icon
                      return <Icon className="w-8 h-8 text-white" />
                    })()}
                  </div>
                </div>
                <DialogTitle className="text-center">{actionConfig[actionDialog.action].title}</DialogTitle>
                <DialogDescription className="text-center">
                  {actionConfig[actionDialog.action].description}
                </DialogDescription>
              </DialogHeader>
              <DialogFooter className="flex gap-2 sm:gap-2">
                <Button variant="ghost" onClick={() => setActionDialog({ open: false, action: null })}>
                  Cancel
                </Button>
                <Button
                  className={actionConfig[actionDialog.action].color}
                  onClick={() => handleAction(actionDialog.action)}
                  disabled={processing}
                >
                  {processing ? (
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                    >
                      <Zap className="w-4 h-4" />
                    </motion.div>
                  ) : (
                    'Confirm'
                  )}
                </Button>
              </DialogFooter>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}
