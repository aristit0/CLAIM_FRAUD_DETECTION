import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  User, Calendar, Stethoscope, 
  Pill, Plus, Trash2, Send, Sparkles, Zap,
  AlertTriangle, CheckCircle, FileText, Heart
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { toast } from '@/components/ui/use-toast'

// Master Data
const ICD10_DATA = [
  { code: "A09", description: "Infectious gastroenteritis and colitis" },
  { code: "E11", description: "Type 2 diabetes mellitus" },
  { code: "I10", description: "Essential hypertension" },
  { code: "J06", description: "Acute upper respiratory infections" },
  { code: "J18", description: "Pneumonia" },
  { code: "K29", description: "Gastritis and duodenitis" },
  { code: "K35", description: "Acute appendicitis" },
  { code: "M54", description: "Dorsalgia (back pain)" },
  { code: "N39", description: "Urinary tract infection" },
  { code: "R50", description: "Fever" },
]

const ICD9_DATA = [
  { code: "03.31", description: "Spinal tap" },
  { code: "45.13", description: "Esophagogastroduodenoscopy" },
  { code: "88.72", description: "Diagnostic ultrasound - heart" },
  { code: "89.02", description: "Interview, evaluation, consultation" },
  { code: "89.14", description: "Electroencephalogram" },
  { code: "89.52", description: "Electrocardiogram" },
  { code: "93.05", description: "Stretching of muscle" },
  { code: "93.39", description: "Other physical therapy" },
  { code: "96.04", description: "Insertion of endotracheal tube" },
  { code: "99.04", description: "Transfusion of packed cells" },
]

const DRUGS_DATA = [
  { code: "KFA001", name: "Paracetamol 500mg" },
  { code: "KFA002", name: "Amoxicillin 500mg" },
  { code: "KFA003", name: "Ibuprofen 400mg" },
  { code: "KFA004", name: "Omeprazole 20mg" },
  { code: "KFA007", name: "Metformin 500mg" },
  { code: "KFA009", name: "Amlodipine 5mg" },
  { code: "KFA023", name: "Ranitidine 150mg" },
  { code: "KFA026", name: "Captopril 25mg" },
  { code: "KFA031", name: "Cefixime 200mg" },
  { code: "KFA035", name: "Ciprofloxacin 500mg" },
]

const VITAMINS_DATA = [
  "Vitamin C 500 mg",
  "Vitamin B Complex",
  "Vitamin B1 100 mg",
  "Vitamin B12 1000 mcg",
  "Vitamin D3 1000 IU",
  "Multivitamin Adult",
  "Zinc 20 mg",
  "Calcium + D",
]

const DEPARTMENTS = ["Poli Umum", "Poli Anak", "Poli Saraf", "IGD", "Poli Gigi"]
const VISIT_TYPES = ["rawat jalan", "rawat inap", "igd"]

// Autofill Templates
const AUTOFILL_TEMPLATES = {
  normalISPA: {
    label: "Normal ISPA",
    description: "Valid respiratory infection claim",
    type: "normal",
    data: {
      patient_name: "Budi Santoso",
      patient_nik: "3174091501800004",
      patient_dob: "1985-06-15",
      patient_gender: "M",
      patient_phone: "081234567810",
      patient_address: "Jl. Merdeka No.100 Jakarta",
      visit_date: "2024-12-04",
      visit_type: "rawat jalan",
      doctor_name: "Dr. Surya",
      department: "Poli Umum",
      diagnosis_primary: "J06",
      diagnosis_secondary: "",
      procedures: [{ code: "89.02", cost: 75000 }],
      drugs: [{ code: "KFA001", cost: 25000 }],
      vitamins: [{ name: "Vitamin C 500 mg", cost: 15000 }],
    }
  },
  normalGastritis: {
    label: "Normal Gastritis",
    description: "Valid digestive disorder claim",
    type: "normal",
    data: {
      patient_name: "Rina Melati",
      patient_nik: "3174082207810005",
      patient_dob: "1981-07-22",
      patient_gender: "F",
      patient_phone: "081555666777",
      patient_address: "Jl. Melati No.15 Bandung",
      visit_date: "2025-01-18",
      visit_type: "rawat jalan",
      doctor_name: "Dr. Wijaya",
      department: "Poli Umum",
      diagnosis_primary: "K29",
      diagnosis_secondary: "",
      procedures: [{ code: "03.31", cost: 120000 }],
      drugs: [{ code: "KFA004", cost: 35000 }, { code: "KFA023", cost: 25000 }],
      vitamins: [{ name: "Multivitamin Adult", cost: 20000 }],
    }
  },
  fraudMismatch: {
    label: "Clinical Mismatch",
    description: "Suspicious: procedures don't match diagnosis",
    type: "fraud",
    data: {
      patient_name: "Maya Pratiwi",
      patient_nik: "3175024302759999",
      patient_dob: "1988-09-14",
      patient_gender: "F",
      patient_phone: "08129822222",
      patient_address: "Jl. Sudirman No.88 Bekasi",
      visit_date: "2025-02-08",
      visit_type: "rawat jalan",
      doctor_name: "Dr. Hartono",
      department: "Poli Umum",
      diagnosis_primary: "J06",
      diagnosis_secondary: "",
      procedures: [{ code: "45.13", cost: 750000 }, { code: "93.05", cost: 250000 }],
      drugs: [{ code: "KFA035", cost: 150000 }, { code: "KFA007", cost: 85000 }],
      vitamins: [{ name: "Vitamin B Complex", cost: 75000 }, { name: "Vitamin C 500 mg", cost: 50000 }, { name: "Zinc 20 mg", cost: 35000 }],
    }
  },
  fraudCostAnomaly: {
    label: "Cost Anomaly",
    description: "Suspicious: inflated costs for simple diagnosis",
    type: "fraud",
    data: {
      patient_name: "Joko Susanto",
      patient_nik: "3173092503760001",
      patient_dob: "1976-03-25",
      patient_gender: "M",
      patient_phone: "081377788899",
      patient_address: "Jl. Gatot Subroto No.45 Jakarta",
      visit_date: "2025-01-20",
      visit_type: "rawat jalan",
      doctor_name: "Dr. Indra",
      department: "Poli Umum",
      diagnosis_primary: "J06",
      diagnosis_secondary: "",
      procedures: [{ code: "89.02", cost: 450000 }, { code: "89.14", cost: 550000 }, { code: "96.04", cost: 650000 }],
      drugs: [{ code: "KFA001", cost: 125000 }, { code: "KFA009", cost: 95000 }, { code: "KFA026", cost: 85000 }, { code: "KFA031", cost: 75000 }],
      vitamins: [{ name: "Multivitamin Adult", cost: 150000 }, { name: "Vitamin C 500 mg", cost: 100000 }, { name: "Vitamin B1 100 mg", cost: 80000 }],
    }
  },
}

export default function SubmitClaimPage() {
  const [formData, setFormData] = useState({
    patient_name: '', patient_nik: '', patient_dob: '', patient_gender: 'M',
    patient_phone: '', patient_address: '', visit_date: '', visit_type: 'rawat jalan',
    doctor_name: '', department: 'Poli Umum', diagnosis_primary: '', diagnosis_secondary: '',
  })

  const [procedures, setProcedures] = useState([{ code: '', cost: '' }])
  const [drugs, setDrugs] = useState([{ code: '', cost: '' }])
  const [vitamins, setVitamins] = useState([{ name: '', cost: '' }])
  const [submitting, setSubmitting] = useState(false)
  const [showQuickFill, setShowQuickFill] = useState(false)

  const handleInputChange = (field, value) => setFormData(prev => ({ ...prev, [field]: value }))

  const addRow = (type) => {
    if (type === 'procedure') setProcedures([...procedures, { code: '', cost: '' }])
    else if (type === 'drug') setDrugs([...drugs, { code: '', cost: '' }])
    else if (type === 'vitamin') setVitamins([...vitamins, { name: '', cost: '' }])
  }

  const removeRow = (type, index) => {
    if (type === 'procedure') setProcedures(procedures.filter((_, i) => i !== index))
    else if (type === 'drug') setDrugs(drugs.filter((_, i) => i !== index))
    else if (type === 'vitamin') setVitamins(vitamins.filter((_, i) => i !== index))
  }

  const updateRow = (type, index, field, value) => {
    if (type === 'procedure') { const u = [...procedures]; u[index][field] = value; setProcedures(u) }
    else if (type === 'drug') { const u = [...drugs]; u[index][field] = value; setDrugs(u) }
    else if (type === 'vitamin') { const u = [...vitamins]; u[index][field] = value; setVitamins(u) }
  }

  const applyAutofill = (templateKey) => {
    const template = AUTOFILL_TEMPLATES[templateKey]
    if (!template) return
    setFormData({
      patient_name: template.data.patient_name, patient_nik: template.data.patient_nik,
      patient_dob: template.data.patient_dob, patient_gender: template.data.patient_gender,
      patient_phone: template.data.patient_phone, patient_address: template.data.patient_address,
      visit_date: template.data.visit_date, visit_type: template.data.visit_type,
      doctor_name: template.data.doctor_name, department: template.data.department,
      diagnosis_primary: template.data.diagnosis_primary, diagnosis_secondary: template.data.diagnosis_secondary || '',
    })
    setProcedures(template.data.procedures.map(p => ({ code: p.code, cost: p.cost.toString() })))
    setDrugs(template.data.drugs.map(d => ({ code: d.code, cost: d.cost.toString() })))
    setVitamins(template.data.vitamins.map(v => ({ name: v.name, cost: v.cost.toString() })))
    toast({ title: `Template Applied: ${template.label}`, description: template.description, variant: template.type === 'fraud' ? 'destructive' : 'success' })
    setShowQuickFill(false)
  }

  const calculateTotal = () => {
    const procTotal = procedures.reduce((sum, p) => sum + (parseFloat(p.cost) || 0), 0)
    const drugTotal = drugs.reduce((sum, d) => sum + (parseFloat(d.cost) || 0), 0)
    const vitTotal = vitamins.reduce((sum, v) => sum + (parseFloat(v.cost) || 0), 0)
    return procTotal + drugTotal + vitTotal
  }

  // ✅ ACTUAL API CALL
  const handleSubmit = async (e) => {
    e.preventDefault()
    setSubmitting(true)

    try {
      const payload = {
        ...formData,
        procedures: procedures.filter(p => p.code).map(p => ({ code: p.code, cost: parseFloat(p.cost) || 0 })),
        drugs: drugs.filter(d => d.code).map(d => ({ code: d.code, cost: parseFloat(d.cost) || 0 })),
        vitamins: vitamins.filter(v => v.name).map(v => ({ name: v.name, cost: parseFloat(v.cost) || 0 })),
      }

      console.log('Submitting:', payload)

      const response = await fetch('/api/claims', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify(payload),
      })

      const data = await response.json()
      console.log('Response:', data)

      if (response.ok && data.success) {
        toast({
          title: "✅ Claim Submitted!",
          description: `Claim ID: ${data.claim_id} - Total: Rp ${(data.total_claim || calculateTotal()).toLocaleString('id-ID')}`,
          variant: "success",
        })
        // Reset form
        setFormData({ patient_name: '', patient_nik: '', patient_dob: '', patient_gender: 'M', patient_phone: '', patient_address: '', visit_date: '', visit_type: 'rawat jalan', doctor_name: '', department: 'Poli Umum', diagnosis_primary: '', diagnosis_secondary: '' })
        setProcedures([{ code: '', cost: '' }])
        setDrugs([{ code: '', cost: '' }])
        setVitamins([{ name: '', cost: '' }])
      } else {
        throw new Error(data.error || 'Failed to submit claim')
      }
    } catch (error) {
      console.error('Submit error:', error)
      toast({ title: "❌ Error", description: error.message, variant: "destructive" })
    } finally {
      setSubmitting(false)
    }
  }

  const formatRupiah = (num) => new Intl.NumberFormat('id-ID').format(num)

  return (
    <div className="max-w-5xl mx-auto px-4 py-8">
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white flex items-center gap-3">
              <FileText className="w-8 h-8 text-cyan-400" />Submit New Claim
            </h1>
            <p className="text-slate-400 mt-1">Fill in the claim details below</p>
          </div>
          <Button type="button" variant={showQuickFill ? "default" : "outline"} onClick={() => setShowQuickFill(!showQuickFill)} className="gap-2">
            <Sparkles className="w-4 h-4" />Quick Fill
          </Button>
        </div>
      </motion.div>

      <AnimatePresence>
        {showQuickFill && (
          <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }} className="mb-6 overflow-hidden">
            <Card className="border-slate-700 bg-slate-900/50">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium text-slate-300 flex items-center gap-2">
                  <Zap className="w-4 h-4 text-cyan-400" />Quick Fill Templates
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {Object.entries(AUTOFILL_TEMPLATES).map(([key, template]) => (
                    <button key={key} type="button" onClick={() => applyAutofill(key)} className={`p-4 rounded-xl text-left transition-all duration-200 ${template.type === 'normal' ? 'bg-emerald-500/10 border border-emerald-500/20 hover:border-emerald-500/40' : 'bg-red-500/10 border border-red-500/20 hover:border-red-500/40'}`}>
                      <div className="flex items-center gap-2 mb-2">
                        {template.type === 'normal' ? <CheckCircle className="w-4 h-4 text-emerald-400" /> : <AlertTriangle className="w-4 h-4 text-red-400" />}
                        <span className={`text-sm font-medium ${template.type === 'normal' ? 'text-emerald-400' : 'text-red-400'}`}>{template.label}</span>
                      </div>
                      <p className="text-xs text-slate-500">{template.description}</p>
                    </button>
                  ))}
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      <form onSubmit={handleSubmit}>
        <div className="space-y-6">
          {/* Patient */}
          <Card className="border-slate-800 bg-slate-900/50 backdrop-blur">
            <CardHeader><CardTitle className="text-lg flex items-center gap-2"><User className="w-5 h-5 text-cyan-400" />Patient Information</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2"><Label>Patient Name *</Label><Input value={formData.patient_name} onChange={(e) => handleInputChange('patient_name', e.target.value)} placeholder="Full name" required /></div>
                <div className="space-y-2"><Label>NIK *</Label><Input value={formData.patient_nik} onChange={(e) => handleInputChange('patient_nik', e.target.value)} placeholder="16-digit NIK" required /></div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="space-y-2"><Label>Date of Birth *</Label><Input type="date" value={formData.patient_dob} onChange={(e) => handleInputChange('patient_dob', e.target.value)} required /></div>
                <div className="space-y-2"><Label>Gender *</Label><Select value={formData.patient_gender} onValueChange={(v) => handleInputChange('patient_gender', v)}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent><SelectItem value="M">Male</SelectItem><SelectItem value="F">Female</SelectItem></SelectContent></Select></div>
                <div className="space-y-2"><Label>Phone</Label><Input value={formData.patient_phone} onChange={(e) => handleInputChange('patient_phone', e.target.value)} placeholder="08xxx" /></div>
              </div>
              <div className="space-y-2"><Label>Address</Label><Input value={formData.patient_address} onChange={(e) => handleInputChange('patient_address', e.target.value)} placeholder="Full address" /></div>
            </CardContent>
          </Card>

          {/* Visit */}
          <Card className="border-slate-800 bg-slate-900/50 backdrop-blur">
            <CardHeader><CardTitle className="text-lg flex items-center gap-2"><Calendar className="w-5 h-5 text-purple-400" />Visit Information</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2"><Label>Visit Date *</Label><Input type="date" value={formData.visit_date} onChange={(e) => handleInputChange('visit_date', e.target.value)} required /></div>
                <div className="space-y-2"><Label>Visit Type *</Label><Select value={formData.visit_type} onValueChange={(v) => handleInputChange('visit_type', v)}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent>{VISIT_TYPES.map(t => <SelectItem key={t} value={t}>{t}</SelectItem>)}</SelectContent></Select></div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2"><Label>Doctor Name *</Label><Input value={formData.doctor_name} onChange={(e) => handleInputChange('doctor_name', e.target.value)} placeholder="Dr. ..." required /></div>
                <div className="space-y-2"><Label>Department *</Label><Select value={formData.department} onValueChange={(v) => handleInputChange('department', v)}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent>{DEPARTMENTS.map(d => <SelectItem key={d} value={d}>{d}</SelectItem>)}</SelectContent></Select></div>
              </div>
            </CardContent>
          </Card>

          {/* Diagnosis */}
          <Card className="border-slate-800 bg-slate-900/50 backdrop-blur">
            <CardHeader><CardTitle className="text-lg flex items-center gap-2"><Heart className="w-5 h-5 text-pink-400" />Diagnosis (ICD-10)</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2"><Label>Primary Diagnosis *</Label><Select value={formData.diagnosis_primary} onValueChange={(v) => handleInputChange('diagnosis_primary', v)}><SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger><SelectContent>{ICD10_DATA.map(d => <SelectItem key={d.code} value={d.code}>{d.code} - {d.description}</SelectItem>)}</SelectContent></Select></div>
                <div className="space-y-2"><Label>Secondary Diagnosis</Label><Select value={formData.diagnosis_secondary || "none"} onValueChange={(v) => handleInputChange("diagnosis_secondary", v === "none" ? "" : v)}><SelectTrigger><SelectValue placeholder="Optional" /></SelectTrigger><SelectContent><SelectItem value="none">(None)</SelectItem>{ICD10_DATA.map(d => <SelectItem key={d.code} value={d.code}>{d.code} - {d.description}</SelectItem>)}</SelectContent></Select></div>
              </div>
            </CardContent>
          </Card>

          {/* Procedures */}
          <Card className="border-slate-800 bg-slate-900/50 backdrop-blur">
            <CardHeader><CardTitle className="text-lg flex items-center justify-between"><span className="flex items-center gap-2"><Stethoscope className="w-5 h-5 text-cyan-400" />Procedures (ICD-9)</span><Button type="button" variant="outline" size="sm" onClick={() => addRow('procedure')} className="gap-1"><Plus className="w-4 h-4" /> Add</Button></CardTitle></CardHeader>
            <CardContent>
              <div className="space-y-3">
                {procedures.map((proc, idx) => (
                  <motion.div key={idx} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="flex gap-3 items-end">
                    <div className="flex-1 space-y-1"><Label className="text-xs text-slate-500">Procedure</Label><Select value={proc.code} onValueChange={(v) => updateRow('procedure', idx, 'code', v)}><SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger><SelectContent>{ICD9_DATA.map(p => <SelectItem key={p.code} value={p.code}>{p.code} - {p.description}</SelectItem>)}</SelectContent></Select></div>
                    <div className="w-40 space-y-1"><Label className="text-xs text-slate-500">Cost (Rp)</Label><Input type="number" value={proc.cost} onChange={(e) => updateRow('procedure', idx, 'cost', e.target.value)} placeholder="0" /></div>
                    {procedures.length > 1 && <Button type="button" variant="ghost" size="icon" onClick={() => removeRow('procedure', idx)} className="text-red-400 hover:text-red-300 hover:bg-red-500/10"><Trash2 className="w-4 h-4" /></Button>}
                  </motion.div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Drugs */}
          <Card className="border-slate-800 bg-slate-900/50 backdrop-blur">
            <CardHeader><CardTitle className="text-lg flex items-center justify-between"><span className="flex items-center gap-2"><Pill className="w-5 h-5 text-emerald-400" />Medications</span><Button type="button" variant="outline" size="sm" onClick={() => addRow('drug')} className="gap-1"><Plus className="w-4 h-4" /> Add</Button></CardTitle></CardHeader>
            <CardContent>
              <div className="space-y-3">
                {drugs.map((drug, idx) => (
                  <motion.div key={idx} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="flex gap-3 items-end">
                    <div className="flex-1 space-y-1"><Label className="text-xs text-slate-500">Drug</Label><Select value={drug.code} onValueChange={(v) => updateRow('drug', idx, 'code', v)}><SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger><SelectContent>{DRUGS_DATA.map(d => <SelectItem key={d.code} value={d.code}>{d.code} - {d.name}</SelectItem>)}</SelectContent></Select></div>
                    <div className="w-40 space-y-1"><Label className="text-xs text-slate-500">Cost (Rp)</Label><Input type="number" value={drug.cost} onChange={(e) => updateRow('drug', idx, 'cost', e.target.value)} placeholder="0" /></div>
                    {drugs.length > 1 && <Button type="button" variant="ghost" size="icon" onClick={() => removeRow('drug', idx)} className="text-red-400 hover:text-red-300 hover:bg-red-500/10"><Trash2 className="w-4 h-4" /></Button>}
                  </motion.div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Vitamins */}
          <Card className="border-slate-800 bg-slate-900/50 backdrop-blur">
            <CardHeader><CardTitle className="text-lg flex items-center justify-between"><span className="flex items-center gap-2"><Sparkles className="w-5 h-5 text-amber-400" />Vitamins</span><Button type="button" variant="outline" size="sm" onClick={() => addRow('vitamin')} className="gap-1"><Plus className="w-4 h-4" /> Add</Button></CardTitle></CardHeader>
            <CardContent>
              <div className="space-y-3">
                {vitamins.map((vit, idx) => (
                  <motion.div key={idx} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="flex gap-3 items-end">
                    <div className="flex-1 space-y-1"><Label className="text-xs text-slate-500">Vitamin</Label><Select value={vit.name} onValueChange={(v) => updateRow('vitamin', idx, 'name', v)}><SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger><SelectContent>{VITAMINS_DATA.map(v => <SelectItem key={v} value={v}>{v}</SelectItem>)}</SelectContent></Select></div>
                    <div className="w-40 space-y-1"><Label className="text-xs text-slate-500">Cost (Rp)</Label><Input type="number" value={vit.cost} onChange={(e) => updateRow('vitamin', idx, 'cost', e.target.value)} placeholder="0" /></div>
                    {vitamins.length > 1 && <Button type="button" variant="ghost" size="icon" onClick={() => removeRow('vitamin', idx)} className="text-red-400 hover:text-red-300 hover:bg-red-500/10"><Trash2 className="w-4 h-4" /></Button>}
                  </motion.div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Total & Submit */}
          <Card className="border-cyan-500/30 bg-gradient-to-br from-slate-900 to-slate-800">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-400">Total Claim Amount</p>
                  <p className="text-3xl font-bold gradient-text">Rp {formatRupiah(calculateTotal())}</p>
                </div>
                <Button type="submit" size="lg" className="gap-2 px-8" disabled={submitting}>
                  {submitting ? <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1, ease: "linear" }} className="w-5 h-5 border-2 border-slate-900 border-t-transparent rounded-full" /> : <><Send className="w-4 h-4" />Submit Claim</>}
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </form>
    </div>
  )
}
