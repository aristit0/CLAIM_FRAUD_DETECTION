import { useState, useCallback } from 'react'

const API_BASE = '/api'

export function useApi() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const fetchClaims = useCallback(async (page = 1, limit = 20) => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/claims?page=${page}&limit=${limit}`)
      const data = await res.json()
      return data
    } catch (err) {
      setError(err.message)
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  const fetchClaimDetail = useCallback(async (claimId) => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/get_claim`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ claim_id: claimId }),
      })
      const data = await res.json()
      return data
    } catch (err) {
      setError(err.message)
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  const updateClaimStatus = useCallback(async (claimId, status) => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/update_status/${claimId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status }),
      })
      const data = await res.json()
      return data
    } catch (err) {
      setError(err.message)
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  const login = useCallback(async (username, password) => {
    setLoading(true)
    setError(null)
    try {
      const formData = new FormData()
      formData.append('username', username)
      formData.append('password', password)
      
      const res = await fetch('/', {
        method: 'POST',
        body: formData,
      })
      
      if (res.redirected || res.url.includes('dashboard')) {
        return { success: true }
      }
      return { success: false, error: 'Invalid credentials' }
    } catch (err) {
      setError(err.message)
      return { success: false, error: err.message }
    } finally {
      setLoading(false)
    }
  }, [])

  return {
    loading,
    error,
    fetchClaims,
    fetchClaimDetail,
    updateClaimStatus,
    login,
  }
}

// Mock data for development
export const mockClaims = [
  {
    claim_id: 1001,
    patient_name: 'Ahmad Rizki',
    visit_date: '2024-01-15',
    visit_type: 'Inpatient',
    department: 'Cardiology',
    total_claim_amount: 45000000,
    status: 'pending',
  },
  {
    claim_id: 1002,
    patient_name: 'Siti Nurhaliza',
    visit_date: '2024-01-14',
    visit_type: 'Outpatient',
    department: 'Orthopedics',
    total_claim_amount: 12500000,
    status: 'pending',
  },
  {
    claim_id: 1003,
    patient_name: 'Budi Santoso',
    visit_date: '2024-01-13',
    visit_type: 'Emergency',
    department: 'Emergency',
    total_claim_amount: 78000000,
    status: 'pending',
  },
  {
    claim_id: 1004,
    patient_name: 'Dewi Lestari',
    visit_date: '2024-01-12',
    visit_type: 'Inpatient',
    department: 'Oncology',
    total_claim_amount: 125000000,
    status: 'pending',
  },
  {
    claim_id: 1005,
    patient_name: 'Eko Prasetyo',
    visit_date: '2024-01-11',
    visit_type: 'Outpatient',
    department: 'Dermatology',
    total_claim_amount: 3500000,
    status: 'pending',
  },
]

export const mockClaimDetail = {
  claim_id: 1001,
  header: {
    claim_id: 1001,
    patient_name: 'Ahmad Rizki',
    patient_nik: '3201234567890001',
    patient_gender: 'Male',
    patient_dob: '1985-06-15',
    patient_phone: '081234567890',
    patient_address: 'Jl. Sudirman No. 123, Jakarta Selatan',
    visit_date: '2024-01-15',
    visit_type: 'Inpatient',
    department: 'Cardiology',
    doctor_name: 'Dr. Agus Setiawan, Sp.JP',
    status: 'pending',
    total_claim_amount: 45000000,
    total_procedure_cost: 25000000,
    total_drug_cost: 15000000,
    total_vitamin_cost: 5000000,
  },
  model_output: {
    fraud_score: 0.72,
    fraud_probability: '72%',
    risk_level: 'HIGH RISK',
    risk_color: 'red',
    clinical_compatibility: {
      procedure_compatible: true,
      drug_compatible: false,
      vitamin_compatible: true,
      overall_compatible: false,
    },
    features: {
      mismatch_count: 2,
      cost_anomaly: 3,
      cost_anomaly_score: 3,
      total_claim: 45000000,
    },
    explanation: 'Ketidaksesuaian terdeteksi: Obat tidak sesuai dengan diagnosis utama. Biaya prosedur melebihi rata-rata 3 standar deviasi.',
    recommendation: 'Perlu investigasi lebih lanjut terkait resep obat.',
    confidence: 0.85,
    fraud_flag: 1,
    top_risk_factors: [
      { interpretation: 'Drug prescription mismatch', importance: 0.92 },
      { interpretation: 'High cost anomaly detected', importance: 0.88 },
      { interpretation: 'Unusual procedure combination', importance: 0.75 },
    ],
  },
  diagnosis: [
    { icd10_code: 'I25.1', icd10_description: 'Atherosclerotic heart disease', is_primary: true },
    { icd10_code: 'E11.9', icd10_description: 'Type 2 diabetes mellitus', is_primary: false },
  ],
  procedures: [
    { icd9_code: '36.01', icd9_description: 'PTCA - Single vessel', cost: 15000000 },
    { icd9_code: '88.72', icd9_description: 'Diagnostic cardiac catheterization', cost: 10000000 },
  ],
  drugs: [
    { drug_code: 'ASP100', drug_name: 'Aspirin 100mg', cost: 500000 },
    { drug_code: 'CLO75', drug_name: 'Clopidogrel 75mg', cost: 2500000 },
    { drug_code: 'ATV40', drug_name: 'Atorvastatin 40mg', cost: 12000000 },
  ],
  vitamins: [
    { vitamin_name: 'Vitamin B Complex', cost: 2500000 },
    { vitamin_name: 'Omega-3 Fish Oil', cost: 2500000 },
  ],
}
