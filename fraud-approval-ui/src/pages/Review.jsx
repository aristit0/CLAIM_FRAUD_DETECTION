// ================================================================
// Review Page Component - Claim Details & Decision
// ================================================================

import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useApi } from '../hooks/useApi'
import { Button } from '../components/ui/button'
import { Card } from '../components/ui/card'
import { Badge } from '../components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs'
import { AlertCircle, CheckCircle2, XCircle, ArrowLeft } from 'lucide-react'

export default function Review({ onLogout }) {
  const { claimId } = useParams()
  const navigate = useNavigate()
  const { loading, error, request } = useApi()
  const [claim, setClaim] = useState(null)
  const [scoring, setScoring] = useState(null)
  const [decision, setDecision] = useState(null)
  const [notes, setNotes] = useState('')
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    fetchClaimDetails()
  }, [claimId])

  const fetchClaimDetails = async () => {
    const response = await request(`/api/claims/${claimId}`, 'GET')
    if (response && response.success) {
      setClaim(response.header)
      setScoring(response.scoring)
    }
  }

  const handleApprove = async () => {
    setSubmitting(true)
    const response = await request(`/api/claims/${claimId}/update`, 'POST', {
      status: 'approved',
      notes: notes,
    })
    setSubmitting(false)
    if (response && response.success) {
      navigate('/')
    }
  }

  const handleDecline = async () => {
    setSubmitting(true)
    const response = await request(`/api/claims/${claimId}/update`, 'POST', {
      status: 'declined',
      notes: notes,
    })
    setSubmitting(false)
    if (response && response.success) {
      navigate('/')
    }
  }

  const handleManualReview = async () => {
    setSubmitting(true)
    const response = await request(`/api/claims/${claimId}/update`, 'POST', {
      status: 'manual_review',
      notes: notes,
    })
    setSubmitting(false)
    if (response && response.success) {
      navigate('/')
    }
  }

  if (loading) {
    return <div className="flex items-center justify-center h-screen">Loading...</div>
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-screen">
        <Card className="p-8 max-w-md">
          <p className="text-red-600">Error: {error}</p>
          <Button onClick={() => navigate('/')} className="mt-4 w-full">
            Back to Dashboard
          </Button>
        </Card>
      </div>
    )
  }

  if (!claim || !scoring) {
    return <div className="flex items-center justify-center h-screen">No data found</div>
  }

  const model = scoring.model_output
  const getRiskColor = (level) => {
    if (level === 'HIGH RISK') return 'destructive'
    if (level === 'MODERATE RISK') return 'secondary'
    return 'default'
  }

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('id-ID', {
      style: 'currency',
      currency: 'IDR',
      minimumFractionDigits: 0
    }).format(amount || 0)
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <Button
            variant="outline"
            onClick={() => navigate('/')}
            className="flex items-center gap-2 mb-4"
          >
            <ArrowLeft className="w-4 h-4" />
            Back
          </Button>
          <div className="flex justify-between items-start">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Claim Review #{claimId}
              </h1>
              <p className="text-gray-600">{claim.patient_name}</p>
            </div>
            <Badge variant={getRiskColor(model.risk_level)} className="text-lg px-4 py-2">
              {model.risk_level}
            </Badge>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left: Claim Details */}
          <div className="lg:col-span-2 space-y-6">
            {/* AI Analysis */}
            <Card className="p-6 border-2 border-yellow-200 bg-yellow-50">
              <div className="flex gap-4">
                <AlertCircle className="w-6 h-6 text-yellow-600 flex-shrink-0 mt-1" />
                <div>
                  <h3 className="font-bold text-gray-900 mb-2">AI Analysis</h3>
                  <p className="text-gray-700 text-sm mb-4">{scoring.ai_explanation}</p>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-gray-600">Fraud Probability</p>
                      <p className="text-xl font-bold text-gray-900">{model.fraud_probability}</p>
                    </div>
                    <div>
                      <p className="text-gray-600">Confidence</p>
                      <p className="text-xl font-bold text-gray-900">{(model.confidence * 100).toFixed(0)}%</p>
                    </div>
                  </div>
                </div>
              </div>
            </Card>

            {/* Tabs */}
            <Tabs defaultValue="details" className="w-full">
              <TabsList>
                <TabsTrigger value="details">Claim Details</TabsTrigger>
                <TabsTrigger value="analysis">Analysis</TabsTrigger>
              </TabsList>

              <TabsContent value="details" className="space-y-4">
                <Card className="p-6">
                  <h3 className="font-bold mb-4">Patient Information</h3>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-gray-600">NIK</p>
                      <p className="font-medium">{claim.patient_nik}</p>
                    </div>
                    <div>
                      <p className="text-gray-600">Name</p>
                      <p className="font-medium">{claim.patient_name}</p>
                    </div>
                    <div>
                      <p className="text-gray-600">Visit Date</p>
                      <p className="font-medium">{new Date(claim.visit_date).toLocaleDateString('id-ID')}</p>
                    </div>
                    <div>
                      <p className="text-gray-600">Department</p>
                      <p className="font-medium">{claim.department}</p>
                    </div>
                  </div>
                </Card>

                <Card className="p-6">
                  <h3 className="font-bold mb-4">Claim Amount</h3>
                  <p className="text-3xl font-bold text-gray-900">
                    {formatCurrency(claim.total_claim_amount)}
                  </p>
                </Card>
              </TabsContent>

              <TabsContent value="analysis" className="space-y-4">
                <Card className="p-6">
                  <h3 className="font-bold mb-4">Risk Factors</h3>
                  <div className="space-y-3">
                    {model.top_risk_factors.map((factor, idx) => (
                      <div key={idx} className="flex justify-between items-center p-3 bg-red-50 rounded-lg">
                        <span className="text-gray-900">{factor.interpretation}</span>
                        <Badge variant="destructive">
                          {(factor.importance * 100).toFixed(0)}%
                        </Badge>
                      </div>
                    ))}
                  </div>
                </Card>

                <Card className="p-6">
                  <h3 className="font-bold mb-4">Features</h3>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="p-3 bg-gray-50 rounded">
                      <p className="text-gray-600">Mismatch Count</p>
                      <p className="text-2xl font-bold text-gray-900">
                        {model.features.mismatch_count}
                      </p>
                    </div>
                    <div className="p-3 bg-gray-50 rounded">
                      <p className="text-gray-600">Cost Anomaly Score</p>
                      <p className="text-2xl font-bold text-gray-900">
                        {model.features.cost_anomaly_score}
                      </p>
                    </div>
                  </div>
                </Card>
              </TabsContent>
            </Tabs>
          </div>

          {/* Right: Decision Panel */}
          <div>
            <Card className="p-6 sticky top-4">
              <h3 className="font-bold text-lg mb-4">Your Decision</h3>

              <div className="mb-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
                <p className="text-sm text-blue-900">
                  <strong>Model Recommendation:</strong> {model.recommendation}
                </p>
              </div>

              <textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Add reviewer notes..."
                className="w-full p-3 border rounded-lg text-sm mb-4 focus:outline-none focus:ring-2 focus:ring-blue-500"
                rows="4"
              />

              <div className="space-y-2">
                <Button
                  onClick={handleApprove}
                  disabled={submitting}
                  className="w-full bg-green-600 hover:bg-green-700 text-white flex items-center justify-center gap-2"
                >
                  <CheckCircle2 className="w-4 h-4" />
                  {submitting ? 'Processing...' : 'Approve'}
                </Button>

                <Button
                  onClick={handleDecline}
                  disabled={submitting}
                  className="w-full bg-red-600 hover:bg-red-700 text-white flex items-center justify-center gap-2"
                >
                  <XCircle className="w-4 h-4" />
                  {submitting ? 'Processing...' : 'Decline'}
                </Button>

                <Button
                  onClick={handleManualReview}
                  disabled={submitting}
                  variant="outline"
                  className="w-full"
                >
                  {submitting ? 'Processing...' : 'Manual Review'}
                </Button>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
