// ================================================================
// Dashboard Page Component - Claims List
// ================================================================

import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useApi } from '../hooks/useApi'
import { Button } from '../components/ui/button'
import { Card } from '../components/ui/card'
import { Badge } from '../components/ui/badge'
import { LogOut, Eye, ChevronLeft, ChevronRight } from 'lucide-react'

export default function Dashboard({ onLogout }) {
  const navigate = useNavigate()
  const { loading, error, request } = useApi()
  const [claims, setClaims] = useState([])
  const [page, setPage] = useState(1)
  const [totalPages, setTotalPages] = useState(0)
  const [stats, setStats] = useState({})

  useEffect(() => {
    fetchClaims()
    fetchStats()
  }, [page])

  const fetchClaims = async () => {
    const response = await request(`/api/claims?page=${page}&limit=20`, 'GET')
    if (response && response.success) {
      setClaims(response.data || [])
      setTotalPages(response.pagination?.total_pages || 0)
    }
  }

  const fetchStats = async () => {
    const response = await request('/api/statistics', 'GET')
    if (response && response.success) {
      setStats(response)
    }
  }

  const handleLogout = async () => {
    await request('/api/logout', 'POST')
    onLogout()
    navigate('/login')
  }

  const getRiskBadge = (amount) => {
    if (amount > 1000000) return 'HIGH'
    if (amount > 500000) return 'MEDIUM'
    return 'LOW'
  }

  const formatDate = (dateStr) => {
    return new Date(dateStr).toLocaleDateString('id-ID', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    })
  }

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('id-ID', {
      style: 'currency',
      currency: 'IDR',
      minimumFractionDigits: 0
    }).format(amount)
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Claims Approval</h1>
            <p className="text-gray-600 mt-1">Fraud Detection System</p>
          </div>
          <Button
            variant="outline"
            onClick={handleLogout}
            className="flex items-center gap-2"
          >
            <LogOut className="w-4 h-4" />
            Logout
          </Button>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <Card className="p-6">
            <p className="text-gray-600 text-sm font-medium">Pending</p>
            <p className="text-3xl font-bold text-gray-900 mt-2">
              {stats.status_counts?.pending || 0}
            </p>
          </Card>
          <Card className="p-6">
            <p className="text-gray-600 text-sm font-medium">Approved</p>
            <p className="text-3xl font-bold text-green-600 mt-2">
              {stats.status_counts?.approved || 0}
            </p>
          </Card>
          <Card className="p-6">
            <p className="text-gray-600 text-sm font-medium">Declined</p>
            <p className="text-3xl font-bold text-red-600 mt-2">
              {stats.status_counts?.declined || 0}
            </p>
          </Card>
          <Card className="p-6">
            <p className="text-gray-600 text-sm font-medium">Manual Review</p>
            <p className="text-3xl font-bold text-yellow-600 mt-2">
              {stats.status_counts?.manual_review || 0}
            </p>
          </Card>
        </div>

        {/* Claims Table */}
        <Card>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 border-b">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase">
                    Claim ID
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase">
                    Patient
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase">
                    Department
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase">
                    Amount
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase">
                    Date
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase">
                    Action
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {loading && (
                  <tr>
                    <td colSpan="7" className="px-6 py-8 text-center text-gray-600">
                      Loading claims...
                    </td>
                  </tr>
                )}
                {error && (
                  <tr>
                    <td colSpan="7" className="px-6 py-8 text-center text-red-600">
                      Error: {error}
                    </td>
                  </tr>
                )}
                {claims.length === 0 && !loading && !error && (
                  <tr>
                    <td colSpan="7" className="px-6 py-8 text-center text-gray-600">
                      No pending claims
                    </td>
                  </tr>
                )}
                {claims.map((claim) => (
                  <tr key={claim.claim_id} className="hover:bg-gray-50 transition">
                    <td className="px-6 py-4 text-sm font-medium text-blue-600">
                      #{claim.claim_id}
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-900">
                      {claim.patient_name}
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-700">
                      {claim.department}
                    </td>
                    <td className="px-6 py-4 text-sm font-medium text-gray-900">
                      {formatCurrency(claim.total_claim_amount)}
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-700">
                      {formatDate(claim.visit_date)}
                    </td>
                    <td className="px-6 py-4">
                      <Badge
                        variant={
                          claim.status === 'pending' ? 'default' :
                          claim.status === 'approved' ? 'success' :
                          claim.status === 'declined' ? 'destructive' : 'secondary'
                        }
                      >
                        {claim.status}
                      </Badge>
                    </td>
                    <td className="px-6 py-4">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => navigate(`/review/${claim.claim_id}`)}
                        className="flex items-center gap-2"
                      >
                        <Eye className="w-4 h-4" />
                        Review
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="px-6 py-4 border-t flex items-center justify-between">
              <p className="text-sm text-gray-600">
                Page {page} of {totalPages}
              </p>
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  disabled={page === 1}
                  onClick={() => setPage(page - 1)}
                  className="flex items-center gap-1"
                >
                  <ChevronLeft className="w-4 h-4" />
                  Previous
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  disabled={page === totalPages}
                  onClick={() => setPage(page + 1)}
                  className="flex items-center gap-1"
                >
                  Next
                  <ChevronRight className="w-4 h-4" />
                </Button>
              </div>
            </div>
          )}
        </Card>
      </div>
    </div>
  )
}
