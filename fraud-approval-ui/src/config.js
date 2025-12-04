// ================================================================
// API Configuration - Production Ready
// ================================================================

const isDev = import.meta.env.DEV

export const API_BASE = isDev
  ? 'http://127.0.0.1:2223/api'
  : '/api'

export const API_ENDPOINTS = {
  // Authentication
  login: `${API_BASE}/login`,
  logout: `${API_BASE}/logout`,

  // Claims Management
  getClaimsList: `${API_BASE}/claims`,
  getClaimDetail: (claimId) => `${API_BASE}/claims/${claimId}`,
  updateClaim: (claimId) => `${API_BASE}/claims/${claimId}/update`,

  // Statistics & Dashboard
  getStatistics: `${API_BASE}/statistics`,
}

export default API_BASE
