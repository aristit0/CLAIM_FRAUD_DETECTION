import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Auth
export const login = async (username, password) => {
  const response = await api.post('/login', { username, password })
  return response.data
}

export const logout = async () => {
  const response = await api.post('/logout')
  return response.data
}

export const checkAuth = async () => {
  const response = await api.get('/auth/check')
  return response.data
}

// Master Data
export const getMasterData = async () => {
  const response = await api.get('/master-data')
  return response.data
}

// Claims
export const submitClaim = async (claimData) => {
  const response = await api.post('/claims', claimData)
  return response.data
}

export const getClaims = async (page = 1, limit = 10) => {
  const response = await api.get(`/claims?page=${page}&limit=${limit}`)
  return response.data
}

export const getClaimById = async (id) => {
  const response = await api.get(`/claims/${id}`)
  return response.data
}

export default api
