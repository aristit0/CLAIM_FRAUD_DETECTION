import React, { useEffect, useState } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { useApi } from './hooks/useApi'
import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import Review from './pages/Review'
import './index.css'

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [loading, setLoading] = useState(true)
  const { request } = useApi()

  useEffect(() => {
    // Check if user is already logged in (session cookie)
    // This would be set by the backend
    checkAuth()
  }, [])

  const checkAuth = async () => {
    // Try to fetch claims to verify session
    try {
      const response = await request('/api/claims?limit=1', 'GET')
      if (response && response.success) {
        setIsAuthenticated(true)
      }
    } catch (err) {
      setIsAuthenticated(false)
    } finally {
      setLoading(false)
    }
  }

  const handleLogin = (success) => {
    setIsAuthenticated(success)
  }

  const handleLogout = () => {
    setIsAuthenticated(false)
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading...</p>
        </div>
      </div>
    )
  }

  return (
    <BrowserRouter>
      <Routes>
        <Route
          path="/login"
          element={
            isAuthenticated ? <Navigate to="/" /> : <Login onLogin={handleLogin} />
          }
        />
        <Route
          path="/"
          element={
            isAuthenticated ? <Dashboard onLogout={handleLogout} /> : <Navigate to="/login" />
          }
        />
        <Route
          path="/review/:claimId"
          element={
            isAuthenticated ? <Review onLogout={handleLogout} /> : <Navigate to="/login" />
          }
        />
        <Route path="*" element={<Navigate to="/" />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
