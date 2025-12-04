import React, { useState, useEffect } from 'react'
import { Routes, Route, Navigate, useNavigate } from 'react-router-dom'
import Layout from '@/components/Layout'
import LoginPage from '@/pages/LoginPage'
import SubmitClaimPage from '@/pages/SubmitClaimPage'
import ClaimsListPage from '@/pages/ClaimsListPage'
import { Toaster } from '@/components/ui/toaster'

function App() {
  const [user, setUser] = useState(() => {
    return localStorage.getItem('claimflow_user') || null
  })
  const navigate = useNavigate()

  const handleLogin = (username) => {
    setUser(username)
    localStorage.setItem('claimflow_user', username)
    navigate('/submit')
  }

  const handleLogout = () => {
    setUser(null)
    localStorage.removeItem('claimflow_user')
    navigate('/')
  }

  if (!user) {
    return (
      <>
        <LoginPage onLogin={handleLogin} />
        <Toaster />
      </>
    )
  }

  return (
    <>
      <Layout onLogout={handleLogout}>
        <Routes>
          <Route path="/" element={<Navigate to="/submit" replace />} />
          <Route path="/submit" element={<SubmitClaimPage />} />
          <Route path="/claims" element={<ClaimsListPage />} />
          <Route path="*" element={<Navigate to="/submit" replace />} />
        </Routes>
      </Layout>
      <Toaster />
    </>
  )
}

export default App
