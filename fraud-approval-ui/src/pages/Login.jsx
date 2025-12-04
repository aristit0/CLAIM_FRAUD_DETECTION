// ================================================================
// Login Page Component
// ================================================================

import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useApi } from '../hooks/useApi'
import { Button } from '../components/ui/button'
import { Input } from '../components/ui/input'
import { Card } from '../components/ui/card'

export default function Login({ onLogin }) {
  const navigate = useNavigate()
  const { loading, error, request } = useApi()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [localError, setLocalError] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLocalError('')

    if (!username || !password) {
      setLocalError('Username and password are required')
      return
    }

    const response = await request('/api/login', 'POST', {
      username,
      password,
    })

    if (response && response.success) {
      onLogin(true)
      navigate('/')
    } else {
      setLocalError(error || 'Login failed')
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <Card className="w-full max-w-md shadow-lg">
        <div className="p-8">
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-gray-900">BPJS Fraud Detection</h1>
            <p className="text-gray-600 mt-2">Approval Review System</p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            {(localError || error) && (
              <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
                {localError || error}
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Username
              </label>
              <Input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter username"
                disabled={loading}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Password
              </label>
              <Input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter password"
                disabled={loading}
              />
            </div>

            <Button
              type="submit"
              disabled={loading}
              className="w-full"
            >
              {loading ? 'Signing in...' : 'Sign In'}
            </Button>
          </form>

          <div className="mt-6 text-center text-sm text-gray-600">
            <p>Demo credentials:</p>
            <p>Username: <strong>aris</strong></p>
            <p>Password: <strong>Admin123</strong></p>
          </div>
        </div>
      </Card>
    </div>
  )
}
