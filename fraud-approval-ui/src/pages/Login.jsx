import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Shield, User, Lock, Sparkles } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/components'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'

export default function Login() {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()

  const handleSubmit = (e) => {
    e.preventDefault()
    setLoading(true)
    if (username === 'aris' && password === 'Admin123') {
      localStorage.setItem('auth', 'true')
      localStorage.setItem('user', username)
      navigate('/dashboard')
    } else {
      setError('Username atau password salah')
    }
    setLoading(false)
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
        <Card className="w-full max-w-md glass-strong neon-border">
          <CardHeader className="text-center space-y-4">
            <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: 0.2, type: 'spring' }}
              className="mx-auto w-20 h-20 rounded-2xl bg-gradient-to-br from-purple-600 to-cyan-500 flex items-center justify-center glow-purple">
              <Shield className="w-10 h-10 text-white" />
            </motion.div>
            <CardTitle className="text-3xl gradient-text">Fraud Approval</CardTitle>
            <p className="text-white/60">AI-Powered Claim Review System</p>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-5">
              {error && <div className="bg-red-500/10 border border-red-500/30 text-red-400 px-4 py-3 rounded-xl text-sm">{error}</div>}
              <div className="relative">
                <User className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-white/40" />
                <Input placeholder="Username" value={username} onChange={(e) => setUsername(e.target.value)} className="pl-12" required />
              </div>
              <div className="relative">
                <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-white/40" />
                <Input type="password" placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)} className="pl-12" required />
              </div>
              <Button type="submit" className="w-full" size="lg" disabled={loading}>
                <Sparkles className="w-5 h-5 mr-2" /> Sign In
              </Button>
              <p className="text-center text-xs text-white/40">Demo: aris / Admin123</p>
            </form>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  )
}
