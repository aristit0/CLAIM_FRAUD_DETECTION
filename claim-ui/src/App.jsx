import React, { useState } from 'react'
import LoginPage from './LoginPage'
import ClaimsListPage from './ClaimsListPage'
import SubmitClaimPage from './SubmitClaimPage'

function App() {
  const [user, setUser] = useState(null)
  const [page, setPage] = useState('claims') // 'claims' | 'submit'

  const handleLogin = (username) => {
    setUser({ username })
  }

  const handleLogout = () => {
    setUser(null)
    setPage('claims')
  }

  if (!user) {
    // Penting: selalu kirim prop onLogin ke LoginPage
    return <LoginPage onLogin={handleLogin} />
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-50">
      {/* Navbar sederhana */}
      <header className="border-b border-slate-800 px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-cyan-400 font-semibold">ClaimFlow</span>
          <span className="text-xs text-slate-500">Dashboard</span>
        </div>
        <div className="flex items-center gap-4 text-sm">
          <span className="text-slate-400">Hi, {user.username}</span>
          <button
            onClick={handleLogout}
            className="px-3 py-1 rounded-md border border-slate-700 hover:bg-slate-800 text-xs"
          >
            Logout
          </button>
        </div>
      </header>

      {/* Tabs / menu */}
      <nav className="px-6 py-3 border-b border-slate-800 flex gap-2 text-sm">
        <button
          onClick={() => setPage('claims')}
          className={
            'px-3 py-1 rounded-md ' +
            (page === 'claims'
              ? 'bg-cyan-500 text-slate-900'
              : 'bg-slate-800 text-slate-300')
          }
        >
          Daftar Klaim
        </button>
        <button
          onClick={() => setPage('submit')}
          className={
            'px-3 py-1 rounded-md ' +
            (page === 'submit'
              ? 'bg-cyan-500 text-slate-900'
              : 'bg-slate-800 text-slate-300')
          }
        >
          Submit Klaim
        </button>
      </nav>

      {/* Konten utama */}
      <main className="p-6">
        {page === 'claims' && <ClaimsListPage />}
        {page === 'submit' && <SubmitClaimPage />}
      </main>
    </div>
  )
}

export default App