import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { TooltipProvider } from '@/components/ui/tooltip'
import LoginPage from '@/pages/Login'
import Dashboard from '@/pages/Dashboard'
import ReviewPage from '@/pages/Review'

function ProtectedRoute({ children }) {
  const isAuth = localStorage.getItem('isAuthenticated')
  if (!isAuth) {
    return <Navigate to="/" replace />
  }
  return children
}

export default function App() {
  return (
    <TooltipProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<LoginPage />} />
          <Route
            path="/dashboard"
            element={
              <ProtectedRoute>
                <Dashboard />
              </ProtectedRoute>
            }
          />
          <Route
            path="/review/:claimId"
            element={
              <ProtectedRoute>
                <ReviewPage />
              </ProtectedRoute>
            }
          />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  )
}
