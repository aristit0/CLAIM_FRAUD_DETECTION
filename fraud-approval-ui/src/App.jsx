import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Login from '@/pages/Login'
import Dashboard from '@/pages/Dashboard'
import Review from '@/pages/Review'

const Protected = ({ children }) => localStorage.getItem('auth') ? children : <Navigate to="/" replace />

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Login />} />
        <Route path="/dashboard" element={<Protected><Dashboard /></Protected>} />
        <Route path="/review/:id" element={<Protected><Review /></Protected>} />
      </Routes>
    </BrowserRouter>
  )
}
