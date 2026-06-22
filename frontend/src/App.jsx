/*
 * App.jsx — root component
 *
 * Sets up React Router and wraps every route with AuthProvider.
 *
 * Route map:
 *   /                          → LandingPage    (public)
 *   /login                     → LoginPage      (public)
 *   /signup                    → SignUpPage      (public)
 *   /dashboard                 → DashboardPage  (protected)
 *   /scan/new                  → NewScanPage     (protected)
 *   /scan/:scanId/results      → ResultsDetailPage (protected)
 *   /patients                  → PatientHistoryPage (protected)
 *   /analytics                 → AnalyticsPage  (protected)
 *   /settings                  → SettingsPage   (protected)
 *   *                          → redirect to /  (catch-all)
 *
 * ProtectedRoute:
 *   Reads isAuthenticated from AuthContext.
 *   If not authenticated → redirects to /login, preserving the
 *   intended destination in location.state.from so LoginPage can
 *   send the user there after a successful login.
 */

import { Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { useAuth } from './context/AuthContext.jsx'

// Public pages
import LandingPage        from './pages/LandingPage.jsx'
import LoginPage          from './pages/LoginPage.jsx'
import SignUpPage         from './pages/SignUpPage.jsx'

// Protected portal pages
import DashboardPage      from './pages/DashboardPage.jsx'
import NewScanPage        from './pages/NewScanPage.jsx'
import ResultsDetailPage  from './pages/ResultsDetailPage.jsx'
import PatientHistoryPage from './pages/PatientHistoryPage.jsx'
import AnalyticsPage      from './pages/AnalyticsPage.jsx'
import SettingsPage       from './pages/SettingsPage.jsx'

// ── ProtectedRoute ────────────────────────────────────────────────────────────
// Wraps any route that requires the user to be signed in.
// Shows nothing during rehydration to avoid flashing the login page on refresh.
function ProtectedRoute({ children }) {
  const { isAuthenticated, loading } = useAuth()
  const location = useLocation()

  // Still rehydrating from localStorage — don't redirect yet
  if (loading) return null

  if (!isAuthenticated) {
    // Pass current path so LoginPage can redirect back after login
    return <Navigate to="/login" state={{ from: location }} replace />
  }

  return children
}

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  return (
    <Routes>
      {/* ── Public routes ───────────────────────────────────────────────── */}
      <Route path="/"       element={<LandingPage />} />
      <Route path="/login"  element={<LoginPage />} />
      <Route path="/signup" element={<SignUpPage />} />

      {/* ── Protected portal routes ─────────────────────────────────────── */}
      <Route
        path="/dashboard"
        element={<ProtectedRoute><DashboardPage /></ProtectedRoute>}
      />
      <Route
        path="/scan/new"
        element={<ProtectedRoute><NewScanPage /></ProtectedRoute>}
      />
      <Route
        path="/scan/:scanId/results"
        element={<ProtectedRoute><ResultsDetailPage /></ProtectedRoute>}
      />
      <Route
        path="/patients"
        element={<ProtectedRoute><PatientHistoryPage /></ProtectedRoute>}
      />
      <Route
        path="/analytics"
        element={<ProtectedRoute><AnalyticsPage /></ProtectedRoute>}
      />
      <Route
        path="/settings"
        element={<ProtectedRoute><SettingsPage /></ProtectedRoute>}
      />

      {/* ── Catch-all: redirect any unknown path to landing ─────────────── */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
