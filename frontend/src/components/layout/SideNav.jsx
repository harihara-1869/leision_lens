/*
 * SideNav — shared sidebar navigation for all authenticated portal pages.
 *
 * Layout: fixed left sidebar, 256px wide, full viewport height.
 * Glass morphism background matches the design exactly.
 *
 * Props:
 *   activePage — string matching one of the NAV_ITEMS keys:
 *                'dashboard' | 'scan' | 'history' | 'analytics' | 'settings'
 *
 * Navigation: uses React Router <Link> so clicking changes the URL.
 * When you add the analytics page, set its path in NAV_ITEMS below.
 *
 * Mobile: on small screens (<768px) the sidebar is hidden and a hamburger
 * menu button is shown in the top-left. Clicking it overlays the sidebar.
 * This keeps the portal usable on tablets at the bedside.
 */

import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import Icon from '../ui/Icon.jsx'
import Button from '../ui/Button.jsx'
import { useAuth } from '../../context/AuthContext.jsx'

// Each nav item: icon, label, and the route it links to
const NAV_ITEMS = [
  { key: 'dashboard', icon: 'dashboard',   label: 'Dashboard',       path: '/dashboard' },
  { key: 'scan',      icon: 'visibility',  label: 'New Scan',        path: '/scan/new' },
  { key: 'history',   icon: 'history',     label: 'Patient History', path: '/patients' },
  { key: 'analytics', icon: 'leaderboard', label: 'Analytics',       path: '/analytics' },
  { key: 'settings',  icon: 'settings',    label: 'Settings',        path: '/settings' },
]

export default function SideNav({ activePage }) {
  const [mobileOpen, setMobileOpen] = useState(false)
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  const clinician = user ?? { name: 'Clinician' }

  function handleLogout() {
    logout()
    navigate('/login')
  }

  const sidebarContent = (
    <aside className="flex flex-col h-full p-4">
      {/* ─── Brand ──────────────────────────────────────────────────── */}
      <div className="px-4 py-4 mb-2">
        <h1 className="text-lg font-black text-primary tracking-wide leading-none">
          Leison Lens
        </h1>
        <p className="text-[10px] font-mono text-on-surface-variant uppercase tracking-widest mt-1">
          Retinopathy Analysis
        </p>
      </div>

      {/* ─── Navigation Items ────────────────────────────────────────── */}
      <nav className="flex-1 space-y-1 mt-4">
        {NAV_ITEMS.map((item) => {
          const isActive = activePage === item.key
          return (
            <Link
              key={item.key}
              to={item.path}
              onClick={() => setMobileOpen(false)}
              className={`
                flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold
                tracking-wide transition-all duration-200
                ${
                  isActive
                    ? 'bg-primary/10 text-primary shadow-[inset_0_0_15px_rgba(255,255,255,0.4)]'
                    : 'text-on-surface-variant hover:text-primary hover:bg-white/10'
                }
              `}
            >
              <Icon name={item.icon} filled={isActive} className="text-[20px]" />
              <span>{item.label}</span>
            </Link>
          )
        })}
      </nav>

      {/* ─── Bottom: Clinician Profile + Emergency ───────────────────── */}
      <div className="mt-auto pt-6 border-t border-white/20 space-y-4">
        {/* Clinician profile row */}
        <button
          onClick={handleLogout}
          className="flex items-center gap-3 px-2 w-full hover:bg-white/10 rounded-xl p-2 transition-colors"
          title="Click to log out"
        >
          <div className="w-10 h-10 rounded-full bg-primary-container overflow-hidden flex-shrink-0 ring-2 ring-white/50">
            {clinician.avatarUrl ? (
              <img src={clinician.avatarUrl} alt="Clinician" className="w-full h-full object-cover" />
            ) : (
              <span className="flex items-center justify-center w-full h-full text-white font-bold text-sm">
                {clinician.initials ?? 'Dr'}
              </span>
            )}
          </div>
          <div className="text-left min-w-0">
            <p className="text-xs font-bold text-on-surface truncate">{clinician.name}</p>
            <p className="text-[10px] text-on-surface-variant truncate">{clinician.role}</p>
          </div>
        </button>

        {/* Emergency support — uses secondary (terracotta) color */}
        {/* <Button
          variant="danger"
          size="sm"
          className="w-full rounded-xl justify-center text-xs uppercase tracking-widest"
          onClick={() => alert('Emergency support contact: +1 800 555 0199')}
        >
          Emergency Support
        </Button> */}
      </div>
    </aside>
  )

  return (
    <>
      {/* ─── Mobile hamburger (hidden on md+) ───────────────────────── */}
      <button
        className="fixed top-4 left-4 z-50 md:hidden glass bg-white/60 p-2 rounded-lg"
        onClick={() => setMobileOpen(!mobileOpen)}
        aria-label="Toggle navigation"
      >
        <Icon name={mobileOpen ? 'close' : 'menu'} className="text-primary" />
      </button>

      {/* ─── Mobile overlay backdrop ─────────────────────────────────── */}
      {mobileOpen && (
        <div
          className="fixed inset-0 bg-black/30 z-40 md:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* ─── Sidebar — fixed on desktop, slide-in on mobile ─────────── */}
      <div
        className={`
          fixed left-0 top-0 h-full w-64 z-40
          glass-sidebar
          transform transition-transform duration-300
          ${mobileOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
        `}
      >
        {sidebarContent}
      </div>
    </>
  )
}
