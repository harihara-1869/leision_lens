/*
 * DashboardPage — route: "/dashboard"
 *
 * The main clinician overview page after login.
 *
 * Sections:
 *   1. Header — greeting + notification/search icons
 *   2. Stats row — 4 glass cards (Total Scans, Early Detections, AI Accuracy, Backlog)
 *   3. Recent Activity table — patient rows with Review Scan buttons
 *
 * HOW TO CONNECT:
 *   On mount → calls getDashboard() from api/index.js
 *   → GET /api/dashboard/stats (returns stats + activity)
 *   "Review Scan" button → navigate to /scan/:id/results
 *   "Export Data" button → GET /api/dashboard/export (CSV download)
 */

import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import PortalLayout from '../components/layout/PortalLayout.jsx'
import GlassCard from '../components/ui/GlassCard.jsx'
import Icon from '../components/ui/Icon.jsx'
import Button from '../components/ui/Button.jsx'
import { useAuth } from '../context/AuthContext.jsx'
import { getDashboard } from '../api/index.js'

export default function DashboardPage() {
  const navigate = useNavigate()
  const { user } = useAuth()
  const [data, setData]       = useState(null)
  const [loading, setLoading] = useState(true)

  // Fetch dashboard data on mount
  useEffect(() => {
    getDashboard()
      .then((res) => {
        setData(res)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  return (
    <PortalLayout activePage="dashboard">
      {/* ─── Page Header ──────────────────────────────────────────────── */}
      <header className="flex justify-between items-start mb-10">
        <div>
          <h2 className="text-h1 font-bold text-on-surface">Clinician Overview</h2>
          <p className="text-on-surface-variant mt-1">
            Welcome back, {user?.name ?? 'Doctor'}. Here is your diagnostic summary for today.
          </p>
        </div>
        {/* Notification + search icon buttons */}
        <div className="flex gap-3">
          {['notifications', 'search'].map((icon) => (
            <button
              key={icon}
              className="w-11 h-11 glass bg-white/40 glass-hover rounded-full flex items-center justify-center text-primary transition-all"
              aria-label={icon}
            >
              <Icon name={icon} />
            </button>
          ))}
        </div>
      </header>

      {/* ─── Stats Grid ───────────────────────────────────────────────── */}
      <section className="grid grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
        {loading
          ? Array.from({ length: 4 }).map((_, i) => (
              <GlassCard key={i} className="p-6 rounded-2xl animate-pulse h-36" />
            ))
          : data?.stats.map((stat) => (
              <GlassCard key={stat.id} className="p-6 rounded-2xl flex flex-col">
                <span className="font-mono text-xs text-on-surface-variant uppercase tracking-widest mb-3">
                  {stat.label}
                </span>
                <span className={`font-bold text-3xl md:text-4xl ${stat.color} leading-none`}>
                  {stat.value}
                </span>
                <div className="mt-4 flex items-center gap-2">
                  <span className={`w-0.5 h-4 ${stat.accentColor} rounded-full`} />
                  <span className="font-mono text-xs text-on-surface-variant">{stat.subtext}</span>
                </div>
              </GlassCard>
            ))}
      </section>

      {/* ─── Recent Activity ──────────────────────────────────────────── */}
      <GlassCard className="rounded-2xl p-6 md:p-8">
        {/* Table header row */}
        <div className="flex justify-between items-end mb-6">
          <div>
            <h3 className="text-h2 font-semibold text-on-surface">Recent Activity</h3>
            <p className="text-on-surface-variant text-sm mt-1">
              Patient screening history and automated grading
            </p>
          </div>
          <Button
            variant="ghost"
            size="sm"
            icon="download"
            className="text-primary text-xs"
          >
            Export Data
          </Button>
        </div>

        {/* Activity rows */}
        <div className="space-y-2">
          {loading
            ? Array.from({ length: 4 }).map((_, i) => (
                <div key={i} className="h-16 bg-white/30 rounded-xl animate-pulse" />
              ))
            : data?.recent.map((item) => (
                <div
                  key={item.id}
                  className="grid grid-cols-6 items-center p-4 rounded-xl hover:bg-white/40 transition-all duration-200 border border-transparent hover:border-white/50 gap-2"
                >
                  {/* Patient name + avatar */}
                  <div className="col-span-6 sm:col-span-2 flex items-center gap-3">
                    <div className="w-9 h-9 rounded-full bg-primary-container/20 flex items-center justify-center font-bold text-primary text-xs flex-shrink-0">
                      {item.initials}
                    </div>
                    <div className="min-w-0">
                      <p className="font-semibold text-on-surface text-sm truncate">{item.name}</p>
                      <p className="font-mono text-xs text-on-surface-variant">ID: {item.patientId}</p>
                    </div>
                  </div>

                  {/* Scan Date */}
                  <div className="hidden sm:block col-span-1">
                    <p className="font-mono text-[10px] text-on-surface-variant uppercase tracking-wider mb-1">Scan Date</p>
                    <p className="font-mono text-sm text-on-surface">{item.scanDate}</p>
                  </div>

                  {/* DR Grade */}
                  <div className="hidden sm:block col-span-1">
                    <p className="font-mono text-[10px] text-on-surface-variant uppercase tracking-wider mb-1">DR Grade</p>
                    <div className="flex items-center gap-1.5">
                      <span className={`w-2 h-2 rounded-full ${item.drDotColor}`} />
                      <p className={`font-semibold text-sm ${item.drGradeColor}`}>
                        {item.drGradeLabel}
                      </p>
                    </div>
                  </div>

                  {/* Confidence */}
                  <div className="hidden sm:block col-span-1">
                    <p className="font-mono text-[10px] text-on-surface-variant uppercase tracking-wider mb-1">Confidence</p>
                    <p className="font-mono text-sm">{item.confidence}</p>
                  </div>

                  {/* Action */}
                  <div className="col-span-6 sm:col-span-1 flex justify-end">
                    <button
                      onClick={() => navigate(`/scan/${item.id}/results`)}
                      className="px-4 py-2 border border-primary text-primary rounded-full text-xs font-bold hover:bg-primary hover:text-white transition-all whitespace-nowrap"
                    >
                      Review Scan
                    </button>
                  </div>
                </div>
              ))}
        </div>
      </GlassCard>
    </PortalLayout>
  )
}
