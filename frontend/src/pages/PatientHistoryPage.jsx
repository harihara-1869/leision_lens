/*
 * PatientHistoryPage — route: "/patients"
 *
 * Comprehensive registry of all diagnostic retinal scans.
 *
 * Layout:
 *   Header — title + search bar + filter button
 *   Glass table — scan preview, patient name, ID, date, DR severity, action
 *   Pagination row — "Showing X of Y" + Previous/Next
 *   Quick Insights — 3 stat mini-cards (Critical Flagging, Weekly Volume, AI Confidence)
 *
 * HOW TO CONNECT:
 *   On mount → getPatients() → GET /api/patients?page=1
 *   Search input → getPatients({ search }) → GET /api/patients?search=...
 *   "View Full Report" → navigate to /scan/:id/results
 *   Filter button → open filter drawer (Phase 2 feature — skeleton only)
 */

import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import PortalLayout from '../components/layout/PortalLayout.jsx'
import GlassCard from '../components/ui/GlassCard.jsx'
import SeverityBadge from '../components/ui/SeverityBadge.jsx'
import Icon from '../components/ui/Icon.jsx'
import { getPatients } from '../api/index.js'

// Quick insight cards at the bottom of the page
const INSIGHTS = [
  // {
  //   label: 'Critical Flagging',
  //   value: '12',
  //   badge: '+3 today',
  //   badgeColor: 'bg-error-container text-error',
  //   valueColor: 'text-error',
  //   desc: 'Scans requiring immediate clinician review.',
  // },
  // {
  //   label: 'Weekly Volume',
  //   value: '342',
  //   badge: 'Normal',
  //   badgeColor: 'bg-primary-fixed text-primary',
  //   valueColor: 'text-primary',
  //   desc: 'Diagnostic retinal scans processed this week.',
  // },
  // {
  //   label: 'AI Confidence',
  //   value: '98.4%',
  //   badge: 'Stable',
  //   badgeColor: 'bg-tertiary-fixed text-tertiary',
  //   valueColor: 'text-tertiary',
  //   desc: 'Average verification score across all models.',
  // },
]

export default function PatientHistoryPage() {
  const navigate = useNavigate()

  const [patients, setPatients]   = useState([])
  const [total, setTotal]         = useState(0)
  const [search, setSearch]       = useState('')
  const [loading, setLoading]     = useState(true)

  // Fetch patients — re-runs when search changes (debounced via useEffect)
  useEffect(() => {
    const timer = setTimeout(() => {
      setLoading(true)
      getPatients({ search })
        .then(({ patients: list, total: count }) => {
          setPatients(list)
          setTotal(count)
          setLoading(false)
        })
        .catch(() => setLoading(false))
    }, 300) // 300ms debounce so we don't spam on every keystroke
    return () => clearTimeout(timer)
  }, [search])

  return (
    <PortalLayout activePage="history">
      {/* ─── Header ───────────────────────────────────────────────────── */}
      <header className="flex flex-col sm:flex-row justify-between items-start sm:items-end gap-6 mb-10">
        <div>
          <h1 className="text-h1 font-bold text-on-surface">Patient History</h1>
          <p className="text-on-surface-variant mt-1 max-w-lg">
            Registry of diagnostic retinal scans
          </p>
        </div>

        {/* Search + Filter controls */}
        <div className="flex gap-3 w-full sm:w-auto">
          {/* Search bar */}
          <div className="glass bg-white/40 px-4 py-2.5 rounded-xl flex items-center gap-3 flex-1 sm:flex-none sm:w-52">
            <Icon name="search" className="text-on-surface-variant text-[20px]" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search patient name…"
              className="bg-transparent border-none focus:ring-0 text-sm w-full placeholder:text-outline/60 text-on-surface"
            />
            {search && (
              <button onClick={() => setSearch('')} className="text-outline hover:text-primary">
                <Icon name="close" className="text-[16px]" />
              </button>
            )}
          </div>

          {/* Filter button */}
          <button className="glass bg-white/40 glass-hover px-4 py-2.5 rounded-xl flex items-center gap-2 text-sm font-semibold text-on-surface-variant hover:text-primary transition-colors">
            <Icon name="filter_list" className="text-[20px]" />
            <span className="hidden sm:inline">Filter</span>
          </button>
        </div>
      </header>

      {/* ─── Patient Table ─────────────────────────────────────────────── */}
      <GlassCard className="rounded-3xl p-6 md:p-8 mb-8 overflow-x-auto">
        <table className="w-full text-left min-w-[600px]">
          {/* Column headers */}
          <thead>
            <tr>
              {['Scan Preview', 'Patient Name', 'ID Number', 'Analysis Date', 'DR Severity', 'Action'].map(
                (col) => (
                  <th
                    key={col}
                    className="pb-4 font-mono text-[10px] uppercase tracking-widest text-on-surface-variant/60 px-3 first:pl-0 last:pr-0 last:text-right"
                  >
                    {col}
                  </th>
                )
              )}
            </tr>
          </thead>

          <tbody className="divide-y divide-white/10">
            {loading
              ? Array.from({ length: 4 }).map((_, i) => (
                  <tr key={i}>
                    <td colSpan={6} className="py-4 px-3">
                      <div className="h-16 bg-white/30 rounded-2xl animate-pulse" />
                    </td>
                  </tr>
                ))
              : patients.map((patient) => (
                  <tr
                    key={patient.id}
                    className="group hover:bg-white/40 transition-all duration-200"
                  >
                    {/* Scan thumbnail */}
                    <td className="py-4 pl-0 pr-3">
                      <div className="w-14 h-14 rounded-full overflow-hidden border-2 border-primary/20 relative">
                        <img
                          src={patient.previewUrl}
                          alt={`${patient.name} scan`}
                          className="w-full h-full object-cover"
                        />
                        {/* Hover zoom hint */}
                        <div className="absolute inset-0 bg-primary/20 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                          <Icon name="zoom_in" className="text-white text-sm" />
                        </div>
                      </div>
                    </td>

                    {/* Patient name */}
                    <td className="py-4 px-3 font-semibold text-on-surface">{patient.name}</td>

                    {/* Patient ID */}
                    <td className="py-4 px-3 font-mono text-sm text-on-surface-variant">
                      #{patient.patientId}
                    </td>

                    {/* Scan date */}
                    <td className="py-4 px-3 text-on-surface-variant text-sm">{patient.scanDate}</td>

                    {/* DR Severity badge */}
                    <td className="py-4 px-3">
                      <SeverityBadge grade={patient.drGrade} />
                    </td>

                    {/* Action button */}
                    <td className="py-4 pl-3 pr-0 text-right">
                      <button
                        onClick={() => navigate(`/scan/${patient.id}/results`)}
                        className="bg-secondary-container text-on-secondary-container px-4 py-2 rounded-lg font-bold text-xs uppercase tracking-wider hover:shadow-lg hover:shadow-secondary-container/30 transition-all"
                      >
                        View Full Report
                      </button>
                    </td>
                  </tr>
                ))}
          </tbody>
        </table>

        {/* Pagination row */}
        <div className="mt-6 pt-6 border-t border-white/20 flex justify-between items-center">
          <span className="font-mono text-xs text-on-surface-variant uppercase tracking-widest">
            Showing {patients.length} of {total.toLocaleString()} patients
          </span>
          <div className="flex gap-4">
            <button
              disabled
              className="flex items-center gap-1 font-mono text-xs text-on-surface-variant/40 cursor-not-allowed uppercase tracking-widest"
            >
              <Icon name="chevron_left" className="text-[18px]" />
              Previous
            </button>
            <button className="flex items-center gap-1 font-mono text-xs text-on-surface-variant hover:text-primary transition-colors uppercase tracking-widest">
              Next
              <Icon name="chevron_right" className="text-[18px]" />
            </button>
          </div>
        </div>
      </GlassCard>

      {/* ─── Quick Insights ─────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
        {INSIGHTS.map((insight) => (
          <GlassCard key={insight.label} className="p-6 rounded-3xl">
            <p className="font-mono text-[10px] text-on-surface-variant uppercase tracking-widest mb-3">
              {insight.label}
            </p>
            <div className="flex items-end justify-between mb-3">
              <span className={`text-3xl font-bold ${insight.valueColor}`}>{insight.value}</span>
              <span className={`text-xs font-bold px-2 py-1 rounded ${insight.badgeColor}`}>
                {insight.badge}
              </span>
            </div>
            <p className="text-xs text-on-surface-variant/70 leading-relaxed">{insight.desc}</p>
          </GlassCard>
        ))}
      </div>
    </PortalLayout>
  )
}
