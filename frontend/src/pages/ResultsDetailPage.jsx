/*
 * ResultsDetailPage — route: "/scan/:scanId/results"
 *
 * Full AI diagnostic output:
 *   1. Patient info header
 *   2. Side-by-side: original + Grad-CAM heatmap
 *   3. Lesion segmentation overlay with per-type toggles
 *   4. DR Severity Scale
 *   5. Lesion Probabilities + AI Diagnostic Insight
 */

import { useState, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import PortalLayout from '../components/layout/PortalLayout.jsx'
import GlassCard from '../components/ui/GlassCard.jsx'
import Icon from '../components/ui/Icon.jsx'
import Button from '../components/ui/Button.jsx'
import { getScanResults } from '../api/index.js'
import { DR_GRADE_LABELS, LESION_DISPLAY } from '../mock/mockPatients.js'

// DR severity scale
const DR_STEPS = [
  { grade: 0, label: 'No DR' },
  { grade: 1, label: 'Mild' },
  { grade: 2, label: 'Moderate' },
  { grade: 3, label: 'Severe' },
  { grade: 4, label: 'Proliferative' },
]

// Lesion metadata for the probability bars
const LESION_META = {
  MA: { full: 'MA (Microaneurysms)', barColor: 'bg-primary-container' },
  HE: { full: 'HE (Hemorrhages)',    barColor: 'bg-secondary-container' },
  EX: { full: 'EX (Exudates)',       barColor: 'bg-outline-variant' },
  SE: { full: 'SE (Soft Exudates)',  barColor: 'bg-outline-variant' },
}

export default function ResultsDetailPage() {
  const { scanId }        = useParams()
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(true)
  const [activeSegLesions, setActiveSegLesions] = useState({
    MA: true, HE: true, EX: true, SE: true,
  })

  useEffect(() => {
    getScanResults(scanId).then((data) => {
      setResult(data)
      setLoading(false)
    })
  }, [scanId])

  const toggleLesion = (key) => {
    setActiveSegLesions((prev) => ({ ...prev, [key]: !prev[key] }))
  }

  // Count how many are active
  const activeCount = Object.values(activeSegLesions).filter(Boolean).length

  if (loading) {
    return (
      <PortalLayout activePage="scan">
        <div className="space-y-6 animate-pulse">
          <div className="h-20 bg-white/30 rounded-2xl" />
          <div className="h-96 bg-white/30 rounded-3xl" />
          <div className="h-32 bg-white/30 rounded-3xl" />
        </div>
      </PortalLayout>
    )
  }

  return (
    <PortalLayout activePage="scan">
      {/* ─── 1. Patient info header ──────────────────────────────────── */}
      <GlassCard className="rounded-2xl p-5 md:p-6 mb-6 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div className="flex flex-wrap items-center gap-6 md:gap-8">
          <div>
            <p className="font-mono text-[10px] text-on-surface-variant uppercase tracking-widest mb-1">
              Patient Name
            </p>
          </div>
        </div>
      </GlassCard>

      {/* ─── 2. Side-by-side: Original + Grad-CAM ───────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        {/* Original scan */}
        <div className="glass bg-white/30 rounded-3xl overflow-hidden relative group h-64 md:h-96">
          <img
            src={result.originalUrl}
            alt="Original retinal scan"
            className="w-full h-full object-cover grayscale opacity-90 group-hover:scale-105 transition-transform duration-700"
          />
          <div className="absolute top-4 left-4 bg-black/30 backdrop-blur-sm px-3 py-1.5 rounded-full border border-white/20">
            <span className="font-mono text-[10px] text-white uppercase tracking-widest">
              Original Scan
            </span>
          </div>
        </div>

        {/* Grad-CAM heatmap */}
        <div className="glass bg-white/30 rounded-3xl overflow-hidden relative group h-64 md:h-96">
          <div className="absolute inset-0 bg-gradient-to-tr from-violet-500/20 to-orange-500/20 mix-blend-overlay z-10" />
          <img
            src={result.heatmapUrl}
            alt="AI Grad-CAM heatmap overlay"
            className="w-full h-full object-cover opacity-80 group-hover:scale-105 transition-transform duration-700"
          />
          <div className="absolute top-4 left-4 bg-primary/40 backdrop-blur-sm px-3 py-1.5 rounded-full border border-white/20 z-20">
            <span className="font-mono text-[10px] text-white uppercase tracking-widest">
              AI Heatmap Overlay
            </span>
          </div>
          <div className="absolute bottom-4 right-4 flex gap-2 z-20">
            <div className="w-2 h-2 rounded-full bg-orange-400" />
            <div className="w-2 h-2 rounded-full bg-violet-400" />
          </div>
          <div className="absolute inset-0 border-2 border-primary-container/40 rounded-3xl pointer-events-none" />
        </div>
      </div>

      {/* ─── 3. Lesion Segmentation Overlay ──────────────────────────── */}
      <GlassCard className="rounded-3xl p-6 md:p-8 mb-6">
        <div className="flex justify-between items-center mb-6">
          <h3 className="font-mono text-[11px] text-on-surface-variant/60 uppercase tracking-[0.2em]">
            Lesion Segmentation
          </h3>
          <span className="font-mono text-[10px] text-on-surface-variant/40">
            {activeCount}/4 visible
          </span>
        </div>

        {/* Segmentation image */}
        <div className="glass bg-white/30 rounded-3xl overflow-hidden relative group h-64 md:h-80 mb-6">
          <img
            src={result.segOverlayUrl}
            alt="Lesion segmentation overlay"
            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-700"
          />
          <div className="absolute top-4 left-4 bg-black/40 backdrop-blur-sm px-3 py-1.5 rounded-full border border-white/20">
            <span className="font-mono text-[10px] text-white uppercase tracking-widest">
              Lesion Overlay
            </span>
          </div>
        </div>

        {/* Per-lesion toggle pills */}
        <div className="flex flex-wrap gap-3 justify-center">
          {Object.entries(LESION_DISPLAY).map(([key, meta]) => {
            const isActive = activeSegLesions[key]
            return (
              <button
                key={key}
                onClick={() => toggleLesion(key)}
                className={`
                  flex items-center gap-2 px-4 py-2 rounded-full font-mono text-xs
                  transition-all duration-200 border-2
                  ${isActive
                    ? 'bg-white/90 text-on-surface shadow-md border-transparent'
                    : 'bg-white/20 text-on-surface-variant/50 border-outline-variant/30'
                  }
                `}
              >
                {/* Colour dot */}
                <span
                  className={`w-2.5 h-2.5 rounded-full transition-opacity ${
                    isActive ? 'opacity-100' : 'opacity-30'
                  }`}
                  style={{ backgroundColor: meta.color }}
                />
                <span>{meta.short}</span>
                <span className="text-on-surface-variant/60">{meta.full}</span>
              </button>
            )
          })}
        </div>
      </GlassCard>

      {/* ─── 4. DR Severity Scale ───────────────────────────────────── */}
      <GlassCard className="rounded-3xl p-6 md:p-8 mb-6">
        <h3 className="font-mono text-[11px] text-on-surface-variant/60 uppercase tracking-[0.2em] mb-8 text-center">
          Diabetic Retinopathy Severity Classification
        </h3>

        <div className="relative px-6 md:px-12">
          <div className="absolute h-0.5 top-[6px] left-12 right-12 bg-outline-variant/40 rounded-full" />

          <div className="relative flex justify-between">
            {DR_STEPS.map((step) => {
              const isActive = step.grade === result.drGrade
              return (
                <div key={step.grade} className="flex flex-col items-center gap-3 relative z-10">
                  <div
                    className={`rounded-full ring-4 ring-white transition-all ${
                      isActive
                        ? 'w-6 h-6 bg-primary-container -mt-1.5 shadow-xl shadow-primary-container/40'
                        : 'w-3 h-3 bg-outline-variant/50'
                    }`}
                  />
                  {isActive ? (
                    <div className="flex flex-col items-center gap-1">
                      <div className="bg-primary-container px-3 py-0.5 rounded-full">
                        <span className="font-bold text-[11px] text-on-primary-container">{step.label}</span>
                      </div>
                    </div>
                  ) : (
                    <span className="font-semibold text-[11px] text-on-surface-variant/50">
                      {step.label}
                    </span>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      </GlassCard>

      {/* ─── 5. Lesion Probabilities + AI Insight ───────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Lesion probability bars */}
        <GlassCard className="col-span-1 md:col-span-2 rounded-3xl p-6 md:p-8">
          <div className="flex justify-between items-end mb-8">
            <h3 className="font-mono text-[11px] text-on-surface-variant/60 uppercase tracking-[0.2em]">
              Lesion Probabilities
            </h3>
            <p className="font-bold text-xs text-primary">
              Confidence: {result.confidence}%
            </p>
          </div>

          <div className="space-y-6">
            {Object.entries(result.lesionProbs).map(([key, pct]) => {
              const meta = LESION_META[key]
              return (
                <div key={key}>
                  <div className="flex justify-between mb-2">
                    <span className="text-xs font-bold text-on-surface">{meta.full}</span>
                    <span className="font-mono text-xs">
                      {String(pct).padStart(2, '0')}%
                    </span>
                  </div>
                  <div className="h-1 w-full bg-outline-variant/20 rounded-full overflow-hidden">
                    <div
                      className={`h-full ${meta.barColor} progress-animate rounded-full`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              )
            })}
          </div>
        </GlassCard>

        {/* AI Diagnostic Insight */}
        <GlassCard className="col-span-1 rounded-3xl p-6 md:p-8 flex flex-col bg-primary/5 border border-primary/20">
          <div className="w-12 h-12 bg-white rounded-2xl flex items-center justify-center mb-6 shadow-sm">
            <Icon name="neurology" className="text-primary text-2xl" />
          </div>

          <h3 className="text-lg font-bold text-primary mb-4">Diagnostic Insight</h3>

          <p className="text-sm text-on-surface-variant leading-relaxed flex-1">
            {result.aiInsight}
          </p>
        </GlassCard>
      </div>
    </PortalLayout>
  )
}
