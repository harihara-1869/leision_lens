/*
 * NewScanPage — route: "/scan/new"
 *
 * The "Upload & Analyse" page — the core feature of the portal.
 *
 * Layout:
 *   Header — "Analyse Retina" title + Station ID
 *   Main panel (glass card):
 *     Left half  — drag-drop upload zone + latency/encryption metadata
 *     Right half — circular scan viewer with pulsing ring + AI confidence
 *   Action row — START ANALYSIS button + RESET SCAN button
 *   Bottom bento — AI Enhancement, HIPAA Compliant, Auto-Reporting cards
 *
 * HOW TO CONNECT:
 *   1. File drop → useFileUpload hook captures the file
 *   2. "START ANALYSIS" click → uploadScan(file) → POST /api/scan/upload
 *      → returns { scanId }
 *   3. Then startAnalysis(scanId) → POST /api/scan/:id/analyze
 *      → returns { resultId }
 *   4. Navigate to /scan/:resultId/results
 *
 * The analysis is currently mocked (2s delay in api/index.js).
 * When your EfficientNet model is wrapped in Flask, replace those stubs.
 */

import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import PortalLayout from '../components/layout/PortalLayout.jsx'
import GlassCard from '../components/ui/GlassCard.jsx'
import Icon from '../components/ui/Icon.jsx'
import Button from '../components/ui/Button.jsx'
import { useFileUpload } from '../hooks/useFileUpload.js'
import { uploadScan, startAnalysis } from '../api/index.js'

// Default eye image shown in the scan viewer before a file is uploaded
const DEFAULT_EYE_IMG = 'https://lh3.googleusercontent.com/aida-public/AB6AXuCqG4YAncf0mXngpLbbNwRIkJOqDniYnKVmVrumpqGmFpgauFQGqZXUtf814txd6pgxZlBxtkdKCGdAmaPJVKOWUD0whsnYOaTk5DMtOp0-IcderWm0PNtBGsPz2GlB1HzWTF65bGETKk1ER47n66oQIxITXt-86bz_J4hDHPCxaNDuvCEt_FpfAF7Hve9Ar-vm3pshqhXiAey1Njzp_8etZA1sf0bZhVUudrSu1fCEpbHANPDT1IWIoiJOIglJqQwZGzyQfiJVT0zY'

// Bottom bento feature cards
const FEATURES = [
  {
    icon: 'auto_awesome',
    title: 'Ben Graham Preprocessing',
    desc: 'Illumination bias removal via Gaussian subtraction and CLAHE contrast enhancement on the green channel — standard in DR literature since 2015.',
  },
  {
    icon: 'security',
    title: 'Class Based Inference',
    desc: 'Trained with Weighted Random Sampling across five DR severity grades on APTOS 2019 — 5,993 labelled fundus photographs',
  },
  {
    icon: 'clinical_notes',
    title: 'Multi-Task Architecture',
    desc: 'Simultaneously grades DR severity (0–4) and detects four lesion types: microaneurysms, haemorrhages, hard and soft exudates.',
  },
]

// Analysis states: idle → uploading → analyzing → done
const STATE = { IDLE: 'idle', UPLOADING: 'uploading', ANALYZING: 'analyzing', DONE: 'done' }

export default function NewScanPage() {
  const navigate = useNavigate()
  const { isDragging, file, previewUrl, error: fileError, getRootProps, getInputProps, clearFile } =
    useFileUpload()

  const [analysisState, setAnalysisState] = useState(STATE.IDLE)
  const [scanError, setScanError]         = useState(null)
  const [prediction, setPrediction]       = useState(null)

  // The displayed eye image — user's upload preview or default
  const eyeImage = previewUrl ?? DEFAULT_EYE_IMG

  async function handleStartAnalysis() {
    if (!file) {
      setScanError('Please upload a retinal image first.')
      return
    }
    setScanError(null)

    try {
      setAnalysisState(STATE.UPLOADING)
      const { scanId, confidence, grade_label } = await uploadScan(file)
      setPrediction({ confidence, grade_label })

      setAnalysisState(STATE.ANALYZING)
      const { resultId } = await startAnalysis(scanId)

      setAnalysisState(STATE.DONE)
      navigate(`/scan/${resultId}/results`)
    } catch (err) {
      setScanError('Analysis failed. Please try again.')
      setAnalysisState(STATE.IDLE)
    }
  }

  function handleReset() {
    clearFile()
    setAnalysisState(STATE.IDLE)
    setScanError(null)
    setPrediction(null)
  }

  // Label shown inside the START ANALYSIS button based on current state
  const buttonLabel = {
    [STATE.IDLE]:      'Start Analysis',
    [STATE.UPLOADING]: 'Uploading…',
    [STATE.ANALYZING]: 'Analysing…',
    [STATE.DONE]:      'Complete!',
  }[analysisState]

  const isProcessing = analysisState === STATE.UPLOADING || analysisState === STATE.ANALYZING

  return (
    <PortalLayout activePage="scan">
      {/* ─── Header ───────────────────────────────────────────────────── */}
      <header className="flex justify-between items-start mb-10">
        <div>
          <h1 className="text-h1 font-bold text-on-surface">Analyse Retina</h1>
          <p className="text-on-surface-variant mt-1">
            Upload high-resolution fundus imagery for diagnostic assessment.
          </p>
        </div>
        {/* <div className="text-right">
          <p className="font-mono text-[10px] text-outline uppercase tracking-widest">Station ID</p>
          <p className="font-mono text-sm text-primary font-bold">LL-XC-9021</p>
        </div> */}
      </header>

      {/* ─── Main Upload + Viewer Panel ───────────────────────────────── */}
      <GlassCard className="rounded-3xl p-6 md:p-10 mb-8">
        <div className="flex flex-col lg:flex-row gap-10 items-center">

          {/* ── Left: Drop zone ──────────────────────────────────────── */}
          <div className="w-full lg:w-1/2 flex flex-col gap-6">
            {/* Drag-and-drop zone */}
            <div
              {...getRootProps()}
              className={`
                relative flex flex-col items-center justify-center
                border-2 border-dashed rounded-3xl p-10 md:p-12 cursor-pointer
                transition-all duration-300
                ${isDragging
                  ? 'border-primary bg-primary/10 shadow-[inset_0_0_20px_rgba(99,87,137,0.15)]'
                  : 'border-primary-container/40 hover:border-primary-container hover:bg-primary-container/5'
                }
              `}
            >
              {/* Hidden file input */}
              <input {...getInputProps()} />

              {/* Upload icon or preview thumbnail */}
              {previewUrl ? (
                <div className="relative mb-4">
                  <img
                    src={previewUrl}
                    alt="Uploaded scan"
                    className="w-24 h-24 rounded-full object-cover border-4 border-primary-container/40 shadow-lg"
                  />
                  <span className="absolute -top-1 -right-1 w-6 h-6 bg-primary-container rounded-full flex items-center justify-center">
                    <Icon name="check" className="text-white text-sm" />
                  </span>
                </div>
              ) : (
                <div className="w-16 h-16 rounded-full bg-primary-container/10 flex items-center justify-center mb-6 transition-transform group-hover:scale-110">
                  <Icon name="add" className="text-primary text-3xl" />
                </div>
              )}

              <h2 className="text-h2 font-semibold text-on-surface text-center mb-2">
                {previewUrl ? file?.name ?? 'Image ready' : 'Drop image here'}
              </h2>
              <p className="text-on-surface-variant text-sm text-center px-4">
                {previewUrl
                  ? `${(file?.size / (1024 * 1024)).toFixed(1)} MB — ready for analysis`
                  : 'Supports JPG, PNG. Recommended: high-resolution fundus photograph, minimum 512×512px.'}
              </p>

              {/* Indicator dots */}
              <div className="mt-6 flex gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-primary-container/60" />
                <div className="w-1.5 h-1.5 rounded-full bg-primary-container/30" />
                <div className="w-1.5 h-1.5 rounded-full bg-primary-container/10" />
              </div>
            </div>

            {/* File validation error */}
            {fileError && (
              <p className="text-error text-sm bg-error-container px-4 py-3 rounded-xl">{fileError}</p>
            )}

            {/* Metadata pills: latency + encryption */}
            {/* <div className="grid grid-cols-2 gap-4">
              {[
                { label: 'Latency',    value: '14ms Global Hook' },
                { label: 'Encryption', value: 'AES-256 Protocol' },
              ].map((meta) => (
                <div key={meta.label} className="p-4 bg-white/20 rounded-xl border border-white/30">
                  <span className="font-mono text-[10px] text-outline uppercase tracking-widest block mb-1">
                    {meta.label}
                  </span>
                  <span className="font-mono text-sm text-on-surface">{meta.value}</span>
                </div>
              ))}
            </div> */}
          </div>

          {/* ── Right: Scan Viewer ────────────────────────────────────── */}
          <div className="w-full lg:w-1/2 flex flex-col items-center gap-8">
            {/* Circular scan viewer with pulsing ring */}
            <div className="relative">
              {/* Outer ring */}
              <div className="w-56 h-56 md:w-72 md:h-72 rounded-full border-2 border-primary/20 flex items-center justify-center p-2">
                {/* Pulsing inner ring */}
                <div className="w-full h-full rounded-full border border-primary/40 scan-ring-pulse flex items-center justify-center overflow-hidden bg-slate-900 relative">
                  {/* Eye image */}
                  <img
                    src={eyeImage}
                    alt="Retina scan viewer"
                    className="w-full h-full object-cover opacity-70 mix-blend-screen"
                  />
                  {/* Crosshair overlays */}
                  <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                    <div className="w-px h-24 bg-primary/40" />
                    <div className="h-px w-24 bg-primary/40 absolute" />
                    <div className="w-20 h-20 rounded-full border border-primary/20 absolute" />
                  </div>
                </div>
              </div>

              {/* AI Confidence badge — top right */}
              <div className="absolute top-0 -right-4 md:-right-10 glass bg-white/80 px-3 py-2 rounded-xl border border-white/60 shadow-sm">
                <span className="font-mono text-[9px] font-bold text-primary block uppercase">AI Confidence</span>
                <span className="font-mono text-sm text-on-surface font-bold">
                  {prediction ? `${(prediction.confidence * 100).toFixed(1)}%` : '-- %'}
                </span>
              </div>

              {/* Grade badge — bottom left */}
              <div className="absolute bottom-8 -left-4 md:-left-12 glass bg-white/80 px-3 py-2 rounded-xl border border-white/60 shadow-sm">
                <span className="font-mono text-[9px] font-bold text-secondary block uppercase">Grade</span>
                <span className="font-mono text-sm text-on-surface font-bold">
                  {prediction ? prediction.grade_label : 'Pending'}
                </span>
              </div>
            </div>

            {/* Processing status + action buttons */}
            <div className="text-center">
              <span className="font-mono text-[10px] text-outline block mb-4 uppercase tracking-[0.2em]">
                LESION-LENS v1.0 — EFFICIENTNET-B4
              </span>

              {scanError && (
                <p className="text-error text-sm bg-error-container px-4 py-2 rounded-xl mb-4">{scanError}</p>
              )}

              <div className="flex items-center gap-4">
                <Button
                  variant="primary"
                  size="md"
                  loading={isProcessing}
                  onClick={handleStartAnalysis}
                  className="px-8 py-3.5 rounded-full shadow-lg shadow-primary/20"
                >
                  {buttonLabel}
                </Button>
                <Button
                  variant="secondary"
                  size="md"
                  onClick={handleReset}
                  disabled={isProcessing}
                  className="px-6 py-3.5"
                >
                  Reset Scan
                </Button>
              </div>
            </div>
          </div>
        </div>
      </GlassCard>

      {/* ─── Bottom Bento Feature Cards ───────────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {FEATURES.map((feat) => (
          <GlassCard key={feat.title} className="p-6 md:p-8 rounded-3xl">
            <div className="flex items-center gap-3 mb-4">
              <Icon name={feat.icon} className="text-primary text-2xl" />
              <h3 className="font-semibold text-on-surface">{feat.title}</h3>
            </div>
            <p className="text-sm text-on-surface-variant leading-relaxed">{feat.desc}</p>
          </GlassCard>
        ))}
      </div>
    </PortalLayout>
  )
}
