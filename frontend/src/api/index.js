import { MOCK_PATIENTS } from '../mock/mockPatients.js'
import { MOCK_STATS, MOCK_ACTIVITY, MOCK_CLINICIAN } from '../mock/mockDashboard.js'

const BASE_URL = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')

const GRADE_DESCRIPTIONS = {
  0: 'No signs of diabetic retinopathy detected. Routine annual screening recommended.',
  1: 'Mild non-proliferative DR. Microaneurysms present. Annual monitoring advised.',
  2: 'Moderate non-proliferative DR. Referral to ophthalmologist within 6 months.',
  3: 'Severe non-proliferative DR. Urgent referral required.',
  4: 'Proliferative DR. Neovascularisation present. Immediate referral required.',
}

function authHeader() {
  const token = localStorage.getItem('token')
  return token ? { Authorization: `Bearer ${token}` } : {}
}

async function handleResponse(res) {
  if (!res.ok) {
    let message = `Request failed (${res.status})`
    try {
      const body = await res.json()
      if (body.error) message = body.error
    } catch (_) {}
    throw new Error(message)
  }
  return res.json()
}

// ─── Auth ────────────────────────────────────────────────────────────────────

export async function login(email, password) {
  const res = await fetch(`${BASE_URL}/api/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  })
  return handleResponse(res)
}

export async function signup(name, email, password) {
  const res = await fetch(`${BASE_URL}/api/auth/signup`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, email, password }),
  })
  return handleResponse(res)
}

// Kept for SignUpPage compatibility — maps the form object to signup()
export async function register(formData) {
  const res = await fetch(`${BASE_URL}/api/auth/signup`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      name: formData.fullName || formData.name || '',
      email: formData.email || '',
      password: formData.password || '',
    }),
  })
  await handleResponse(res)
  return { message: 'Account created successfully' }
}

// ─── Core predict (new) ──────────────────────────────────────────────────────

export async function predictScan(imageFile, patientName = '', patientAge = '') {
  const form = new FormData()
  form.append('image', imageFile)
  if (patientName) form.append('patient_name', patientName)
  if (patientAge)  form.append('patient_age', String(patientAge))

  const res = await fetch(`${BASE_URL}/api/predict`, {
    method: 'POST',
    headers: authHeader(),
    body: form,
  })
  return handleResponse(res)
}

// ─── Scans (new) ─────────────────────────────────────────────────────────────

export async function getScans() {
  const res = await fetch(`${BASE_URL}/api/scans/`, {
    headers: authHeader(),
  })
  return handleResponse(res)
}

export async function getScan(scanId) {
  const res = await fetch(`${BASE_URL}/api/scans/${scanId}`, {
    headers: authHeader(),
  })
  return handleResponse(res)
}

// ─── Scan result shape for ResultsDetailPage ─────────────────────────────────

function scanToResultShape(scan) {
  return {
    drGrade:     scan.grade,
    confidence:  parseFloat((scan.confidence * 100).toFixed(1)),
    patientName: scan.patient_name || 'Unknown Patient',
    patientId:   'PT-' + String(scan.id).padStart(6, '0'),
    scanDate:    scan.uploaded_at
      ? new Date(scan.uploaded_at).toLocaleDateString('en-GB', {
          day: '2-digit', month: 'short', year: 'numeric',
        })
      : '—',
    originalUrl: scan.original_image_path,
    heatmapUrl:  scan.heatmap_path,
    lesionProbs: {
      MA: Math.round((scan.lesions?.MA ?? scan.prob_ma ?? 0) * 100),
      HE: Math.round((scan.lesions?.HE ?? scan.prob_he ?? 0) * 100),
      EX: Math.round((scan.lesions?.EX ?? scan.prob_ex ?? 0) * 100),
      SE: Math.round((scan.lesions?.SE ?? scan.prob_se ?? 0) * 100),
    },
    aiInsight: GRADE_DESCRIPTIONS[scan.grade] || '',
    segOverlayUrl: scan.seg_overlay_url,
    segMasks: scan.seg_masks,
  }
}

// ─── uploadScan / startAnalysis — wired to /api/predict ──────────────────────
// NewScanPage calls uploadScan(file) then startAnalysis(scanId).
// We run the full prediction in uploadScan and cache the result so
// getScanResults can find it even though the page navigates to /scan/1/results.

export async function uploadScan(file) {
  const result = await predictScan(file, '', '')
  localStorage.setItem('last_scan_id', String(result.scan_id))
  return {
    scanId:      result.scan_id,
    previewUrl:  URL.createObjectURL(file),
    confidence:  result.confidence,   // 0–1 float
    grade_label: result.grade_label,  // e.g. "No DR"
  }
}

export async function startAnalysis(scanId) {
  return { jobId: 'done', resultId: scanId }
}

// ─── getScanResults — wired to /api/scans/:id ────────────────────────────────

export async function getScanResults(scanId) {
  const scan = await getScan(scanId)
  return scanToResultShape(scan)
}

// ─── getPatients — wired to /api/scans/ ─────────────────────────────────────
// PatientHistoryPage calls getPatients({ search }) and expects the patient-list
// shape. We fetch real scans and transform them.

export async function getPatients({ search = '' } = {}) {
  try {
    const scans = await getScans()
    const patients = scans
      .filter((s) =>
        !search ||
        (s.patient_name || '').toLowerCase().includes(search.toLowerCase()),
      )
      .map((s) => ({
        id:        s.id,
        name:      s.patient_name || 'Unknown Patient',
        patientId: 'PT-' + String(s.id).padStart(6, '0'),
        scanDate:  s.uploaded_at
          ? new Date(s.uploaded_at).toLocaleDateString('en-GB', {
              day: '2-digit', month: 'short', year: 'numeric',
            })
          : '—',
        drGrade:    s.grade,
        confidence: parseFloat((s.confidence * 100).toFixed(1)),
        previewUrl: null,
      }))
    return { patients, total: patients.length }
  } catch (_) {
    const filtered = MOCK_PATIENTS.filter(
      (p) =>
        !search ||
        p.name.toLowerCase().includes(search.toLowerCase()) ||
        p.patientId.toLowerCase().includes(search.toLowerCase()),
    )
    return { patients: filtered, total: 1284 }
  }
}

// ─── Dashboard (mock — no backend endpoint) ───────────────────────────────────

export async function getDashboard() {
  return { stats: MOCK_STATS, activity: MOCK_ACTIVITY, clinician: MOCK_CLINICIAN }
}

// ─── Settings (mock) ──────────────────────────────────────────────────────────

export async function saveSettings(settingsData) {
  return { message: 'Settings saved successfully' }
}
