export const DR_GRADE_LABELS = {
  0: { label: 'No DR', severity: 'None' },
  1: { label: 'Mild NPDR', severity: 'Early' },
  2: { label: 'Moderate NPDR', severity: 'Moderate' },
  3: { label: 'Severe NPDR', severity: 'Advanced' },
  4: { label: 'Proliferative DR', severity: 'Critical' },
}

export const MOCK_PATIENTS = [
  { id: 1, name: 'Priya Sharma', patientId: 'PT-000001', scanDate: '12 May 2026', drGrade: 2, confidence: 94.2, previewUrl: null },
  { id: 2, name: 'Arun Kumar', patientId: 'PT-000002', scanDate: '12 May 2026', drGrade: 0, confidence: 98.7, previewUrl: null },
  { id: 3, name: 'Lakshmi Devi', patientId: 'PT-000003', scanDate: '11 May 2026', drGrade: 4, confidence: 96.1, previewUrl: null },
  { id: 4, name: 'Ravi Patel', patientId: 'PT-000004', scanDate: '11 May 2026', drGrade: 1, confidence: 92.8, previewUrl: null },
  { id: 5, name: 'Sunita Reddy', patientId: 'PT-000005', scanDate: '10 May 2026', drGrade: 3, confidence: 95.5, previewUrl: null },
]

export const SEVERITY_BADGE = {
  0: { bg: 'bg-success/20', text: 'text-success', label: 'No DR' },
  1: { bg: 'bg-info/20', text: 'text-info', label: 'Mild' },
  2: { bg: 'bg-warning/20', text: 'text-warning', label: 'Moderate' },
  3: { bg: 'bg-error/20', text: 'text-error', label: 'Severe' },
  4: { bg: 'bg-error/20', text: 'text-error', label: 'Proliferative' },
}

// Lesion display configuration
export const LESION_DISPLAY = {
  MA: { full: 'Microaneurysms', short: 'MA', color: '#FFD700', barColor: 'bg-yellow-500' },
  HE: { full: 'Haemorrhages', short: 'HE', color: '#FF4444', barColor: 'bg-red-500' },
  EX: { full: 'Hard Exudates', short: 'EX', color: '#4488FF', barColor: 'bg-blue-500' },
  SE: { full: 'Soft Exudates', short: 'SE', color: '#44FF44', barColor: 'bg-green-500' },
}

