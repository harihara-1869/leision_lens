export const DR_GRADE_LABELS = {
  0: { label: 'No DR', severity: 'None' },
  1: { label: 'Mild NPDR', severity: 'Early' },
  2: { label: 'Moderate NPDR', severity: 'Moderate' },
  3: { label: 'Severe NPDR', severity: 'Advanced' },
  4: { label: 'Proliferative DR', severity: 'Critical' },
}

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

