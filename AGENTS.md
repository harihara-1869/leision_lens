# Memory

## Project Overview
See @README.md for project overview and @package.json for available npm/pnpm commands for this project.

## Architecture: Image Serving Pipeline

### Where images are stored
All generated images live under `instance/` (project root, gitignored) with subdirectories:

| Directory | Contents | URL pattern |
|-----------|----------|-------------|
| `instance/uploads/` | Original uploaded fundus images | `/static/uploads/scan_{id}.png` |
| `instance/heatmaps/` | Grad-CAM heatmap overlays | `/static/heatmaps/scan_{id}.png` |
| `instance/segmentations/` | Lesion segmentation overlays + masks | `/static/segmentations/scan_{id}_overlay.png`<br>`/static/segmentations/scan_{id}_MA.png`<br>`/static/segmentations/scan_{id}_HE.png`<br>`/static/segmentations/scan_{id}_EX.png`<br>`/static/segmentations/scan_{id}_SE.png` |
| `instance/app.db` | SQLite database | — |

### How images are served
1. **Flask** (`backend/app.py`): `static_folder=instance_dir, static_url_path='/static'` — serves everything under `instance/` at `/static/*`
2. **Vite proxy** (`frontend/vite.config.js`): Forwards `/static` requests from dev server (port 5173) to Flask (port 5000)
3. **Flow**: Browser → `localhost:5173/static/...` → Vite proxy → `localhost:5000/static/...` → Flask reads from `instance/`

### Predict pipeline (POST /api/predict)
Order of operations in `backend/routes/predict.py`:
1. DR grade prediction (EfficientNet-B4 multi-task)
2. Save original image → `instance/uploads/scan_{id}.png`
3. Grad-CAM heatmap → `instance/heatmaps/scan_{id}.png`
4. Lesion segmentation → `instance/segmentations/scan_{id}_overlay.png` + per-lesion masks

Each step has try/except — if any fails, the others still proceed.

## What Was Fixed

### Bugs fixed
- **Missing `import cv2` in `predict.py`**: `_save_segmentation()` called `cv2.imwrite()` but `cv2` wasn't imported at module level. Caused silent segmentation failure (caught by bare except). Added `import cv2`, `import numpy as np`, `import json` at module level.
- **NewScanPage hardcoded `navigate('/scan/1/results')`** (line 89): After upload, always navigated to scan ID 1 regardless of the actual scan. Changed to `navigate(\`/scan/${resultId}/results\`)`.
- **DashboardPage hardcoded `/scan/1/results`** (line 153): "Review Scan" button always went to scan 1. Changed to `navigate(\`/scan/${item.id}/results\`)`.
- **`Scan.to_dict()` always computed seg URLs**: Even when segmentation silently failed, `to_dict()` returned hardcoded paths like `/static/segmentations/scan_{id}_overlay.png`. Fixed by storing `seg_overlay_url` and `seg_masks_json` as DB columns (null = not generated).
- **Original image never saved**: `predict.py` didn't save the original uploaded image. Added `_save_original()` + `original_image_path` column.

### Placeholder cleanup (done)
- Removed `MOCK_SCAN_RESULT` from `mockPatients.js` (was used as fallback in `ResultsDetailPage`)
- Removed `frontend/public/assets/fundus-scan.png` and `gradcam-heatmap.png`
- `scanToResultShape()` no longer falls back to mock placeholders — uses backend fields directly
- `getScanResults()` no longer has mock fallback — just calls API directly
- Patient list `previewUrl` set to `null` instead of placeholder

### Database changes
- Added columns to `scans` table via `backend/migrate_add_columns.py`:
  - `original_image_path` (VARCHAR 512)
  - `seg_overlay_url` (VARCHAR 512)
  - `seg_masks_json` (TEXT — stores `{"MA":"...","HE":"...","EX":"...","SE":"..."}`)
- DB is Supabase PostgreSQL, configured in `.env`

## What Remains Mock/Placeholder (Needs Functional Integration)

### Dashboard (`/dashboard`)
- `getDashboard()` in `api/index.js` returns `MOCK_STATS`, `MOCK_ACTIVITY`, `MOCK_CLINICIAN` — no real backend endpoint
- No backend endpoint exists for dashboard stats/activity
- **Would need**: `GET /api/dashboard` endpoint in Flask, wired to real scan data

### Settings (`/settings`)
- `saveSettings()` in `api/index.js` is a stub returning `{ message: 'Settings saved successfully' }`
- No backend endpoint for saving user settings
- **Would need**: `PATCH /api/settings` or similar endpoint

### Patient History (`/patients`)
- `getPatients()` falls back to `MOCK_PATIENTS` when API fails (catch block)
- Patient preview images are `null` (no real preview stored for list view)
- **Could improve**: Store a thumbnail or use `original_image_path` for list previews

### NewScanPage
- `startAnalysis()` in `api/index.js` is a stub — just returns `{ jobId: 'done', resultId: scanId }`
- It's called after `uploadScan()` but does nothing (all work is already done in `/api/predict`)
- **Should**: Remove `startAnalysis` entirely, prediction is synchronous in `/api/predict`

### General
- `MOCK_PATIENTS` in `mockPatients.js` — used as fallback
- `MOCK_STATS`, `MOCK_ACTIVITY`, `MOCK_CLINICIAN` in `mockDashboard.js` — used as mock data
- `SEVERITY_BADGE` in `mockPatients.js` — used by `SeverityBadge.jsx` component (this is fine, it's just display constants)
- `DR_GRADE_LABELS` and `LESION_DISPLAY` in `mockPatients.js` — these are display constants, not mock data

## Common Workflows

### Add new columns to scans table
```bash
cd leison-lens
python backend/migrate_add_columns.py
# Edit backend/models/scan.py to add the column
# Edit backend/routes/predict.py to populate it
# Restart Flask
```

### Test the full predict pipeline
```bash
# Start backend
cd leison-lens && PYTHONPATH="$PWD" python backend/app.py
# Upload a scan via curl
curl -X POST http://localhost:5000/api/predict \
  -H "Authorization: Bearer $(TOKEN)" \
  -F "image=@path/to/fundus.png"
# Check files
ls backend/static/uploads/
ls backend/static/heatmaps/
ls backend/static/segmentations/
```

### Debug silent failures in predict pipeline
Search Flask server stdout for `[warn]` or `Traceback` messages. Each step (save original, Grad-CAM, segmentation) is wrapped in try/except that prints warnings.

### Run frontend + backend together
```bash
./run.sh
# Frontend: localhost:5173
# Backend: localhost:5000
# Ctrl+C stops both
```

## Code Style Guidelines
- Use descriptive variable names
- Follow existing patterns in the codebase
- Extract complex conditions into meaningful boolean variables
- No placeholder images — if backend doesn't return a URL, show nothing
- No mock fallbacks in data fetching — just call the API directly
