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
2. **Vite proxy** (`frontend/vite.config.js`): Forwards `/api` and `/static` requests from dev server (port 5173) to Flask (port 5000)
3. **Flow**: Browser → `localhost:5173/static/...` → Vite proxy → `localhost:5000/static/...` → Flask reads from `instance/`

### Predict pipeline (POST /api/predict)
Order of operations in `backend/routes/predict.py`:
1. DR grade prediction (EfficientNet-B4 multi-task)
2. Save original image → `instance/uploads/scan_{id}.png`
3. Grad-CAM heatmap → `instance/heatmaps/scan_{id}.png`
4. Lesion segmentation → `instance/segmentations/scan_{id}_overlay.png` + per-lesion masks

Each step has try/except — if any fails, print `[warn]` + traceback, others still proceed.

## API Endpoints

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| POST | `/api/auth/signup` | No | Create account (name, email, password ≥8 chars) |
| POST | `/api/auth/login` | No | Login → returns JWT (2h expiry) |
| POST | `/api/predict` | JWT | Upload fundus → grade + Grad-CAM + segmentation |
| GET | `/api/scans/` | JWT | List user's scans (newest first) |
| GET | `/api/scans/<id>` | JWT | Single scan detail |
| GET | `/api/dashboard` | JWT | Stats (total, early, accuracy, backlog) + 10 recent scans |

**Auth**: JWT in `Authorization: Bearer <token>` header. Token stored in `localStorage` under `token`. `AuthContext` rehydrates on page load — `ProtectedRoute` waits for rehydration before redirecting.

## Database Schema: `scans` table

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PK | Auto-increment |
| `user_id` | INTEGER FK | Owner |
| `patient_name` | VARCHAR(256) | From NewScanPage input |
| `grade` | INTEGER | 0–4 DR grade |
| `grade_label` | VARCHAR(64) | e.g. "Moderate DR" |
| `confidence` | FLOAT | Softmax probability of predicted grade |
| `prob_ma/he/ex/se` | FLOAT | Sigmoid lesion probabilities |
| `heatmap_path` | VARCHAR(512) | `/static/heatmaps/scan_{id}.png` or null |
| `original_image_path` | VARCHAR(512) | `/static/uploads/scan_{id}.png` or null |
| `seg_overlay_url` | VARCHAR(512) | `/static/segmentations/scan_{id}_overlay.png` or null |
| `seg_masks_json` | TEXT | `{"MA":"...","HE":"...","EX":"...","SE":"..."}` or null |

New columns added via `python backend/migrate_add_columns.py`.

## Remaining Mocks / Stubs

| Area | File | Issue |
|------|------|-------|
| Settings | `frontend/src/pages/SettingsPage.jsx` | `saveSettings()` is a stub — no backend endpoint |
| NewScanPage | `frontend/src/api/index.js` | `startAnalysis()` is a no-op stub — prediction is synchronous in `/api/predict` |
| Patient History | `frontend/src/pages/PatientHistoryPage.jsx` | `total` count is computed client-side from fetched scans (no pagination endpoint) |

Display constants (`SEVERITY_BADGE`, `DR_GRADE_LABELS`, `LESION_DISPLAY`) live in `mockPatients.js` — these are not mock data, they're shared UI config.

## Common Workflows

### Start both servers
```bash
./run.sh
# Frontend: localhost:5173 | Backend: localhost:5000 | Ctrl+C stops both
```

### Start backend separately
```bash
source venv/bin/activate
PYTHONPATH="$PWD" python backend/app.py
```

### Add new columns to scans table
```bash
python backend/migrate_add_columns.py
# Edit backend/models/scan.py to add column
# Edit backend/routes/predict.py to populate it
# Restart Flask
```

### Debug silent failures
Search Flask stdout for `[warn]` or `Traceback`. Each predict step is wrapped in try/except that prints warnings.

## Code Style
- No placeholder images — if backend doesn't return a URL, show nothing
- No mock fallbacks in data fetching — call the API directly
- Follow existing patterns in the codebase
- Use descriptive variable names
