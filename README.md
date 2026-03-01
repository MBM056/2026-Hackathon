# PyroPlan (2026 Hackathon)

PyroPlan is a web-based evacuation simulator for building fire scenarios.
Users can create or upload floor plans, run evacuation simulations, and compare outcomes across multiple scenarios.

## Features

- Map Editor (paint semantic layers)
- Simulation Runner (single run)
- Scenario Compare (batch runs, ranked by survival)
- Cloud video output (Google Cloud Storage signed URLs)

## Project Structure

- `main.py` - core simulation loop
- `api.py` - FastAPI backend (`/run`, `/run-upload`, `/run-batch-upload`)
- `sim/` - agents, fire, routing, renderer, map loading
- `public/` - Firebase-hosted frontend pages

## Requirements

- Python 3.11+ (or compatible with your environment)
- `pip install -r requirements.txt`
- `ffmpeg` support via `imageio[ffmpeg]` (if not already included in requirements)

## Quick Start (Local)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run backend API

```bash
uvicorn api:app --host 0.0.0.0 --port 8080
```

Health check:

```bash
curl http://localhost:8080/health
```

### 3. Run simulator from CLI (optional)

Random fire start:

```bash
python main.py --map floor1.png --out evac.mp4 --fire random
```

Fixed fire start (`row col`):

```bash
python main.py --map floor1.png --out evac.mp4 --fire 70 45
```

## Workflow

### A) Build a map

1. Open `public/map-editor.html`.
2. Set grid size.
3. Paint layers:
   - `walkable`
   - `walls`
   - `exits` (required)
   - `doors` (optional)
   - `stairs` (optional / WIP)
4. Export semantic JSON (recommended) or PNG.

Important:
- Fire coordinates use `row,col` in the web form.
- PNG input requires visible red exits.

### B) Run a single simulation

1. Open `public/simulation.html`.
2. Upload PNG or semantic JSON map.
3. Set parameters (`people`, `steps`, `alarm_at`, `awareness_radius`, `fire`).
4. Click `Run Simulation`.
5. Open returned video link.

### C) Compare scenarios

1. In Simulation page, edit `Scenario Compare JSON`.
2. Click `Run Scenario Compare`.
3. Review table sorted by survival rate.
4. Open each scenario video.

Example:

```json
[
  {"name":"Early Alarm","alarm_at":5,"people":50,"awareness_radius":12,"fire":"random"},
  {"name":"Default","alarm_at":10,"people":50,"awareness_radius":10,"fire":"random"},
  {"name":"Late Alarm","alarm_at":20,"people":50,"awareness_radius":8,"fire":"random"}
]
```

## Deployment

### Backend (Google Cloud Run)

```bash
gcloud run deploy evac-api --source . --region us-east1 --allow-unauthenticated
```

### Frontend (Firebase Hosting)

```bash
firebase deploy --only hosting
```

## Environment Notes

Typical Cloud Run environment variables:

- `OUTPUT_BUCKET`
- `ALLOWED_ORIGINS`
- `REQUIRE_FIREBASE_AUTH`

You can manage these with `gcloud run services update ... --env-vars-file env.yaml`.

## Common Issues

- `Failed to fetch` in frontend:
  - Check Cloud Run URL is correct.
  - Check `ALLOWED_ORIGINS` includes your Firebase domain.
  - Confirm backend revision is up to date.

- CORS preflight errors:
  - Re-check `ALLOWED_ORIGINS` formatting.
  - Redeploy Cloud Run after env var changes.

- `No exits detected`:
  - For PNG maps, ensure exits are red.
  - For JSON maps, ensure `exits` list is not empty.

## Legal

- See `license.txt` for licensing.
- See `legal_disclaimer.txt` for legal disclaimer.
