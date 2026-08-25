# Deployment Guide — Auto Data Analyst Platform

This document outlines the deployment process, environment configurations, and operational verification procedures for the Auto Data Analyst Platform.

---

## 1. System Requirements

### Backend Services
- **Python**: 3.11+ (Tested on Python 3.12)
- **FastAPI**: 0.110+
- **Uvicorn / Gunicorn**: Production ASGI runner
- **Dependencies**: `scikit-learn`, `pandas`, `numpy`, `scipy`, `reportlab`, `plotly`, `pydantic`

### Frontend Application
- **Node.js**: 18.x or 20.x LTS
- **Package Manager**: `npm` 9+ or `pnpm`
- **Build Tool**: Vite 5.x + TypeScript 5.x
- **Static Hosting**: Nginx, Cloudflare Pages, AWS S3 + CloudFront, Vercel, or Docker container

---

## 2. Environment Variables

### Frontend Environment Variables (`frontend/.env.production`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `VITE_API_BASE_URL` | No | `""` (same origin) | Base URL for FastAPI backend (e.g., `https://api.yourdomain.com`). If empty, relative `/api/v1` routes are used. |

### Backend Environment Variables (`.env`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `ENVIRONMENT` | Yes | `production` | Deployment mode (`development`, `staging`, `production`). |
| `API_V1_STR` | No | `/api/v1` | Root API route prefix. |
| `SECRET_KEY` | Yes | `—` | Cryptographic secret key for JWT token signing. |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | No | `1440` | JWT token validity window (1 day). |
| `UPLOAD_DIR` | No | `./uploads` | Directory for temporary dataset storage. |

---

## 3. Local Development Setup

### 1. Backend Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start backend development server
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start Vite dev server
npm run dev
```

---

## 4. Production Build & Verification

### Building the Frontend
```bash
cd frontend
npm run build
```
The production bundle will be generated in `frontend/dist/`.

### Running Automated Test Suites
```bash
# Run Frontend Vitest Unit Tests
cd frontend
npx vitest run

# Run Backend Pytest Suite
pytest test_milestone4_task2_model_monitoring.py
pytest test_milestone5_task2_conversational_analyst.py
pytest test_milestone5_task3_forecasting_and_whatif.py
pytest test_forecast_report_api.py
```

---

## 5. Production Deployment Architectures

### Option A: Unified Container (Docker + Nginx + Uvicorn)
```dockerfile
# Multi-stage Dockerfile
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . ./
COPY --from=frontend-builder /app/frontend/dist /app/static

EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Option B: Decoupled Static Host (Cloudflare / S3 / Vercel)
1. Build frontend: `cd frontend && npm run build`.
2. Deploy `frontend/dist/` to your static hosting CDN.
3. Configure `VITE_API_BASE_URL=https://api.yourdomain.com` in build settings.
4. Ensure CORS in `backend/app/main.py` permits the static domain.

---

## 6. Security & Hardening Checklist

- [x] JWT authentication tokens stored in `localStorage` and attached via `Authorization: Bearer <token>`
- [x] Epistemic non-causal attribution disclaimers attached to counterfactual What-If scenarios
- [x] Verified statistical hypothesis tests (KS, Chi-Square, PSI) for data drift
- [x] No private chain-of-thought, system prompts, or model secrets exposed to the browser
- [x] Strict TypeScript typing across all API payloads and view layers
- [x] Global Error Boundary preventing white-screen crashes on uncaught runtime errors

---

## 7. Troubleshooting

| Issue | Potential Cause | Resolution |
|---|---|---|
| `401 Unauthorized` | Expired or missing token | User is automatically logged out; re-authenticate via `/login`. |
| `Dataset records required` | No dataset selected | Upload a dataset or select an active dataset from `/datasets`. |
| `PDF Compilation Error` | Missing ReportLab dependency | Ensure `reportlab` is installed in the Python environment. |

