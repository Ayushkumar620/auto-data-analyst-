# 🤖 Auto Data Analyst Platform

A full-stack, enterprise-grade autonomous data analysis, predictive modeling, and MLOps observability platform. The platform converts raw tabular and unstructured datasets into verifiable, evidence-grounded insights, machine learning models, probabilistic time-series forecasts, counterfactual What-If scenarios, data drift audits, and executive PDF deliverables through multi-agent AI orchestration.

---

## 🌟 Key Workspaces & Capabilities

- 🤖 **Conversational AI Analyst (`/analyst`)**: Multi-turn dialogue with pronoun/anaphoric context resolution, deterministic statistical computing, and verifiable provenance tracking (`ClaimType`, confidence score, source column).
- 💾 **Dataset Workspaces (`/datasets`)**: Upload, paginated data preview, interactive schema explorer with type inference, and statistical data quality scoring.
- 🧠 **Model Registry & Leaderboard (`/models`)**: Model candidate benchmarking (Linear, Random Forest, Gradient Boosting, Multi-Layer Perceptron), loss curves, feature importances, and interactive real-time inference testing.
- 📈 **Forecasting & What-If Simulations (`/forecasts`)**: Autonomous time-series candidate model selection (Exponential Smoothing, AutoRegressive ML, Holt-Winters, Seasonal Naive) with shaded probabilistic prediction intervals and counterfactual perturbation modeling with non-causal attribution safeguards.
- 🛡️ **Model Monitoring & Data Drift (`/monitoring`)**: Production MLOps observability using 2-sample Kolmogorov-Smirnov tests, Chi-Square homogeneity tests, Population Stability Index (PSI), and ground-truth performance degradation tracking.
- 📄 **Executive Reports & Decision Deliverables (`/reports`)**: Structured analytical deliverables with executive summary narratives, KPI metric strips, statistical insight cards, verifiable evidence provenance, and multi-page ReportLab PDF export.
- 📁 **Project & Workspace Collaboration (`/projects`, `/workspaces`)**: Organize datasets, models, and analytical runs across collaborative projects.

---

## 🏗️ Architecture & Technology Stack

```
                          AUTO DATA ANALYST PLATFORM
                                       │
                         Global Dataset & Auth Context
                                       │
       ┌────────────────┬──────────────┼────────────────┬────────────────┐
       ▼                ▼              ▼                ▼                ▼
   AI Analyst     Data Workspace    Model Registry   Forecasting     ML Monitoring
  (/analyst)       (/datasets)       (/models)       (/forecasts)    (/monitoring)
       │                │              │                │                │
       └────────────────┴──────────────┼────────────────┴────────────────┘
                                       ▼
                           Executive PDF Reports
                                (/reports)
```

### Frontend Architecture
- **Framework**: React 18 + TypeScript + Vite 5
- **Routing**: React Router DOM v6 with route-level Error Boundaries
- **Design System**: Responsive glassmorphism (Plus Jakarta Sans, Space Grotesk, JetBrains Mono) with accessible contrast, focus states, and `prefers-reduced-motion` support
- **Visualizations**: Plotly.js (`PlotlyChart`) with responsive resizing and interactive tooltips
- **Testing**: Vitest unit testing suite

### Backend Architecture
- **Framework**: FastAPI (Python 3.11 / 3.12)
- **Agent Engines**: Multi-agent autonomous pipelines (`ConversationalAnalystAgent`, `AutonomousForecasterAgent`, `ModelMonitorAgent`, `EDAOrchestrator`, `InsightEngine`)
- **Machine Learning**: `scikit-learn`, `numpy`, `pandas`, `scipy`
- **Deliverables Engine**: `ReportLab` executive PDF generator

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.11+
- Node.js 18+ & npm

### 1. Backend Server Setup
```bash
# Create & activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start backend server on http://localhost:8000
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend Application Setup
```bash
cd frontend

# Install dependencies
npm install

# Start Vite development server on http://localhost:5173
npm run dev
```

---

## 🧪 Verification & Test Suites

```bash
# Frontend TypeScript Compilation & Unit Tests
cd frontend
npm run build
npx vitest run

# Backend Pytest Suites
pytest test_milestone4_task2_model_monitoring.py
pytest test_milestone5_task2_conversational_analyst.py
pytest test_milestone5_task3_forecasting_and_whatif.py
pytest test_forecast_report_api.py
```

---

## 🔒 Security & Epistemic Non-Causal Attribution Safeguards

1. **Epistemic Safeguards**: All What-If counterfactual scenario projections explicitly include non-causal simulation notices stating that model co-movements hold unperturbed features constant without claiming unverified real-world causal certainty.
2. **Provenance & Verification**: Every analytical finding returned by conversational or monitoring engines binds to traceable `Evidence` objects with mathematical test statistics (p-values, PSI, sample sizes).
3. **Authentication**: JWT authentication tokens are safely managed in `localStorage` and attached via `Authorization: Bearer <token>` headers with global 401 interceptors.
4. **Data Isolation**: Secrets, private system prompts, and model internal parameters are never leaked to client browsers.
