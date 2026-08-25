# Frontend Status — Auto Data Analyst Platform

**Status Date:** 2026-08-25  
**Current Milestone:** Frontend Phase 6 (Model Monitoring & Data Drift) — COMPLETE  
**Build Status:** PASSING (`tsc` + `vite build` clean, `vitest` passing 7/7, `pytest` backend monitoring passing 18/18)

---

## 1. Application Architecture & Observability

The platform provides a comprehensive MLOps lifecycle from profiling to production observability:

```
[Dataset Workspace] ──> [Model Registry] ──> [Forecasting & What-If] ──> [Model Monitoring]
  - Datasets (/datasets)    - Models (/models)     - Time-Series Forecasts     - Observability (/monitoring)
  - Details (/datasets/:id) - Details (/models/:id)- What-If Simulation        - Statistical Drift (KS, PSI)
  - Workspaces              - Live Inference       - Uncertainty Intervals     - Performance Tracking
```

---

## 2. Active Routes (Phase 1, 2, 3, 4, 5 & 6)

| Route | View Component | Status | Description |
|---|---|---|---|
| `/overview` | `OverviewPage` | **Active** | Landing page with KPIs, project lists, and quick actions. |
| `/analyst` | `AnalystPage` | **Active** | Full Conversational AI Data Analyst workspace with multi-turn reasoning and active dataset context. |
| `/monitoring` | `MonitoringPage` | **Active** | MLOps observability workspace for statistical data drift, schema consistency, and performance degradation. |
| `/monitoring/:modelId` | `MonitoringDetailPage` | **Active** | Dedicated model monitoring profile view. |
| `/forecasts` | `ForecastsPage` | **Active** | Autonomous time-series forecasting with probabilistic uncertainty intervals and What-If scenario simulations. |
| `/datasets` | `DatasetsPage` | **Active** | Dataset explorer with live search, sorting, delete actions, and responsive grid. |
| `/datasets/:datasetId` | `DatasetWorkspacePage` | **Active** | Multi-tab workspace: Overview, Schema Explorer, Paginated Preview Table, Data Quality. |
| `/models` | `ModelRegistryPage` | **Active** | Model leaderboard with family/status filtering, search, and KPI strip. |
| `/models/:modelId` | `ModelDetailPage` | **Active** | Loss curves, feature importances, schema, hyperparameters, and live inference form. |
| `/analyses` | `AnalysesPage` | **Active** | History tracker for past autonomous analyses and evidence chains. |
| `/analyses/:analysisId` | `AnalysisDetailPage` | **Active** | Executive findings synthesis, detected intent, pipeline operations, and evidence chain. |
| `/workspaces` | `WorkspacesPage` | **Active** | Collaborative environment manager and project organizer. |
| `/upload` | `UploadPage` | **Active** | Full-featured dataset upload, profiling, cleaning, EDA, and report generation. |
| `/chat` | `ChatPage` | **Active** | Autonomous Command Studio with real-time DAG execution and Plotly rendering. |
| `/projects` | `ProjectsPage` | **Active** | Project list and project creation. |
| `/projects/:projectId` | `ProjectViewPage` | **Active** | Project detail view with dataset links. |
| `/profile` | `ProfilePage` | **Active** | User session and account details. |
| `/dashboard` | `DashboardPage` | **Active** | Legacy dashboard route preserved. |
| `/reports` | `ComingSoonPage` | **Scaffolded** | Planned for Phase 7 (Executive PDF & Slide Deck Generator). |
| `/settings` | `ComingSoonPage` | **Scaffolded** | Planned for Phase 8 (Platform Settings & Integration). |

---

## 3. Phase 6: Model Monitoring & Data Drift Architecture

- **Backend APIs (`/api/v1/monitoring`)**:
  - `POST /api/v1/monitoring/run`: Runs statistical hypothesis testing (2-sample Kolmogorov-Smirnov test for numeric columns, Chi-Square test of homogeneity for categorical columns, Population Stability Index (PSI), and missing rate shift detection) via `ModelMonitorAgent`.
  - `GET /api/v1/monitoring/history`: Retrieves chronological monitoring run records.
  - `GET /api/v1/monitoring/overview`: Aggregates model health counts (`Healthy`, `Warning`, `Critical`) and last run timestamps across all registered models.
- **Frontend Services & Components**:
  - `monitoringService.ts`: Added `runMonitoring`, `getMonitoringHistory`, and `getMonitoringOverview`.
  - `MonitoringPage.tsx`: Model selector, active evaluation batch indicator, "⚡ Run Monitoring" action, and KPI strip.
  - `MonitoringStatusBadge.tsx`: Multi-modal status badge (`HEALTHY`, `WARNING`, `CRITICAL`, `UNKNOWN`) combining distinct icons and text.
  - `MonitoringOverview.tsx`: High-level summary of total models, healthy models, warning models, and critical models.
  - `DriftPanel.tsx` & `DriftTable.tsx`: Statistical drift table with search, test names, p-values, thresholds, severity badges, and Plotly divergence bar chart.
  - `PerformanceMonitoringPanel.tsx`: Reference vs monitored evaluation metrics comparison table with absolute delta calculations.
  - `MonitoringHistory.tsx`: Audit history of previous monitoring evaluations with timestamp, severities, and drifted feature counts.

---

## 4. Build & Test Verification

```bash
> tsc && vite build
✓ 112 modules transformed.
dist/index.html                     0.47 kB │ gzip:     0.31 kB
dist/assets/index-CLBCHm_N.css     33.35 kB │ gzip:     6.93 kB
dist/assets/index-CpJ3Y4yD.js   5,026.11 kB │ gzip: 1,513.43 kB
✓ built in 20.77s

> vitest run
 Test Files  3 passed (3)
      Tests  7 passed (7)

> pytest test_milestone4_task2_model_monitoring.py
======================= 18 passed, 62 warnings in 5.17s =======================
```
- **TypeScript**: 0 errors.
- **Frontend Build**: Succeeded in 20.77s.
- **Frontend Unit Tests**: 7/7 passed.
- **Backend Monitoring Tests**: **18/18 passed (100%)**.
