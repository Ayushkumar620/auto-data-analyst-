# Frontend Status — Auto Data Analyst Platform

**Status Date:** 2026-08-25  
**Current Milestone:** Frontend Phase 5 (Autonomous Forecasting & What-If Scenario Analysis) — COMPLETE  
**Build Status:** PASSING (`tsc` + `vite build` clean, `vitest` passing 7/7, `pytest` backend forecasting passing 19/19)

---

## 1. Application Architecture & Data Flow

The platform provides an end-to-end autonomous predictive analytics pipeline:

```
[Dataset Workspace] ──> [Model Registry] ──> [Forecasting & Scenarios] ──> [Conversational Analyst]
  - Datasets (/datasets)    - Models (/models)     - Time-Series Forecasts     - Multi-turn Agent (/analyst)
  - Details (/datasets/:id) - Details (/models/:id)- What-If Simulation        - Verifiable Evidence
  - Workspaces              - Live Inference       - Uncertainty Intervals     - Cross-domain Linking
```

---

## 2. Active Routes (Phase 1, 2, 3, 4 & 5)

| Route | View Component | Status | Description |
|---|---|---|---|
| `/overview` | `OverviewPage` | **Active** | Landing page with KPIs, project lists, and quick actions. |
| `/analyst` | `AnalystPage` | **Active** | Full Conversational AI Data Analyst workspace with multi-turn reasoning and active dataset context. |
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
| `/monitoring` | `ComingSoonPage` | **Scaffolded** | Planned for Phase 6 (Data Drift & Performance Monitoring). |
| `/reports` | `ComingSoonPage` | **Scaffolded** | Planned for Phase 7 (Executive PDF & Slide Deck Generator). |
| `/settings` | `ComingSoonPage` | **Scaffolded** | Planned for Phase 8 (Platform Settings & Integration). |

---

## 3. Phase 5: Forecasting & What-If Architecture

- **Backend APIs (`/api/v1/forecast`)**:
  - `POST /api/v1/forecast/run`: Executes autonomous candidate model benchmarking (Exponential Smoothing, AutoRegressive ML, Holt-Winters, Seasonal Naive) and generates forecast trajectories with shaded probabilistic prediction intervals.
  - `POST /api/v1/forecast/whatif`: Runs deterministic counterfactual simulations with mathematical delta calculations and epistemic non-causal attribution safeguards.
- **Frontend Services & Components**:
  - `forecastService.ts`: Added `runForecast` and `runWhatIfScenario`.
  - `ForecastsPage.tsx`: Dual-tab predictive intelligence workspace.
  - `ForecastBuilder.tsx`: Dynamic target column, optional date column, horizon slider (1–24), confidence intervals (80%, 90%, 95%).
  - `ForecastSummary.tsx`: Summary strip with target metric, model algorithm, delta trajectory, and interval level.
  - `ForecastResultView.tsx`: Plotly forecast chart with historical actuals, projected trajectory, and confidence interval shaded band, alongside predictions table and validation metrics.
  - `ScenarioBuilder.tsx`: Counterfactual simulation builder with quick presets (Aggressive, Moderate, Contraction, Downside) and custom shift slider.
  - `ScenarioResultView.tsx`: Metric tiles, comparison bar chart (Baseline vs Scenario), assumptions, and epistemic disclaimer.
  - AI Analyst cross-linking: "Ask Analyst about this forecast/scenario" actions seamlessly transfer context.

---

## 4. Build & Test Verification

```bash
> tsc && vite build
✓ 103 modules transformed.
dist/index.html                     0.47 kB │ gzip:     0.31 kB
dist/assets/index-CLBCHm_N.css     33.35 kB │ gzip:     6.93 kB
dist/assets/index-D5y88IZp.js   5,004.90 kB │ gzip: 1,509.18 kB
✓ built in 21.15s

> vitest run
 Test Files  3 passed (3)
      Tests  7 passed (7)

> pytest test_milestone5_task3_forecasting_and_whatif.py
======================== 19 passed, 1 warning in 4.22s ========================
```
- **TypeScript**: 0 errors.
- **Frontend Build**: Succeeded in 21.15s.
- **Frontend Unit Tests**: 7/7 passed.
- **Backend Tests**: 19/19 passed (100%).
