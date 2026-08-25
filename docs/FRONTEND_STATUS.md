# Frontend Status — Auto Data Analyst Platform

**Status Date:** 2026-08-25  
**Current Milestone:** Frontend Phase 3 (Model Registry & Leaderboard UI) — COMPLETE  
**Build Status:** PASSING (`tsc` + `vite build` clean, `vitest` passing 7/7)

---

## 1. Application Architecture & Navigation

The platform integrates data management, model evaluation, and autonomous AI reasoning within a unified glassmorphic shell:

```
[Datasets Workspace] ──> [Model Registry & Leaderboard] ──> [AI Analyst / Studio]
  - Datasets (/datasets)    - Leaderboard (/models)            - Autonomous Analysis (/analyst)
  - Details (/datasets/:id) - Model Details (/models/:id)      - History (/analyses)
  - Workspaces (/workspaces)- Live Inference Form             - Details (/analyses/:id)
```

---

## 2. Active Routes (Phase 1, 2 & 3)

| Route | View Component | Status | Description |
|---|---|---|---|
| `/overview` | `OverviewPage` | **Active** | Application landing page with real KPIs, project lists, and quick actions. |
| `/datasets` | `DatasetsPage` | **Active** | Dataset explorer with live search, sorting, delete actions, and responsive grid. |
| `/datasets/:datasetId` | `DatasetWorkspacePage` | **Active** | Detailed workspace tabs: Overview, Schema Explorer, Paginated Preview Table, Data Quality. |
| `/models` | `ModelRegistryPage` | **Active** | Model leaderboard with family/status filtering, search, and KPI strip. |
| `/models/:modelId` | `ModelDetailPage` | **Active** | Loss curves, feature importances, schema, hyperparameters, and live inference form. |
| `/analyst` | `AnalystPage` | **Active** | Context-aware AI Analyst query studio connected to `/api/v1/analyze`. |
| `/analyses` | `AnalysesPage` | **Active** | History tracker for past autonomous analyses and evidence chains. |
| `/analyses/:analysisId` | `AnalysisDetailPage` | **Active** | Executive findings synthesis, detected intent, pipeline operations, and evidence chain. |
| `/workspaces` | `WorkspacesPage` | **Active** | Collaborative environment manager and project organizer. |
| `/upload` | `UploadPage` | **Active** | Full-featured dataset upload, profiling, cleaning, EDA, and report generation. |
| `/chat` | `ChatPage` | **Active** | Autonomous Command Studio with real-time DAG execution and Plotly rendering. |
| `/projects` | `ProjectsPage` | **Active** | Project list and project creation. |
| `/projects/:projectId` | `ProjectViewPage` | **Active** | Project detail view with dataset links. |
| `/profile` | `ProfilePage` | **Active** | User session and account details. |
| `/dashboard` | `DashboardPage` | **Active** | Legacy dashboard route preserved. |
| `/forecasts` | `ComingSoonPage` | **Scaffolded** | Planned for Phase 5 (Autonomous Forecasting & What-If). |
| `/monitoring` | `ComingSoonPage` | **Scaffolded** | Planned for Phase 6 (Data Drift & Performance Monitoring). |
| `/reports` | `ComingSoonPage` | **Scaffolded** | Planned for Phase 7 (Executive PDF & Slide Deck Generator). |
| `/settings` | `ComingSoonPage` | **Scaffolded** | Planned for Phase 8 (Platform Settings & Integration). |

---

## 3. Phase 3: Model Registry Architecture

- **`modelService.ts`**:
  - `listModels(params)`: Calls `GET /api/v1/models` with query filters for family, problem type, and deployment status.
  - `getModelMetadata(modelId)`: Calls `GET /api/v1/models/{model_id}` to retrieve metrics, loss curves, feature schema, hyperparameters, and feature importances.
  - `runModelInference(modelId, data)`: Calls `POST /api/v1/models/{model_id}/predict` for real-time interactive predictions.
  - `updateModelStatus(modelId, status)`: Calls `PATCH /api/v1/models/{model_id}/status` to promote models to active or staging.
  - `deleteModel(modelId)`: Calls `DELETE /api/v1/models/{model_id}`.
- **UI Components**:
  - `ModelCard.tsx`: Grid card displaying algorithm, family badge, problem type, target column, primary metric, and status.
  - `StatusChip.tsx`: Styled badges for active (green), staging (blue), and archived (gray) lifecycles.
  - `MetricBadge.tsx`: Performance metric display formatting percentages and decimals.
  - `PlotlyChart.tsx`: Loss curve convergence plot and top-10 feature importance horizontal bar chart.

---

## 4. Build & Test Verification

```bash
> tsc && vite build
✓ 91 modules transformed.
dist/index.html                     0.47 kB │ gzip:     0.31 kB
dist/assets/index-CLBCHm_N.css     33.35 kB │ gzip:     6.93 kB
dist/assets/index-C1MAFofi.js   4,970.53 kB │ gzip: 1,501.78 kB
✓ built in 22.93s

> vitest run
 Test Files  3 passed (3)
      Tests  7 passed (7)
```
- **TypeScript**: 0 errors.
- **Vite Build**: Succeeded in 22.93s.
- **Vitest**: 7/7 tests passed.
