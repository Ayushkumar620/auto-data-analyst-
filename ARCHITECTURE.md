# 🏗️ Auto Data Analyst Agent (ADAA) — System Architecture

> **Tagline:** "Upload your data. Ask questions. Get business insights in minutes."

This document is the **blueprint** for the ADAA platform. It defines the complete
architecture from **upload to report generation**, including the frontend, backend,
AI multi-agent workflow, database schema, API structure, and folder layout.

It serves as the single source of truth for all development that follows.

---

## 1. Architecture Overview

ADAA is a **full-stack, AI-powered data analysis platform** built around a
**multi-agent orchestration core**. Users upload data or connect a database, and
the platform autonomously cleans, analyzes, visualizes, explains, forecasts, and
reports on it.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (Web)                                │
│                                                                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐   │
│  │ Upload   │  │ Chat     │  │ Dashboard│  │ Charts   │  │ Reports    │   │
│  │ (Drag/drop,│  │ (AI Q&A) │  │ (EDAs)   │  │ & Tables │  │ (PDF/PPT) │   │
│  │ DB connect)│  │          │  │          │  │          │  │            │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └────────────┘   │
└───────────────┬────────────────────────────────────────────────────────────┘
                │ HTTPS / REST (JSON)
                ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                         API GATEWAY (Backend)                              │
│                         Flask / FastAPI                                     │
│              Auth · Validation · Rate Limiting · Audit Log                  │
└───────────────┬────────────────────────────────┬────────────────────────────┘
                │                                │
                ▼                                ▼
┌──────────────────────────────┐   ┌──────────────────────────────────────────┐
│   DATA ACCESS LAYER          │   │      AI MULTI-AGENT CORE                 │
│  • File Ingest (CSV/Excel/PDF)│   │                                          │
│  • DB Connectors (SQL)        │   │  ┌────────────┐  ┌────────────┐        │
│  • Google Sheets / API        │   │  │PLANNER     │→ │ ROUTER     │        │
│  • Object Storage (S3)        │   │  │Agent       │  │(LLM/Rules) │        │
│  • Data Cache / Temp          │   │  └─────┬──────┘  └─────┬──────┘        │
└───────────────┬───────────────┘   │        ▼              ▼                │
                │                    │  ┌────────────┐  ┌────────────┐        │
                ▼                    │  │ CLEANER    │  │ EDA        │        │
┌──────────────────────────────┐    │  │ Agent      │  │ Agent      │        │
│   RELATIONAL DATABASE        │    │  └────────────┘  └────────────┘        │
│  • Users / Roles / Sessions  │    │  ┌────────────┐  ┌────────────┐        │
│  • Datasets / Analyses       │    │  │ INSIGHT    │  │ VISUALIZER │        │
│  • Analyses / Audit Logs     │    │  │ Agent      │  │ Agent      │        │
│  • Generated Reports         │    │  └────────────┘  └────────────┘        │
└──────────────────────────────┘    │  ┌────────────┐  ┌────────────┐        │
                                    │  │ ML         │  │ REPORT     │        │
                                    │  │ Agent      │  │ Agent      │        │
       ┌────────────────────────────┘  └────────────┘  └────────────┘        │
       │  POSTGRESQL / SQLite          ┌──────────────────────────────┐       │
       │                              │  LLM SERVICE (optional)      │       │
       │                              │  • NL → Task Plan            │       │
       │                              │  • Natural-language insights  │       │
       │                              └──────────────────────────────┘       │
       └──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Frontend Architecture

### 2.1 Framework
- **React** (or **Vue 3**) with **TypeScript** for a reactive, component-based UI.
- **Vite** for fast development builds.
- **State management:** Redux Toolkit / Zustand.
- **Routing:** React Router.
- **UI library:** Tailwind CSS + component library (shadcn/ui or MUI).
- **Charts:** Recharts / Chart.js / Plotly.js.
- **HTTP client:** Axios (interceptors for auth + error handling).

### 2.2 Key Pages / Components
| Module | Description |
|--------|-------------|
| **Auth** | Sign up, login, profile, role-based access |
| **Upload** | Drag-and-drop file upload; database connection wizard |
| **Chat** | Chat-with-data panel (AI Q&A over the loaded dataset) |
| **Dashboard** | EDA overview: KPIs, charts, tables, summary stats |
| **Data Table** | Interactive, sortable/filterable grid of rows |
| **Charts** | Auto-generated bar/line/scatter/histogram/pie/box |
| **Insights** | Natural-language findings, drivers, anomalies, recommendations |
| **Models** | Predict/classify/cluster/forecast results |
| **Reports** | Executive report preview + PDF/Excel/PPT download |

### 2.3 Data Communication
- All interactions go through the **API Gateway** via REST/JSON.
- The frontend holds the currently active dataset reference (file path or DB id)
  and passes it with each request.

---

## 3. Backend Architecture

### 3.1 Framework
- **Python 3.10+**
- **Flask** (current) → extensible to **FastAPI** for async + auto OpenAPI docs.
- Modular **blueprints/packages** so each concern is isolated.

### 3.2 Core Layers
| Layer | Responsibility |
|-------|----------------|
| **API / Routes** | HTTP endpoints, request validation, error handling |
| **Auth Service** | JWT/OAuth2, password hashing (bcrypt), roles |
| **Data Service** | File ingest, DB connectors, Google Sheets, API adapters |
| **Processing Service** | Cleaning, EDA, statistics |
| **AI Agent Core** | Multi-agent orchestration (Planner → specialized agents) |
| **ML Service** | Model training, evaluation, forecasting, clustering |
| **Reporting Service** | PDF/Excel/PPT generation |
| **Audit Service** | Log every analysis action per user |

---

## 4. AI Multi-Agent Workflow

The heart of ADAA. A **Planner Agent** receives a natural-language request and
orchestrates specialized agents.

```
User Request
      │
      ▼
┌─────────────────┐
│  LLM ROUTER     │  Optional LLM (if API key present)
│  NL → Task Plan │  Falls back to rule-based parser
└────────┬────────┘
         │ structured task plan {action, target, chart_type, ...}
         ▼
┌─────────────────┐
│  PLANNER AGENT  │  Routes to the right agent(s), in order
└────────┬────────┘
         │
   ┌─────┼─────┬─────────┬──────────┬───────────┐
   ▼     ▼     ▼         ▼          ▼           ▼
 CLEANER  EDA   INSIGHT  VISUALIZER  ML        REPORT
 Agent   Agent  Agent    Agent      Agent      Agent
```

### 4.1 Agent Catalog
| Agent | Role | Sample Task |
|-------|------|-------------|
| **Data Loading** | Load & validate any source | "Load uploads/sales.csv" |
| **Cleaning** | Missing values, dupes, outliers, types | "Clean the data" |
| **EDA / Analysis** | Summary, describe, nulls, correlation, head | "summary" |
| **Visualization** | Charts (bar/line/scatter/hist/pie/box) | "chart by category" |
| **Insight** | NL findings, anomalies, drivers, recommendations | "Which region has highest profit?" |
| **ML** | Regression, classification, clustering, forecasting | "Predict next quarter sales" |
| **Report** | Executive narrative + PDF/Excel/PPT | "Generate report" |

### 4.2 Orchestration Flow
1. **Router** converts the user's message into a structured task plan.
2. **Planner** picks the agent(s) and execution order.
3. Each agent runs atomically and returns `{status, output, duration_ms}`.
4. The Planner can run a **pipeline** (e.g. `clean → EDA → insights → report`).
5. Results are returned to the API layer and rendered by the frontend.

---

## 5. Database Schema

### 5.1 Users & Auth
```sql
users (
  id            UUID PRIMARY KEY,
  email         VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  full_name     VARCHAR(150),
  role          VARCHAR(50) DEFAULT 'analyst',   -- admin | analyst | viewer
  created_at    TIMESTAMP DEFAULT now(),
  updated_at    TIMESTAMP
)

user_sessions (
  id            UUID PRIMARY KEY,
  user_id       UUID REFERENCES users(id) ON DELETE CASCADE,
  token         TEXT UNIQUE,
  expires_at    TIMESTAMP,
  created_at    TIMESTAMP
)
```

### 5.2 Datasets & Analyses
```sql
datasets (
  id            UUID PRIMARY KEY,
  user_id       UUID REFERENCES users(id) ON DELETE CASCADE,
  name          VARCHAR(255),
  source_type   VARCHAR(50),   -- csv | excel | pdf | sql | sheets | api
  source_ref    TEXT,          -- file path or connection id
  row_count     INT,
  column_count  INT,
  schema_json   JSONB,         -- column name/type metadata
  created_at    TIMESTAMP DEFAULT now()
)

analyses (
  id            UUID PRIMARY KEY,
  user_id       UUID REFERENCES users(id),
  dataset_id    UUID REFERENCES datasets(id) ON DELETE CASCADE,
  request       TEXT,          -- original user command
  task_plan     JSONB,         -- parsed structured plan
  result_json   JSONB,         -- full analysis result
  status        VARCHAR(20) DEFAULT 'pending',  -- pending|running|completed|error
  duration_ms   INT,
  created_at    TIMESTAMP DEFAULT now()
)
```

### 5.3 Reports & Audit
```sql
reports (
  id            UUID PRIMARY KEY,
  user_id       UUID REFERENCES users(id),
  dataset_id    UUID REFERENCES datasets(id),
  analysis_id   UUID REFERENCES analyses(id),
  format        VARCHAR(10),   -- pdf | xlsx | pptx
  file_path     TEXT,
  created_at    TIMESTAMP DEFAULT now()
)

audit_logs (
  id            UUID PRIMARY KEY,
  user_id       UUID REFERENCES users(id),
  action        VARCHAR(100),  -- upload|analyze|chat|predict|report|login...
  dataset_id    UUID,
  details       JSONB,
  ip_address    INET,
  created_at    TIMESTAMP DEFAULT now()
)
```

> **Note:** In the MVP, SQLite is used. PostgreSQL is the production target.

---

## 6. API Structure (REST)

All endpoints are under `/api` and return JSON. Auth via `Authorization: Bearer <token>`.

### 6.1 Auth
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/signup` | Create account |
| POST | `/api/auth/login` | Login → JWT |
| GET  | `/api/auth/profile` | Current user profile |
| PUT  | `/api/auth/profile` | Update profile |

### 6.2 Data Sources
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/datasets/upload` | Upload file (CSV/Excel/PDF/...) |
| POST | `/api/datasets/connect` | Connect SQL/Sheets/API |
| GET  | `/api/datasets` | List user's datasets |
| GET  | `/api/datasets/{id}` | Dataset metadata + preview |
| DELETE | `/api/datasets/{id}` | Delete dataset |

### 6.3 Analysis & Chat
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/analyze` | Run a command on a dataset |
| POST | `/api/chat` | Chat-with-data (natural language Q&A) |
| POST | `/api/analyses/{id}/pipeline` | Run multi-agent pipeline |
| GET  | `/api/analyses` | User's analysis history |

### 6.4 ML & Reports
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/predict` | Train/run a model |
| POST | `/api/forecast` | Time-series forecast |
| POST | `/api/report` | Generate report (PDF/Excel/PPT) |
| GET  | `/api/reports/{id}/download` | Download generated report |

---

## 7. Folder Structure

```
auto-data-analyst/
├── frontend/                    # React + TypeScript (Vite)
│   ├── src/
│   │   ├── components/          # Reusable UI components
│   │   ├── pages/               # Auth, Upload, Chat, Dashboard, Reports
│   │   ├── services/            # API client (axios)
│   │   ├── store/               # Redux/Zustand state
│   │   ├── hooks/               # Custom React hooks
│   │   └── App.tsx
│   ├── public/
│   └── package.json
│
├── backend/
│   ├── app.py                   # Flask app entrypoint
│   ├── api/                     # Route blueprints
│   │   ├── auth_routes.py
│   │   ├── dataset_routes.py
│   │   ├── analysis_routes.py
│   │   └── report_routes.py
│   ├── services/
│   │   ├── auth_service.py
│   │   ├── data_service.py
│   │   ├── audit_service.py
│   │   └── report_service.py
│   ├── agents/                  # Multi-agent core
│   │   ├── planner.py
│   │   ├── router.py            # LLM/Rule router
│   │   ├── cleaner.py
│   │   ├── analyzer.py
│   │   ├── visualizer.py
│   │   ├── insights.py
│   │   ├── predictor.py
│   │   └── report_agent.py
│   ├── models/                  # DB models (SQLAlchemy)
│   ├── utils/
│   └── requirements.txt
│
├── database/                    # Migrations / seeds
├── storage/                    # Uploaded files, generated reports
├── tests/
├── docs/                       # PRD, ARCHITECTURE, etc.
├── .env.example
├── .gitignore
└── README.md
```

---

## 8. Data Flow: Upload → Report

```
1. UPLOAD
   User drags a CSV → frontend POSTs to /api/datasets/upload
   → Data Service validates & saves file to storage/
   → Loader parses into DataFrame → dataset row created in DB
   → returns {dataset_id, preview, schema}

2. CLEAN (automatic)
   Planner → CleaningAgent
   → detect types, fill/drop missing, remove dupes, flag outliers
   → returns cleaning report + cleaned data

3. EDA
   Planner → EDA/AnalysisAgent
   → summary, describe, nulls, correlation
   → returns tables + computed stats

4. VISUALIZE
   Planner → VisualizationAgent
   → auto-charts based on data types
   → returns base64 chart images

5. INSIGHTS
   Planner → InsightAgent (+ optional LLM)
   → NL findings, drivers, anomalies, recommendations
   → returns human-readable insights

6. PREDICT / FORECAST (optional)
   Planner → MLAgent
   → auto-selects model (regression/classification/clustering/forecast)
   → returns metrics + predictions

7. REPORT
   Planner → ReportAgent
   → aggregates all agent outputs
   → ReportService renders PDF/Excel/PPT
   → stored in reports table + downloadable

8. AUDIT
   Every step logged to audit_logs for traceability.
```

---

## 9. Non-Functional Requirements Mapping

| Requirement | Architecture Choice |
|-------------|---------------------|
| **Fast performance** | Async agents, caching, DB indexes, lazy chart loading |
| **Secure auth** | JWT + bcrypt, role-based access, HTTPS-only |
| **Scalable** | Stateless API, PostgreSQL, object storage, queue for heavy jobs |
| **Responsive UI** | React + Tailwind, mobile-first |
| **Reliable file handling** | MIME validation, size limits, virus/temp cleanup |
| **Audit logs** | Central `audit_logs` table + structured details |

---

## 10. Success Metrics & MVP Scope

### Success Metrics
- ⏱️ Analysis completed in **under 2 minutes** for medium datasets.
- 🎯 High accuracy of generated insights.
- 🖱️ Minimal manual intervention required.
- 👍 Positive user feedback on usability.

### MVP (First Release)
- ✅ User login
- ✅ CSV/Excel upload
- ✅ Automatic data cleaning
- ✅ Basic EDA (summary, describe, nulls, correlation)
- ✅ Charts
- ✅ AI-generated insights
- ✅ Chat with data
- ✅ PDF report download

> Everything else (DB connectors, Google Sheets, APIs, clustering, PowerPoint,
> role management, audit dashboards) is scheduled for later milestones.

---

## 11. Development Principles

1. **Build small, test often.**
2. **Keep modules independent** — each agent/service is a black box.
3. **Use APIs between components** — no tight coupling.
4. **Make the UI intuitive** — simple, guided UX.
5. **Reliability before advanced AI** — solid cleaning/EDA first, then LLM features.

---

*This architecture is the blueprint for all development that follows.*
