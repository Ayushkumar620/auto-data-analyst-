# 🛠️ ADAA — Technical Planning (Step 4)

> **Goal:** Complete the technical planning required before building the MVP.
> This covers the database schema, API specifications, folder structure,
> development environment, GitHub repository structure, and sprint planning.

---

## 1. Database Schema (PostgreSQL)

### Tables

#### users
| Column | Type | Notes |
|--------|------|-------|
| id | UUID | PK |
| email | VARCHAR | unique, not null |
| password_hash | VARCHAR | not null |
| name | VARCHAR | |
| role | ENUM('admin','analyst','viewer') | default 'viewer' |
| avatar_url | TEXT | |
| created_at | TIMESTAMP | default now() |
| updated_at | TIMESTAMP | |

#### projects
| Column | Type | Notes |
|--------|------|-------|
| id | UUID | PK |
| user_id | UUID | FK → users.id |
| name | VARCHAR | not null |
| description | TEXT | |
| industry | VARCHAR | |
| language | VARCHAR | |
| created_at | TIMESTAMP | |
| updated_at | TIMESTAMP | |

#### datasets
| Column | Type | Notes |
|--------|------|-------|
| id | UUID | PK |
| project_id | UUID | FK → projects.id |
| name | VARCHAR | |
| file_path | TEXT | stored file location |
| file_type | VARCHAR | csv/xlsx/pdf/sql/json |
| rows | INT | |
| columns | INT | |
| size_bytes | BIGINT | |
| created_at | TIMESTAMP | |

#### analysis_jobs
| Column | Type | Notes |
|--------|------|-------|
| id | UUID | PK |
| dataset_id | UUID | FK → datasets.id |
| user_id | UUID | FK → users.id |
| type | VARCHAR | cleaning/eda/insights/forecast/report |
| status | ENUM('queued','running','completed','failed') | |
| result_json | JSONB | |
| error | TEXT | |
| started_at | TIMESTAMP | |
| completed_at | TIMESTAMP | |

#### reports
| Column | Type | Notes |
|--------|------|-------|
| id | UUID | PK |
| dataset_id | UUID | FK → datasets.id |
| user_id | UUID | FK → users.id |
| format | VARCHAR | pdf/pptx/xlsx |
| file_path | TEXT | |
| title | VARCHAR | |
| created_at | TIMESTAMP | |

#### chat_history
| Column | Type | Notes |
|--------|------|-------|
| id | UUID | PK |
| dataset_id | UUID | FK → datasets.id |
| user_id | UUID | FK → users.id |
| role | ENUM('user','assistant') | |
| content | TEXT | |
| created_at | TIMESTAMP | |

#### audit_logs
| Column | Type | Notes |
|--------|------|-------|
| id | UUID | PK |
| user_id | UUID | FK → users.id |
| action | VARCHAR | |
| entity_type | VARCHAR | |
| entity_id | UUID | |
| details | JSONB | |
| created_at | TIMESTAMP | |

---

## 2. API Specifications (FastAPI)

### Auth
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/register` | Create account |
| POST | `/auth/login` | Login (JWT) |
| POST | `/auth/google` | Google OAuth |
| POST | `/auth/github` | GitHub OAuth |
| GET | `/auth/me` | Current user |

### Projects
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/projects` | List projects |
| POST | `/projects` | Create project |
| GET | `/projects/{id}` | Get project |
| PATCH | `/projects/{id}` | Update project |
| DELETE | `/projects/{id}` | Delete project |

### Datasets
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload dataset |
| GET | `/datasets` | List datasets |
| GET | `/datasets/{id}` | Dataset overview |
| DELETE | `/datasets/{id}` | Delete dataset |

### Analysis
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/analysis/start` | Start analysis job |
| GET | `/analysis/{id}` | Get job status/result |
| POST | `/clean` | Trigger cleaning |
| POST | `/forecast` | Run forecast |

### Insights & Chat
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat` | Ask question about data |
| GET | `/insights/{dataset_id}` | Get AI insights |

### Reports
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/reports` | List reports |
| POST | `/reports/generate` | Generate report (pdf/pptx/xlsx) |
| GET | `/reports/{id}/download` | Download report |

### Dashboard
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/dashboard` | Dashboard data |

### Admin
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/admin/users` | List users |
| GET | `/admin/storage` | Storage usage |
| GET | `/admin/analytics` | Analytics |
| GET | `/admin/logs` | System logs |
| GET | `/admin/model-usage` | Model usage |

---

## 3. Folder Structure

```
auto-data-analyst/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ui/          # Shadcn UI (buttons, modals, badges...)
│   │   │   ├── charts/      # Plotly wrappers
│   │   │   ├── layout/      # Sidebar, topbar
│   │   │   └── shared/      # Cards, loaders, file upload
│   │   ├── pages/
│   │   │   ├── Landing.tsx
│   │   │   ├── Login.tsx
│   │   │   ├── Dashboard.tsx
│   │   │   ├── Project.tsx
│   │   │   ├── Upload.tsx
│   │   │   ├── Overview.tsx
│   │   │   ├── Analysis.tsx
│   │   │   ├── Charts.tsx
│   │   │   ├── Chat.tsx
│   │   │   ├── Reports.tsx
│   │   │   ├── Forecast.tsx
│   │   │   ├── Settings.tsx
│   │   │   └── Admin.tsx
│   │   ├── hooks/
│   │   ├── lib/
│   │   ├── api/             # React Query hooks
│   │   ├── store/           # state
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── tailwind.config.ts
│   ├── package.json
│   └── vite.config.ts
│
└── backend/
    ├── app/
    │   ├── api/             # FastAPI routers
    │   │   ├── auth.py
    │   │   ├── projects.py
    │   │   ├── datasets.py
    │   │   ├── analysis.py
    │   │   ├── chat.py
    │   │   ├── reports.py
    │   │   ├── dashboard.py
    │   │   └── admin.py
    │   ├── agents/          # Multi-agent system
    │   │   ├── planner.py
    │   │   ├── file_agent.py
    │   │   ├── data_agent.py
    │   │   ├── eda_agent.py
    │   │   ├── insight_agent.py
    │   │   ├── forecast_agent.py
    │   │   ├── report_agent.py
    │   │   └── chat_agent.py
    │   ├── services/
    │   ├── models/          # SQLAlchemy models
    │   ├── database/
    │   ├── schemas/         # Pydantic schemas
    │   ├── prompts/         # LLM prompts
    │   ├── reports/         # Report templates
    │   ├── uploads/         # Stored files
    │   ├── utils/
    │   └── main.py
    ├── tests/
    ├── requirements.txt
    ├── .env.example
    └── README.md
```

---

## 4. Development Environment

### Backend
```bash
# Python 3.11+
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend
```bash
cd frontend
pnpm install
pnpm dev
```

### Services
- **PostgreSQL** — primary database (Docker or local)
- **Redis** — task queue / caching (optional, for async jobs)
- **Object storage** — for uploaded files (local `uploads/` in dev)

### Environment Variables (`.env`)
```
DATABASE_URL=postgresql://user:pass@localhost:5432/adaa
JWT_SECRET=...
LLM_API_KEY=...
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
UPLOAD_DIR=app/uploads
```

---

## 5. GitHub Repository Structure

```
.github/
├── workflows/          # CI/CD (lint, test, build)
├── ISSUE_TEMPLATE/
└── PULL_REQUEST_TEMPLATE.md

frontend/               # React app
backend/                # FastAPI app
docs/
├── PRD.md
├── ARCHITECTURE.md
├── UIUX_DESIGN.md
└── TECHNICAL_PLAN.md   # this file
README.md
.gitignore
```

---

## 6. Sprint Planning

### Sprint 0 — Setup (Week 1)
- [ ] Repo structure (frontend/backend)
- [ ] Dev environment + Docker
- [ ] CI/CD skeleton
- [ ] Database migration setup (Alembic)

### Sprint 1 — Auth & Projects (Week 2)
- [ ] User registration/login (JWT)
- [ ] Google/GitHub OAuth
- [ ] Project CRUD
- [ ] Sidebar + Dashboard layout (frontend)

### Sprint 2 — Upload & Overview (Week 3)
- [ ] File upload service (CSV/Excel/PDF/SQL/JSON)
- [ ] Dataset overview (rows, cols, missing, dupes, types)
- [ ] File Agent + Data Agent (validation)

### Sprint 3 — EDA & Charts (Week 4)
- [ ] Cleaning service
- [ ] EDA Agent (stats)
- [ ] Charts (Plotly): bar, pie, line, histogram, box, scatter, heatmap
- [ ] Charts screen with tabs

### Sprint 4 — AI Insights & Chat (Week 5)
- [ ] Insight Agent (AI insights cards)
- [ ] Chat Agent (ChatGPT-style, dataset-aware)
- [ ] LangGraph orchestration
- [ ] Chat screen + streaming

### Sprint 5 — Reports & Forecast (Week 6)
- [ ] Report Agent (PDF/PPTX/XLSX)
- [ ] Forecast Agent
- [ ] Reports screen + preview/download
- [ ] Forecast screen (cards)

### Sprint 6 — Admin & Polish (Week 7)
- [ ] Admin panel
- [ ] Settings screen
- [ ] Audit logs
- [ ] Performance + responsive polish

### Sprint 7 — MVP Release (Week 8)
- [ ] End-to-end testing
- [ ] Deployment (Vercel/Railway/PostgreSQL)
- [ ] Launch MVP

---

*This technical plan is the final blueprint before MVP development begins.*
