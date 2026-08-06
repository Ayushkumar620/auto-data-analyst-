# 🚀 ADAA — Engineering Setup (Step 6)

> **Goal:** Transition from planning to engineering. Define the complete
> technology stack, repository structure, and development workflow, then begin
> building the MVP module by module.

---

## 1. Monorepo Layout

```
auto-data-analyst/
├── .github/
│   ├── workflows/          # CI/CD pipelines
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
│
├── apps/
│   ├── frontend/           # React + TypeScript web app
│   └── backend/            # FastAPI API server
│
├── packages/
│   ├── shared/             # Shared types, schemas, utils
│   └── config/             # Shared config (ESLint, Prettier, tsconfig)
│
├── docs/                   # PRD, Architecture, SDD, agent specs, etc.
│
├── scripts/                # Dev/build/deploy scripts
│
├── docker-compose.yml      # Local dev services
├── package.json            # Root (workspaces)
├── pnpm-workspace.yaml
├── .gitignore
└── README.md
```

---

## 2. Backend Service Architecture

```
backend/
│
├── app/
│   ├── api/               # FastAPI routers (auth, projects, datasets, analysis, chat, reports)
│   ├── agents/            # AI multi-agent system (planner, data, eda, insight, forecast, report, chat)
│   ├── auth/              # JWT, OAuth, password hashing
│   ├── database/          # SQLAlchemy engine, sessions, migrations (Alembic)
│   ├── models/            # ORM models (User, Project, Dataset, Analysis, Report, ChatHistory)
│   ├── schemas/           # Pydantic request/response schemas
│   ├── services/          # Business logic (upload, cleaning, visualization, report)
│   ├── prompts/           # LLM prompt templates per agent
│   ├── reports/           # Report template assets
│   ├── uploads/           # Stored dataset files
│   ├── utils/             # Helpers, error handlers
│   └── main.py            # FastAPI app entrypoint
│
├── tests/                 # Unit + integration tests
├── alembic/               # DB migration scripts
├── requirements.txt
├── .env.example
└── Dockerfile
```

---

## 3. Frontend Architecture

```
frontend/
│
├── src/
│   ├── components/        # Reusable UI (buttons, cards, modals, charts)
│   ├── pages/             # Route-level screens (Landing, Login, Dashboard, Upload, Chat...)
│   ├── hooks/             # Custom React hooks
│   ├── services/          # API client calls
│   ├── layouts/           # Sidebar + topbar layout
│   ├── routes/            # Router configuration
│   ├── context/           # React context (auth, theme, project)
│   ├── assets/            # Static assets
│   │
│   ├── App.tsx
│   └── main.tsx
│
├── tailwind.config.ts
├── package.json
└── Dockerfile
```

---

## 4. Database Migrations

- **Tool:** Alembic (SQLAlchemy)
- **Workflow:** Each schema change is a new migration file; run `alembic upgrade head`
- **Migration folders:** `backend/alembic/versions/`
- **Convention:** `alembic revision -m "add users table"`

---

## 5. Docker Setup

### `docker-compose.yml` (local dev)
```yaml
version: "3.9"
services:
  backend:
    build: ./apps/backend
    ports: ["8000:8000"]
    env_file: .env
    depends_on:
      - db
      - redis
  frontend:
    build: ./apps/frontend
    ports: ["3000:3000"]
  db:
    image: postgres:16
    environment:
      POSTGRES_USER: adaa
      POSTGRES_PASSWORD: adaa
      POSTGRES_DB: adaa
    volumes:
      - pgdata:/var/lib/postgresql/data
  redis:
    image: redis:7
volumes:
  pgdata:
```

---

## 6. CI/CD Pipeline

### Backend (`.github/workflows/backend.yml`)
- Trigger: push/PR to `main`
- Steps: checkout → setup Python → install deps → **lint** (ruff) → **test** (pytest) → **build** Docker image → push to registry → deploy

### Frontend (`.github/workflows/frontend.yml`)
- Trigger: push/PR to `main`
- Steps: checkout → setup Node → install (pnpm) → **lint** (ESLint) → **type-check** (tsc) → **build** (vite) → deploy

---

## 7. Development Workflow

- **Branching:** feature branches (`feat/`, `fix/`) → PR → review → merge to `main`
- **Commits:** Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`)
- **Code review:** every PR requires at least 1 approval
- **Pre-commit:** lint + format + type-check

---

## 8. Coding Standards

### Python
- Type hints on all functions
- `ruff` for linting/formatting
- `pytest` for tests
- Small, focused functions & modules

### TypeScript
- TypeScript strict mode
- ESLint + Prettier
- React Query for server state
- Reusable components in `components/`

### General
- OpenAPI/Swagger docs for all APIs
- Keep functions small and single-purpose
- Write unit tests for core logic

---

## 9. Sprint Plan

| Sprint | Focus |
|--------|-------|
| **Sprint 0** | Repo scaffolding, Docker, CI/CD, DB migrations |
| **Sprint 1** | Auth (register/login/JWT/OAuth) + Projects CRUD + Dashboard layout |
| **Sprint 2** | Upload service + Dataset overview |
| **Sprint 3** | Cleaning + EDA + Charts (Plotly) |
| **Sprint 4** | Insight Agent + Chat Agent (dataset-aware) |
| **Sprint 5** | Report Agent (PDF/PPT/Excel) + Forecast Agent |
| **Sprint 6** | Admin panel + Settings + Audit logs + polish |
| **Sprint 7** | E2E testing + deployment + MVP launch |

---

## 10. Development Environment (Quick Start)

### Backend
```bash
cd apps/backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload
```

### Frontend
```bash
cd apps/frontend
pnpm install
pnpm dev
```

### Environment Variables (`.env`)
```
DATABASE_URL=postgresql://adaa:adaa@localhost:5432/adaa
JWT_SECRET=...
LLM_API_KEY=...
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
UPLOAD_DIR=app/uploads
```

---

## 11. GitHub Repository Setup

1. Create `frontend/` and `backend/` scaffolds
2. Add `.github/workflows/` CI/CD
3. Add Docker + docker-compose
4. Set up branches/protection on `main`
5. Add issue/PR templates
6. Begin **Sprint 0** → **build MVP module by module**

---

*This engineering setup is the final plan before MVP development begins.*
