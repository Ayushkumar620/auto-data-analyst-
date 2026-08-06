# 🚀 ADAA — Engineering Blueprint (Step 6)

> **Goal:** This is the **foundation that every line of code will follow**.

**Project Name:** Auto Data Analyst Agent (ADAA)

---

## 1. Technology Stack

### Frontend
| Technology | Purpose |
|------------|---------|
| React | UI |
| TypeScript | Type safety |
| Tailwind CSS | Styling |
| Shadcn UI | Components |
| React Router | Navigation |
| TanStack Query | API state |
| Plotly | Charts |
| AG Grid | Large data tables |

### Backend
| Technology | Purpose |
|------------|---------|
| FastAPI | REST API |
| Python | Backend |
| LangGraph | Multi-agent orchestration |
| Pydantic | Validation |
| SQLAlchemy | ORM |
| Alembic | Database migrations |
| Celery (or Dramatiq) | Background jobs |
| Redis | Job queue & caching |

### Data Layer
| Technology | Purpose |
|------------|---------|
| Polars | Fast data processing |
| Pandas | Data manipulation |
| DuckDB | SQL over files |
| NumPy | Numerical computation |

### AI Layer
| Technology | Purpose |
|------------|---------|
| OpenAI API | Reasoning and explanations |
| LangGraph | Agent workflows |
| LangChain (optional) | Tool integration |

### Machine Learning
| Technology | Purpose |
|------------|---------|
| Scikit-learn | ML models |
| XGBoost | Gradient boosting |
| Prophet | Time-series forecasting |

### Database
- **PostgreSQL**

### File Storage
```
uploads/
reports/
exports/
```
> For cloud deployment, replace local storage with **object storage**.

---

## 2. Repository Structure

```
auto-data-analyst/
│
├── frontend/
│   ├── src/
│   ├── public/
│   └── package.json
│
├── backend/
│   ├── app/
│   ├── tests/
│   ├── alembic/
│   ├── requirements.txt
│   └── Dockerfile
│
├── docs/
│   ├── vision.md
│   ├── architecture.md
│   ├── api.md
│   └── roadmap.md
│
├── docker-compose.yml
├── .gitignore
├── README.md
└── LICENSE
```

---

## 3. Backend Architecture

```
app/
├── api/
├── agents/
├── services/
├── repositories/
├── database/
├── models/
├── schemas/
├── core/
├── utils/
├── prompts/
├── jobs/
├── reports/
└── uploads/
```

> Each folder has **one responsibility**.

---

## 4. Frontend Architecture

```
src/
├── components/
├── pages/
├── layouts/
├── hooks/
├── services/
├── contexts/
├── types/
├── utils/
└── assets/
```

---

## 5. Git Branch Strategy

> **Never work directly on `main`.**

```
main
 │
 ▼
develop
 │
 ├── feature/auth
 ├── feature/upload
 ├── feature/eda
 ├── feature/chat
 └── feature/report
```

Every new feature gets its own branch.

---

## 6. Development Workflow

```
Idea
 ↓
Issue
 ↓
Feature Branch
 ↓
Code
 ↓
Test
 ↓
Pull Request
 ↓
Review
 ↓
Merge
```

> This keeps the project organized and scalable.

---

## 7. Coding Standards

### Python
- Type hints everywhere
- Small, focused functions
- Clear docstrings
- Meaningful variable names
- Unit tests for business logic

### TypeScript
- Strict mode enabled
- Reusable components
- Avoid duplicated code
- Shared types for API models

---

## 8. Logging

Log important events:
- User login
- Dataset upload
- Analysis start/end
- AI errors
- Report generation

> Use **structured logs (JSON)** in production.

---

## 9. Environment Variables

```
DATABASE_URL
OPENAI_API_KEY
REDIS_URL
JWT_SECRET
APP_ENV
UPLOAD_PATH
```

> **Keep secrets out of source control.**

---

## 10. Development Milestones

| Sprint | Milestone |
|--------|-----------|
| **Sprint 1** | Repository setup, Authentication, Dashboard layout |
| **Sprint 2** | File upload, Dataset preview, Data profiling |
| **Sprint 3** | Data cleaning, EDA, Charts |
| **Sprint 4** | AI insights, Chat with data |
| **Sprint 5** | Forecasting, Reports |
| **Sprint 6** | Deployment, Testing, Optimization |

---

## 11. MVP Scope (First Release)

To keep the project achievable, **Version 1** should include only:
- User authentication
- CSV/Excel upload
- Automatic data profiling
- Data cleaning
- Basic EDA
- Interactive charts
- AI-generated insights
- Chat with uploaded data
- PDF report generation

> Leave advanced features like voice interaction, Google Sheets integration,
> and team collaboration for later releases.

---

## 12. Success Criteria

When **Version 1** is complete, a user should be able to:
1. **Sign in**
2. **Create a project**
3. **Upload a dataset**
4. **Wait for automatic analysis**
5. **View insights and charts**
6. **Ask questions about the data**
7. **Download a report**

> If those **seven steps work smoothly**, you have a strong MVP that can be
> demonstrated to users, recruiters, or potential customers.

---

## 🚀 Step 7 (Next)

This is where **development truly begins**. We'll create the complete project
repository from scratch, including:
- Backend (FastAPI)
- Frontend (React + TypeScript)
- Database (PostgreSQL)
- Docker configuration
- GitHub repository
- Initial folder structure
- Base authentication
- First running application

> From that point onward, every step will involve building a **real, working
> application** rather than planning.

---

*This engineering blueprint is the foundation for the ADAA codebase.*
