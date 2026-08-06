# 🏗️ Auto Data Analyst Agent (ADAA) — System Architecture

> **Tagline:** "Upload your data. Ask questions. Get business insights in minutes."

This document is the **authoritative blueprint** for the ADAA platform. It defines
the high-level architecture, multi-agent workflow, technology stack, database
schema, API design, security, scalability, and version plan.

---

## 1. High-Level Architecture

```
                    ┌──────────────────────────┐
                    │         User             │
                    └────────────┬─────────────┘
                                 │
                         React Frontend
                                 │
                     HTTPS / REST API / WebSocket
                                 │
                    ┌────────────▼────────────┐
                    │     FastAPI Backend     │
                    └────────────┬────────────┘
                                 │
                         Planner Agent
                                 │
        ┌──────────┬─────────┬─────────┬──────────┐
        │          │         │         │          │
        ▼          ▼         ▼         ▼          ▼
 File Agent   Data Agent  EDA Agent Insight   Report Agent
                           Agent      Agent
        │                              │
        └──────────────┬───────────────┘
                       ▼
                 Chat Agent
                       │
                       ▼
             Dashboard & Reports
```

---

## 2. Why a Multi-Agent System?

Instead of one AI trying to do everything, **each agent has one responsibility**.
This makes the application easier to **test, maintain, and extend**.

| Agent | Responsibility |
|-------|----------------|
| **Planner** | Understands the request and coordinates other agents |
| **File Agent** | Reads CSV, Excel, PDF, JSON, SQL |
| **Data Agent** | Cleans and validates data |
| **EDA Agent** | Creates statistics and visualizations |
| **Insight Agent** | Explains trends and anomalies |
| **Forecast Agent** | Builds predictive models |
| **Report Agent** | Generates PDF, PPT, Excel reports |
| **Chat Agent** | Answers user questions about the data |

---

## 3. Technology Stack

### Frontend
| Tech | Purpose |
|------|---------|
| **React** | UI framework |
| **TypeScript** | Type-safe development |
| **Tailwind CSS** | Styling/utility classes |
| **Shadcn UI** | Component library |
| **React Query** | Server-state management & caching |
| **Plotly** | Interactive charts/visualizations |
| **React Router** | Routing/navigation |

### Backend
| Tech | Purpose |
|------|---------|
| **FastAPI** | High-performance async REST API |
| **Python** | Primary language |
| **LangGraph** | Agent orchestration/graph workflows |
| **LangChain** | LLM integration & tooling |
| **Pandas** | Data manipulation |
| **Polars** | Fast columnar processing |
| **DuckDB** | In-process analytical SQL |

### Database
- **PostgreSQL** (production)

#### Tables
- **Users**
- **Projects**
- **Datasets**
- **Analysis Jobs**
- **Reports**
- **Chat History**

---

## 4. AI Layer — Planner Workflow

The **Planner Agent** coordinates the complete workflow from upload to report:

```
Upload CSV
    │
    ▼
Planner
    │
    ▼
Data Cleaning
    │
    ▼
EDA
    │
    ▼
Insights
    │
    ▼
Forecast
    │
    ▼
Report
```

---

## 5. Folder Structure

```
auto-data-analyst/
├── frontend/
│   └── (React + TypeScript + Tailwind + Shadcn UI)
│
└── backend/
    │
    ├── app/
    │   ├── api/
    │   ├── agents/
    │   ├── services/
    │   ├── models/
    │   ├── database/
    │   ├── schemas/
    │   ├── prompts/
    │   ├── reports/
    │   ├── uploads/
    │   ├── utils/
    │   └── main.py
    │
    ├── tests/
    ├── requirements.txt
    └── README.md
```

---

## 6. Data Flow

```
User uploads dataset
    │
    ▼
Validation
    │
    ▼
Store dataset
    │
    ▼
Planner Agent
    │
    ▼
Cleaning Agent
    │
    ▼
EDA Agent
    │
    ▼
Insight Agent
    │
    ▼
Forecast Agent
    │
    ▼
Report Agent
    │
    ▼
Dashboard
```

---

## 7. Backend Services

Each service has a **single purpose** and exposes a **clear API**, making the
system modular and easy to test/extend.

- **Upload Service**
- **Cleaning Service**
- **Visualization Service**
- **Insight Service**
- **Forecast Service**
- **Chat Service**
- **Report Service**
- **Authentication Service**

---

## 8. API Design

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload a dataset |
| GET | `/datasets` | List user's datasets |
| POST | `/analysis/start` | Start an analysis job |
| GET | `/analysis/{id}` | Get analysis status/result |
| POST | `/chat` | Chat with the data (AI Q&A) |
| GET | `/reports` | List generated reports |
| POST | `/forecast` | Run a forecast |
| POST | `/clean` | Trigger data cleaning |
| GET | `/dashboard` | Dashboard data |

---

## 9. Security

- **JWT authentication**
- **Password hashing**
- **File validation**
- **Role-based access** (Admin, Analyst, Viewer)
- **HTTPS**
- **API rate limiting**

---

## 10. Scalability

- Design each agent as an **independent service** so they can later be run
  separately if usage grows.
- **Long-running tasks** (report generation, model training) execute
  **asynchronously** to keep the application responsive.

---

## 11. Version Plan

### Version 1 (MVP)
- [x] CSV/Excel upload
- [x] Data cleaning
- [x] Basic EDA
- [x] Charts
- [x] AI insights
- [x] Chat with data
- [x] PDF report

### Version 2
- [ ] SQL database connections
- [ ] Forecasting
- [ ] PowerPoint export
- [ ] Google Sheets integration
- [ ] Dashboard customization

### Version 3
- [ ] Real-time analytics
- [ ] Team collaboration
- [ ] Scheduled reports
- [ ] Voice assistant
- [ ] Enterprise integrations

---

## 12. Milestone Completed Before Coding

- [x] Vision document
- [x] Product Requirements Document (PRD)
- [x] System architecture
- [ ] UI/UX design (next step)
- [ ] Database schema
- [ ] API specification

---

*This architecture serves as the blueprint for all development that follows.*
