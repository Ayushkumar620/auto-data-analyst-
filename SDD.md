# 📐 ADAA — Technical Design (Software Design Document)

**Project Name:** Auto Data Analyst Agent (ADAA)

---

## 1. Overall Architecture

We will use a modern **3-layer architecture**.

```
                    Frontend (React)
                           │
                     REST API/WebSocket
                           │
                  Backend (FastAPI)
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   AI Agent Layer     Business Logic     Database Layer
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                    PostgreSQL + Files
```

---

## 2. Core Modules

Instead of one large application, we'll build **independent modules**.

- Authentication
- Projects
- Dataset Manager
- Analysis Engine
- Visualization
- AI Chat
- Reports
- Forecasting
- Notifications
- Settings

> Each module has its own **APIs** and **logic**.

---

## 3. Database Design

### Users
| Column | Type |
|--------|------|
| id | UUID |
| name | VARCHAR |
| email | VARCHAR |
| password_hash | VARCHAR |
| role | ENUM |
| created_at | TIMESTAMP |

### Projects
| Column | Type |
|--------|------|
| id | UUID |
| user_id | FK |
| project_name | VARCHAR |
| description | TEXT |
| industry | VARCHAR |
| created_at | TIMESTAMP |

### Datasets
| Column | Type |
|--------|------|
| id | UUID |
| project_id | FK |
| file_name | VARCHAR |
| file_type | VARCHAR |
| rows | INT |
| columns | INT |
| status | VARCHAR |
| uploaded_at | TIMESTAMP |

### Analysis
| Column | Type |
|--------|------|
| id | UUID |
| dataset_id | FK |
| analysis_type | VARCHAR |
| summary | TEXT |
| insights | JSONB |
| created_at | TIMESTAMP |

### Reports
| Column | Type |
|--------|------|
| id | UUID |
| project_id | FK |
| report_name | VARCHAR |
| report_type | VARCHAR |
| file_path | TEXT |
| generated_at | TIMESTAMP |

### Chat History
| Column | Type |
|--------|------|
| id | UUID |
| project_id | FK |
| question | TEXT |
| answer | TEXT |
| timestamp | TIMESTAMP |

---

## 4. Backend Folder Structure

```
backend/
│
├── app/
│   ├── api/
│   ├── agents/
│   ├── auth/
│   ├── database/
│   ├── models/
│   ├── schemas/
│   ├── services/
│   ├── prompts/
│   ├── reports/
│   ├── uploads/
│   ├── utils/
│   └── main.py
│
├── tests/
│
├── requirements.txt
│
└── README.md
```

---

## 5. Frontend Folder Structure

```
frontend/
│
├── src/
│
├── components/
│
├── pages/
│
├── hooks/
│
├── services/
│
├── layouts/
│
├── routes/
│
├── context/
│
├── assets/
│
└── App.tsx
```

---

## 6. API Design

### Authentication
| Method | Endpoint |
|--------|----------|
| POST | `/register` |
| POST | `/login` |
| POST | `/logout` |
| GET | `/profile` |

### Projects
| Method | Endpoint |
|--------|----------|
| GET | `/projects` |
| POST | `/projects` |
| DELETE | `/projects/{id}` |
| PUT | `/projects/{id}` |

### Upload
| Method | Endpoint |
|--------|----------|
| POST | `/upload` |
| GET | `/dataset/{id}` |

### Analysis
| Method | Endpoint |
|--------|----------|
| POST | `/analysis/start` |
| GET | `/analysis/result/{id}` |

### AI Chat
| Method | Endpoint |
|--------|----------|
| POST | `/chat` |
| GET | `/chat/history` |

### Reports
| Method | Endpoint |
|--------|----------|
| POST | `/report/pdf` |
| POST | `/report/ppt` |
| GET | `/reports` |

---

## 7. AI Agent Workflow

```
User Uploads Dataset
    │
    ▼
Planner Agent
    │
    ▼
File Reader
    │
    ▼
Data Cleaner
    │
    ▼
EDA
    │
    ▼
Visualization
    │
    ▼
Insight Generator
    │
    ▼
Forecast
    │
    ▼
Report Generator
    │
    ▼
Chat Agent
```

> The **Planner Agent** decides which agents to call and in what order.

---

## 8. Security

- JWT Authentication
- Password hashing (**bcrypt**)
- Input validation
- File size limits
- File type validation
- Role-based permissions
- HTTPS in production

---

## 9. Performance

- **Background jobs** for long analyses
- **Dataset caching**
- **Database indexing**
- Efficient data processing (**Polars/DuckDB** for large datasets)
- **Lazy loading** in the frontend

---

## 10. Error Handling

Examples:
- Invalid file format
- Corrupted Excel file
- Empty dataset
- Missing required columns
- AI service unavailable
- Database connection failure

Every error should return:
- **Clear message**
- **Error code**
- **Suggested next step**

---

## 11. Logging & Monitoring

Track:
- User logins
- Dataset uploads
- Analysis duration
- Failed analyses
- API errors
- AI request usage

> This helps with debugging and future improvements.

---

## 12. Development Standards

- Use **Git with feature branches**
- Write **unit tests** for core logic
- Use **type hints** in Python and TypeScript
- Keep functions **small and focused**
- Document APIs with **OpenAPI/Swagger**

---

## 13. Project Roadmap

| Phase | Goal | Status |
|-------|------|--------|
| 1 | Vision | ✅ Complete |
| 2 | Product Requirements (PRD) | ✅ Complete |
| 3 | UI/UX Design | ✅ Complete |
| 4 | Technical Design (SDD) | ✅ Complete |
| 5 | Development Environment | ⏳ Next |
| 6 | Build MVP | ⏳ |
| 7 | AI Multi-Agent System | ⏳ |
| 8 | Testing | ⏳ |
| 9 | Deployment | ⏳ |

---

*This SDD is the technical blueprint guiding implementation.*
