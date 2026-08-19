# Auto Data Analyst Agent - Implementation TODO

## Completed: Autonomous Intelligence Layer
- [x] Auto data cleaning (`agent/cleaner.py` + `clean`)
- [x] Smart insights (`insights` command)
- [x] Anomaly detection (`anomalies` command)
- [x] Time-series forecasting (`forecast` command)
- [x] Executive report + PDF download (`report` command)
- [x] Multi-agent system wired (CleaningAgent, ForecastAgent, enhanced InsightAgent)
- [x] Frontend rendering + CSS + README updates

## In Progress: Milestone 3 — AI Multi-Agent Workflow & Chat
### Steps
- [x] 1. Create `agent/planner.py` — Planner Agent that routes requests across specialized agents
- [x] 2. Add `/api/chat` endpoint in `app.py` for chat-style data Q&A (already exists in `backend/app/api/v1/chat.py`)
- [x] 3. Add Chat with Data interface to `templates/index.html` (ChatPage.tsx exists at `/chat` route)
- [ ] 4. Add chat styles to `static/css/style.css`
- [ ] 5. Test planner orchestration and chat end-to-end
- [ ] 6. Update README and TODO

## Future (roadmap)
- [ ] Dashboard builder
- [ ] SQL database connection UI
- [ ] Report scheduling
- [ ] Authentication
- [ ] Deployment (Vercel/Railway/PostgreSQL)
