# 🤖 ADAA — AI Agent Specifications

> **Why this matters:** Before writing any code, we define the detailed
> responsibilities, inputs/outputs, tools, prompts, decision logic, error
> handling, and inter-agent communication for every agent. This gives each
> agent a **clear contract**, making implementation straightforward.

---

## Agent Overview

| Agent | Responsibility |
|-------|----------------|
| **Planner** | Understands the request & coordinates other agents |
| **Data Ingestion** | Reads CSV, Excel, PDF, JSON, SQL |
| **Data Cleaning** | Cleans & validates data |
| **EDA** | Produces statistics & distributions |
| **Visualization** | Creates charts/graphs |
| **Insight** | Explains trends & anomalies |
| **Forecasting** | Builds predictive models |
| **Report** | Generates PDF/PPT/Excel reports |
| **Chat** | Answers user questions about the data |

---

## 1. Planner Agent

- **Inputs:** User request, dataset context, available agents metadata
- **Outputs:** Ordered list of agent tasks (a workflow plan)
- **Tools:** Agent registry, routing rules, LLM (for complex requests)

**Prompt template:**
```
You are the Planner Agent. Given the user request and dataset summary, decide
which specialized agents to call and in what order. Return a JSON workflow:
{"tasks": [{"agent": "...", "params": {...}}]}
```

**Decision logic:**
- Simple metadata request → EDA / Data Ingestion
- "chart/visualize" → Visualization
- "predict/forecast" → Forecasting
- "insights/anomalies" → Insight
- "report/pdf/ppt" → Report
- "why/what/how" question → Insight → Chat
- Full analysis → Clean → EDA → Visualize → Insight → Report

**Error handling:** If no agent can satisfy the request, return a clear
"unsupported request" message with suggested alternatives.

**Inter-agent communication:** Sends task specs to agents; collects and merges
their outputs into a final result.

---

## 2. Data Ingestion Agent

- **Inputs:** Raw file (CSV/Excel/PDF/JSON/SQL) or DB connection
- **Outputs:** A validated DataFrame (or dict of DataFrames) + metadata
- **Tools:** pandas, openpyxl, PyPDF2, sqlite3, SQLAlchemy

**Logic:**
1. Detect file type by extension
2. Read into a DataFrame
3. Infer schema (column names, dtypes)
4. Return dataset + metadata (rows, columns, size)

**Error handling:** invalid format, corrupted file, empty dataset → return
error with code + suggested next step.

**Inter-agent:** Passes the DataFrame to Data Cleaning / EDA.

---

## 3. Data Cleaning Agent

- **Inputs:** Raw DataFrame
- **Outputs:** Cleaned DataFrame + a cleaning report
- **Tools:** pandas, numpy

**Logic:**
1. Drop/flag duplicates
2. Handle missing values (fill or drop)
3. Fix data types
4. Detect & handle outliers
5. Standardize column names

**Error handling:** If a column becomes empty after cleaning, warn the user
instead of failing silently.

**Inter-agent:** Passes cleaned data to EDA / Insight.

---

## 4. EDA Agent

- **Inputs:** Cleaned DataFrame
- **Outputs:** Statistical summary, distributions, correlations
- **Tools:** pandas, numpy, scipy

**Logic:**
- Shape, dtypes, describe()
- Missing values, duplicates
- Correlation matrix (numeric)
- Distribution stats (mean, median, quartiles, skew)

**Error handling:** no numeric columns → return categorical-only summary.

**Inter-agent:** Feeds stats to Visualization and Insight agents.

---

## 5. Visualization Agent

- **Inputs:** DataFrame + chart spec (type, x, y)
- **Outputs:** Chart images / interactive Plotly figures
- **Tools:** matplotlib, plotly, seaborn

**Chart types:** bar, pie, line, heatmap, histogram, box plot, scatter.

**Decision logic:** auto-picks chart type if not specified (numeric vs
categorical, etc.).

**Error handling:** missing columns / non-numeric for numeric chart → clear error.

**Inter-agent:** Receives chart selection from EDA/Planner; returns figures.

---

## 6. Insight Agent

- **Inputs:** DataFrame + optional EDA output
- **Outputs:** Natural-language findings, anomalies, recommendations
- **Tools:** pandas, numpy, LLM (optional), rule-based heuristics

**Logic:**
- Detect trends, top drivers, correlations
- Flag anomalies/outliers
- Generate findings + recommendations

**Error handling:** insufficient data → return "not enough data" message.

**Inter-agent:** Uses EDA output; sends findings to Report & Chat.

---

## 7. Forecasting Agent

- **Inputs:** DataFrame + target column + horizons
- **Outputs:** Forecast values + confidence + metric score
- **Tools:** scikit-learn, statsmodels, pandas

**Logic:**
- Detect time-series column
- Train regression model (or auto-select)
- Forecast future periods
- Return trend, slope, projected change

**Error handling:** no numeric/target column → clear error.

**Inter-agent:** Receives cleaned data; sends forecast to Report & Chat.

---

## 8. Report Agent

- **Inputs:** Analysis outputs (insights, stats, charts, forecasts)
- **Outputs:** PDF / PPT / Excel report file
- **Tools:** reportlab, python-pptx, openpyxl

**Logic:**
- Assemble sections (overview, metrics, findings, recommendations)
- Render to requested format
- Return downloadable file

**Error handling:** missing required sections → warn + include what exists.

**Inter-agent:** Aggregates outputs from all agents into the final report.

---

## 9. Chat Agent

- **Inputs:** User question + dataset context + agent outputs
- **Outputs:** A conversational answer (may include charts/tables)
- **Tools:** LLM, RAG over dataset, current analysis results

**Prompt template:**
```
You are the ADAA Chat Agent. Answer the user's question about the dataset
using the provided context. Be concise and cite the data.
Context: {dataset_summary + recent agent outputs}
Question: {question}
```

**Decision logic:** Direct factual questions → answer directly; complex
analysis → ask Planner to run agents, then answer with results.

**Error handling:** question out of scope / no data → guide the user.

**Inter-agent:** The main gateway — routes to Planner and returns results in a
ChatGPT-style interface.

---

## Summary

Every agent has a **clear contract**:
- **Inputs** (what it receives)
- **Outputs** (what it produces)
- **Tools** (what it uses)
- **Prompt template** (how it reasons — where applicable)
- **Decision logic** (how it decides)
- **Error handling** (how it fails gracefully)
- **Communication** (how it talks to other agents)

With these specs agreed, implementation becomes a straightforward matter of
filling in each agent's contract.

---

*These agent specs are the final design step before implementation begins.*
