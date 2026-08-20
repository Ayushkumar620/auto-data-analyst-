# Auto Data Analyst - Architecture Audit Report

## Executive Summary

The current Auto Data Analyst system works for basic conditions but has significant reliability gaps for real-world datasets and natural-language questions. This audit identifies failures across 10 critical areas and proposes a reliability architecture.

---

## 1. Intent Detection Failures

**Current Implementation:** `agent/nlp_parser.py` (NLPCommandParser) + `agent/llm_router.py` (LLMRouter)

**Problems:**
- **Hardcoded keyword lists** for metrics, actions, chart types, time periods - brittle to paraphrasing
- **Exact column name matching** in `_detect_column()` (line 208-215) - fails on `sales_amount`, `net_sales`, `revenue_amount`
- **No semantic understanding** - treats "total sales" and "sum of revenue" as different patterns
- **LLM fallback is opaque** - no confidence scoring, no structured reasoning trace
- **Time parsing is regex-based** - fails on relative dates like "last quarter", "year to date"
- **No disambiguation** - "show me sales" could mean chart, summary, or aggregation

**Evidence:** `nlp_parser.py` lines 37-75 (hardcoded keyword dictionaries), lines 208-215 (hardcoded known columns)

---

## 2. Agent Selection Failures

**Current Implementation:** `agent/planner.py` (PlannerAgent.REQUEST_MAP) + `agent/command_parser.py` (CommandParser.COMMANDS)

**Problems:**
- **Static mapping** - `REQUEST_MAP` is a fixed dictionary, no dynamic agent selection based on data characteristics
- **No agent capability awareness** - Planner doesn't know if an agent can actually handle the data (e.g., forecasting needs time series)
- **Single-action bias** - `run_agent()` executes one action; pipelines are hardcoded sequences
- **No fallback agents** - If primary agent fails, no alternative is tried
- **Agent output not standardized** - Different agents return different structures, making chaining difficult

**Evidence:** `planner.py` lines 30-115 (static REQUEST_MAP), lines 120-139 (run_agent with no fallback)
---

## 3. Tool Selection Failures

**Current Implementation:** Agents are the tools, selected via PlannerAgent

**Problems:**
- **No tool registry** - Tools (agents) are hardcoded imports
- **No tool metadata** - No description of what each tool requires/prefers (data shape, column types)
- **No tool composition** - Can't combine tools dynamically (e.g., "clean then chart")
- **Tool selection = agent selection** - No finer-grained tool concept (e.g., specific chart type, specific stat)

**Evidence:** `planner.py` lines 14-23 (hardcoded imports), lines 30-115 (static mapping)

---

## 4. Dataset Understanding Failures

**Current Implementation:** `backend/app/profilers/dataset_profiler.py` (DatasetProfiler) + `agent/analyzer.py` (DataAnalyzer)

**Problems:**
- **Structural only** - Profiles columns, types, missing values but NOT semantic meaning
- **No entity recognition** - Doesn't identify "customer", "product", "transaction" entities
- **No metric/dimension classification** - All numeric columns treated equally
- **No relationship detection** - Foreign keys, hierarchies not identified
- **No data quality semantics** - Missing % is computed but not interpreted (e.g., "critical for forecasting")
- **Single-table focus** - Multi-table relationships not modeled

**Evidence:** `dataset_profiler.py` lines 8-45 (only structural profiling), `analyzer.py` lines 21-37 (summary only)

---

## 5. Column Identification Failures

**Current Implementation:** Scattered across `nlp_parser.py`, `insights.py`, `predictor.py`, `visualizer.py`

**Problems:**
- **Exact name matching** - `nlp_parser.py` line 211: `known = ["sales", "revenue", "price", "amount", ...]`
- **Substring matching** - `insights.py` line 540-543: `if intent.column in col.lower()`
- **No semantic equivalence** - `sales`, `sales_amount`, `net_sales`, `revenue`, `turnover` treated as different
- **No confidence scoring** - Match is binary (found/not found)
- **Case sensitivity issues** - Mixed approaches (`.lower()` in some places, not others)
- **No context awareness** - "amount" in banking vs retail means different things

**Evidence:** `nlp_parser.py` line 211, `insights.py` lines 535-543, `predictor.py` lines 36-40

---

## 6. Data Processing Failures

**Current Implementation:** `agent/cleaner.py`, `backend/app/cleaning/`, `agent/analyzer.py`, `agent/visualizer.py`, `agent/predictor.py`

**Problems:**
- **Cleaning is destructive** - `DataCleaner.clean()` modifies data in place, original lost
- **No processing lineage** - Can't trace how a result was derived
- **Hardcoded cleaning rules** - Median for numeric, mode for categorical - no domain awareness
- **Outlier detection only flags** - Doesn't provide context or handling options
- **Forecasting assumes linear trend** - `predictor.py` line 66: simple linear regression on position
- **Visualization column selection is heuristic** - `visualizer.py` lines 88, 97: picks first categorical/numeric
- **No validation of processing outputs** - Cleaned data not verified against expectations

---

## 7. Result Validation Failures

**Current Implementation:** `backend/app/forecasting/validator.py` (ForecastValidator) - only for forecasting

**Problems:**
- **Validation only exists for forecasting** - No validation for analysis, insights, charts, predictions
- **No cross-checking** - Results not verified against source data
- **No schema validation** - Agent outputs not validated against expected schemas
- **Silent failures** - Invalid columns in chart requests produce empty charts instead of errors
- **No evidence linking** - Results don't reference source rows/columns used

**Evidence:** Only `validator.py` exists; no validators for other agent outputs

---

## 8. AI Explanation Failures

**Current Implementation:** `backend/app/insights/interpreter.py` (InsightInterpreter) + `backend/app/insights/rules.py` (InsightRules)

**Problems:**
- **No fact/observation/inference distinction** - All insights presented equally
- **Correlation presented as causation** - `rules.py` line 36-38: "Strong X-Y Relationship" implies causation
- **LLM can hallucinate** - `_looks_like_new_facts()` heuristic (line 142-153) is weak guard
- **No evidence citations in narrative** - LLM output not grounded to specific fact IDs
---

## 9. Error Handling Failures

**Current Implementation:** Try/catch in BaseAgent._error() and various agent run() methods

**Problems:**
- **Errors lose context** - Only error message preserved, no input state
- **No retry logic** - Transient failures (LLM timeout) not retried
- **No graceful degradation** - If one agent fails, pipeline stops
- **Error types not distinguished** - Validation error = runtime error = LLM error
- **No user-facing error explanation** - Technical errors shown to users

**Evidence:** `base.py` lines 45-56 (_error), `planner.py` lines 134-139 (run_agent catches all exceptions)

---

## 10. Agent Communication Failures

**Current Implementation:** Sequential pipeline in `PlannerAgent.run_pipeline()` + in-memory `_analyses` dict in `backend/app/api/insights.py`

**Problems:**
- **No structured message format** - Agents pass raw dicts, no schema
- **No intermediate result validation** - Pipeline passes output directly to next agent
- **In-memory only** - `_analyses` dict lost on restart, no persistence
- **No async/parallel execution** - All steps sequential
- **No feedback loops** - Later agents can't request clarification from earlier ones
- **Context not shared** - Each agent re-discovers dataset properties

**Evidence:** `planner.py` lines 141-173 (linear pipeline), `insights.py` line 13 (global dict)

---

## Proposed Reliability Architecture

```
User Request
    │
    ▼
┌─────────────────────────────────────┐
│ Intent Analyzer                     │
---

## Implementation Priority

### Phase 1: Foundation (Week 1)
1. **Standardized AgentResult schema** - All agents return consistent structure
2. **Semantic Schema Agent** - Column mapping with confidence scores
3. **Dataset Knowledge Object** - Centralized semantic understanding

### Phase 2: Validation & Reliability (Week 2)
4. **Result Validator** - Cross-check all agent outputs
5. **Retry/Repair mechanism** - Automatic recovery from common failures
6. **Error taxonomy** - Structured error types with user-friendly messages

### Phase 3: Intelligence (Week 3)
7. **Intent Analyzer v2** - Semantic parsing with confidence
8. **Task Planner v2** - Dynamic planning with fallbacks
9. **Evidence tracking** - Full lineage from question to answer

### Phase 4: Evaluation (Week 4)
10. **Evaluation framework** - Test datasets + metrics
11. **Automated tests** - Normal, messy, ambiguous, adversarial cases

---

## Files to Create/Modify

### New Core Files:
- `agent/schemas.py` - Standardized AgentResult, DatasetKnowledge, SemanticMapping
- `agent/semantic_schema_agent.py` - Semantic column understanding
- `agent/intent_analyzer.py` - Improved intent detection
- `agent/task_planner.py` - Dynamic planning with fallbacks
- `agent/result_validator.py` - Validation with repair
- `agent/dataset_knowledge.py` - Dataset Knowledge Object
- `agent/evidence.py` - Evidence tracking utilities

### Modified Files:
- `agent/base.py` - Update BaseAgent to use AgentResult
- `agent/planner.py` - Replace with TaskPlanner
- `agent/nlp_parser.py` - Enhance or replace with IntentAnalyzer
- `agent/agents.py` - Update all agents to return AgentResult
- `backend/app/insights/schemas.py` - Add FACT/OBSERVATION/CORRELATION/INFERENCE/RECOMMENDATION types

### Test Files:
- `test_reliability_framework.py` - Core reliability tests
- `test_semantic_schema.py` - Semantic mapping tests
- `test_result_validator.py` - Validation tests
- `evaluation/` - Evaluation datasets and harness
│ - Semantic parsing                  │
│ - Confidence scoring                │
│ - Ambiguity detection               │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Task Planner                        │
│ - Dynamic agent selection           │
│ - Tool capability matching          │
│ - Dependency graph                  │
│ - Fallback planning                 │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Semantic Schema Agent               │
│ - Column → semantic concept mapping │
│ - Confidence scores                 │
│ - Dataset Knowledge Object          │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Agent Execution (with Evidence)     │
│ - Standardized AgentResult schema   │
│ - Evidence tracking                 │
│ - Confidence propagation            │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Result Validator                    │
│ - Schema validation                 │
│ - Data cross-checking               │
│ - Evidence verification             │
│ - Repair/ retry logic               │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Evidence-based Response             │
│ - FACT / OBSERVATION / CORRELATION  │
│   / INFERENCE / RECOMMENDATION      │
│ - Confidence intervals              │
│ - Source tracing                    │
└─────────────────────────────────────┘
```
- **Confidence only 3 levels** - high/medium/low, no numerical scores
- **Recommendations not traced to evidence** - `rules.py` line 55-74 generates recommendations without explicit links

**Evidence:** `interpreter.py` lines 142-153 (weak hallucination guard), `rules.py` lines 36-38 (correlation language)
**Evidence:** `cleaner.py` lines 90-111 (hardcoded fill strategies), `predictor.py` lines 65-75 (simple linear regression)