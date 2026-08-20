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