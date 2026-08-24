"""Dynamic Context Assembly Engine for LLM Prompt Orchestration.

Compresses and budgets dataset metadata, semantic taxonomy, analytical goals,
validation state, and conversation history into a structured prompt context
while enforcing strict grounding guardrails against hallucinated numerical claims.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import json

import pandas as pd

from backend.app.core.dataset_knowledge import DatasetKnowledge
from backend.app.core.semantic import SemanticSchemaAgent
from backend.app.core.llm_provider import LLMMessage


@dataclass
class DynamicPromptContext:
    """Structured assembled context ready for LLM consumption."""
    system_prompt: str
    user_prompt: str
    estimated_tokens: int
    messages: List[LLMMessage]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "estimated_tokens": self.estimated_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
        }


class DynamicContextAssembler:
    """Assembles token-budgeted, grounded prompt context for autonomous agents."""

    def __init__(self, max_context_tokens: int = 3000):
        self.max_context_tokens = max_context_tokens
        self.semantic_agent = SemanticSchemaAgent()

    def assemble(
        self,
        query: str,
        dataframe: Optional[pd.DataFrame] = None,
        knowledge: Optional[DatasetKnowledge] = None,
        agent_outputs: Optional[List[Dict[str, Any]]] = None,
        validation_issues: Optional[List[Dict[str, Any]]] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> DynamicPromptContext:
        """Dynamically assemble system instructions, dataset profiling, and agent state."""
        # 1. Base System Persona & Hard Grounding Guardrails
        system_sections = [
            "You are the Senior AI Technical Architect and Auto Data Analyst Orchestrator.",
            "",
            "### CORE GUARDRAILS & ARCHITECTURAL INVARIANTS:",
            "1. DETERMINISTIC CALCULATIONS ONLY: You must NEVER invent, extrapolate, or hallucinate numerical results. All calculations must originate from the specialized Python engines.",
            "2. NON-CAUSALITY: Never present correlation as causation. Always state appropriate confounding caveats.",
            "3. STRUCTURED ORCHESTRATION: Decide WHAT analytical agents need to run, while delegating the mathematical computations to the underlying engines.",
            "4. TRACEABILITY: Every insight must cite verifiable metrics and underlying data columns.",
            "",
            "### SPECIALIZED AGENTS AVAILABLE:",
            "- CleaningAgent: Handles missing value imputation, outlier winsorization, and type cleaning.",
            "- AnalysisAgent: Computes statistical summaries, frequency tables, and aggregations.",
            "- ModelSelectionAgent: Benchmarks traditional ML models (Linear, Ridge, Trees, RF, GBM, SVM, k-NN) with CV.",
            "- ANNAgent: Multi-layer perceptron neural networks with loss tracking, early stopping, and ML comparison.",
            "- CNNAgent: Convolutional neural networks for 2D images, spatial grids, and signal spectrograms.",
            "- ForecastAgent: Time series forecasting (Holt-Winters, ARIMA, trend projection).",
            "- DataValidationAgent: Audits data leakage, class imbalance, overfit/underfit, and temporal integrity.",
            "- VisualizationAgent: Generates responsive charts (bar, line, scatter, box, heatmap, histogram).",
            "- InsightAgent: Synthesizes evidence-based structured insights (Facts, Observations, Correlations, Inferences, Recommendations).",
            "- ReportAgent: Generates executive narrative summaries.",
        ]

        # 2. Dynamic Dataset Profiling Section
        if dataframe is not None and not dataframe.empty:
            dk = knowledge or self.semantic_agent.build_knowledge(dataframe)
            n_rows, n_cols = dataframe.shape

            metric_names = [m.column if hasattr(m, "column") else str(m) for m in dk.metrics]
            dim_names = [d.column if hasattr(d, "column") else str(d) for d in dk.dimensions]
            date_names = [d.column if hasattr(d, "column") else str(d) for d in dk.date_columns]

            dataset_sec = [
                "",
                "### CURRENT DATASET PROFILE:",
                f"- Dimensions: {n_rows:,} rows x {n_cols} columns",
                f"- Columns: {', '.join(dk.columns)}",
                f"- Primary Metrics: {', '.join(metric_names) if metric_names else 'None'}",
                f"- Primary Dimensions: {', '.join(dim_names) if dim_names else 'None'}",
                f"- Date Columns: {', '.join(date_names) if date_names else 'None'}",
                f"- Data Quality Score: {dk.data_quality.get('quality_score', 100)}/100",
            ]
            system_sections.extend(dataset_sec)

        # 3. Dynamic Validation State
        if validation_issues:
            val_sec = [
                "",
                "### ACTIVE VALIDATION ALERTS:",
            ]
            for iss in validation_issues[:3]:
                val_sec.append(f"- [{iss.get('severity', 'WARNING')}] {iss.get('title')}: {iss.get('description')}")
            system_sections.extend(val_sec)

        # 4. Recent Agent Outputs / Intermediate Findings
        if agent_outputs:
            findings_sec = [
                "",
                "### RECENT AGENT FINDINGS & COMPUTED METRICS:",
            ]
            for out in agent_outputs[-3:]:
                agent_name = out.get("agent", "Agent")
                res = out.get("output", {})
                if isinstance(res, dict):
                    if "best_model" in res:
                        bm = res["best_model"]
                        findings_sec.append(f"- {agent_name}: Best model is {bm.get('model_name')} with {bm.get('primary_metric_name')} = {bm.get('primary_metric_value')}")
                    elif "overall_status" in res:
                        findings_sec.append(f"- {agent_name}: Validation status is {res.get('overall_status')} ({res.get('critical_issues_count', 0)} critical)")
                    elif "summary" in res:
                        findings_sec.append(f"- {agent_name}: {res.get('summary')}")
            system_sections.extend(findings_sec)

        system_prompt = "\n".join(system_sections)

        # 5. Build Message List with History
        messages: List[LLMMessage] = [LLMMessage(role="system", content=system_prompt)]

        if history:
            for turn in history[-4:]:
                if turn.get("role") and turn.get("content"):
                    messages.append(LLMMessage(role=turn["role"], content=turn["content"]))

        messages.append(LLMMessage(role="user", content=query))

        # 6. Estimate token count (heuristic: ~4 chars per token)
        total_chars = sum(len(m.content) for m in messages)
        est_tokens = total_chars // 4

        return DynamicPromptContext(
            system_prompt=system_prompt,
            user_prompt=query,
            estimated_tokens=est_tokens,
            messages=messages,
        )
