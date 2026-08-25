"""
Command Intelligence & Intent Understanding Agent.

Parses freeform natural language user commands into structured, semantic UserIntent models.
Extracts:
- Core analytical intent (EDA, cleaning, aggregation, comparison, forecasting, prediction, root-cause, anomaly detection, etc.)
- Business objectives and requested outputs
- Entities (metrics, dimensions, filters, targets) cross-referenced with DatasetKnowledge
- Natural language temporal constraints (quarters, years, relative offsets) without synthetic date fabrication
- Multi-step capabilities for compound pipelines
- Ambiguity detection and clarification prompts for multiple candidate columns
- Standardized AgentResult integration with confidence scoring and evidence
"""
from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field, field_validator
import pandas as pd

from agent.base import BaseAgent
from agent.dataset_knowledge import ColumnKnowledge, DatasetKnowledge, SemanticType
from agent.schemas import AgentResult, AgentStatus, ClaimType, Evidence, SemanticMapping
from backend.app.core.llm_provider import BaseLLMProvider, LLMClientFactory, LLMMessage
from backend.app.core.semantic import BUSINESS_CONCEPTS, _normalize


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AnalyticalIntent(str, Enum):
    """Legacy AnalyticalIntent enum for backwards compatibility."""
    EDA = "eda"
    CLEANING = "cleaning"
    VISUALIZATION = "visualization"
    PREDICTION = "prediction"
    FORECASTING = "forecasting"
    DEEP_LEARNING = "deep_learning"
    CNN = "cnn"
    EXPLANATION = "explanation"
    ANOMALIES = "anomalies"
    REPORT = "report"
    UNKNOWN = "unknown"


class IntentType(str, Enum):
    """Controlled set of analytical and operational intent types."""
    DATASET_ANALYSIS = "dataset_analysis"
    DATA_CLEANING = "data_cleaning"
    AGGREGATION = "aggregation"
    COMPARISON = "comparison"
    TREND_ANALYSIS = "trend_analysis"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    FORECASTING = "forecasting"
    PREDICTION = "prediction"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    VISUALIZATION = "visualization"
    REPORTING = "reporting"
    DATA_QUESTION = "data_question"
    MODEL_TRAINING = "model_training"
    MULTI_STAGE_PIPELINE = "multi_stage_pipeline"
    RECOMMENDATION = "recommendation"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Structured Intent Models
# ---------------------------------------------------------------------------

class UserIntent(BaseModel):
    """
    Standardized, structured interpretation of a natural language user command.
    Passed as context to the task planner for capability orchestration.
    """
    intent_type: Union[IntentType, str] = IntentType.DATASET_ANALYSIS
    objective: str = ""
    entities: Dict[str, Any] = Field(default_factory=dict)
    metrics: List[str] = Field(default_factory=list)
    dimensions: List[str] = Field(default_factory=list)
    filters: Dict[str, Any] = Field(default_factory=dict)
    time_range: Optional[Dict[str, Any]] = None
    comparison: Optional[Dict[str, Any]] = None
    ranking: Optional[Dict[str, Any]] = None
    aggregation_type: Optional[str] = None
    requested_output: str = "analysis"
    required_capabilities: List[str] = Field(default_factory=list)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    ambiguities: List[str] = Field(default_factory=list)
    needs_clarification: bool = False
    original_command: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return float(v)

    def to_dict(self) -> Dict[str, Any]:
        it_val = self.intent_type.value if isinstance(self.intent_type, IntentType) else str(self.intent_type)
        return {
            "intent_type": it_val,
            "objective": self.objective,
            "entities": self.entities,
            "metrics": self.metrics,
            "dimensions": self.dimensions,
            "filters": self.filters,
            "time_range": self.time_range,
            "comparison": self.comparison,
            "ranking": self.ranking,
            "aggregation_type": self.aggregation_type,
            "requested_output": self.requested_output,
            "required_capabilities": self.required_capabilities,
            "constraints": self.constraints,
            "confidence": round(float(self.confidence), 4),
            "ambiguities": self.ambiguities,
            "needs_clarification": self.needs_clarification,
            "original_command": self.original_command,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserIntent":
        return cls.model_validate(data)


class IntentClassificationResult(BaseModel):
    """Legacy intent classification result for backwards compatibility."""
    primary_intent: Union[AnalyticalIntent, IntentType, str]
    secondary_intents: List[Union[AnalyticalIntent, IntentType, str]] = Field(default_factory=list)
    confidence: float = 0.9
    target_column: Optional[str] = None
    feature_columns: List[str] = Field(default_factory=list)
    group_by: Optional[str] = None
    time_horizon: Optional[int] = None
    chart_type: Optional[str] = None
    top_k: Optional[int] = None
    needs_cleaning: bool = False
    needs_explanation: bool = False
    raw_query: str = ""
    reasoning: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        pi = self.primary_intent.value if hasattr(self.primary_intent, "value") else str(self.primary_intent)
        sec = [i.value if hasattr(i, "value") else str(i) for i in self.secondary_intents]
        return {
            "primary_intent": pi,
            "secondary_intents": sec,
            "confidence": round(float(self.confidence), 3),
            "target_column": self.target_column,
            "feature_columns": self.feature_columns,
            "group_by": self.group_by,
            "time_horizon": self.time_horizon,
            "chart_type": self.chart_type,
            "top_k": self.top_k,
            "needs_cleaning": self.needs_cleaning,
            "needs_explanation": self.needs_explanation,
            "raw_query": self.raw_query,
            "reasoning": self.reasoning,
        }


# ---------------------------------------------------------------------------
# Command Intelligence Agent
# ---------------------------------------------------------------------------

class CommandIntelligenceAgent(BaseAgent):
    """
    Intelligent NLP & Intent Understanding Agent.
    Transforms natural language queries into structured UserIntent specifications.
    """
    name = "Command Intelligence Agent"
    description = "Parses user natural language into structured analytical intents."
    role = "intent_understanding"

    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None):
        super().__init__()
        self.llm_provider = llm_provider

    def run(self, task: Union[str, Dict[str, Any]]) -> AgentResult:
        """Execute intent parsing returning standardized AgentResult."""
        self._start()
        try:
            command = ""
            dk: Optional[DatasetKnowledge] = None
            if isinstance(task, str):
                command = task
            elif isinstance(task, dict):
                command = task.get("command") or task.get("query") or task.get("text", "")
                dk = task.get("dataset_knowledge") or task.get("knowledge")

            if not command or not command.strip():
                return self._error(
                    message="Command Intelligence Agent requires a non-empty user command string.",
                    code="EMPTY_COMMAND",
                )

            intent = self.analyze_intent(command, dataset_knowledge=dk)

            # Build evidence for intent classification
            ev = self.make_evidence(
                method="command_intent_parser",
                data_ref={
                    "command": command,
                    "intent_type": intent.intent_type.value if isinstance(intent.intent_type, IntentType) else str(intent.intent_type),
                    "metrics": intent.metrics,
                    "dimensions": intent.dimensions,
                    "required_capabilities": intent.required_capabilities,
                },
                confidence=intent.confidence,
                claim_type=ClaimType.FACT if intent.confidence >= 0.85 else ClaimType.OBSERVATION,
                raw_value=intent.to_dict(),
                metadata={"objective": intent.objective, "ambiguities": intent.ambiguities},
                operation="parse_intent",
                calculation=f"P(intent={intent.intent_type}|cmd) = {intent.confidence:.2f}",
            )

            status_msg = f"Parsed command intent as '{intent.intent_type}' (confidence: {intent.confidence:.2f})."
            if intent.needs_clarification:
                status_msg += f" Clarification needed: {intent.ambiguities[0]}"

            return self._finish(
                result={"user_intent": intent.to_dict()},
                message=status_msg,
                evidence=[ev],
                confidence=intent.confidence,
                metadata={"needs_clarification": intent.needs_clarification},
            )
        except Exception as exc:
            return self._error(
                message=f"Failed to analyze command intent: {str(exc)}",
                code="INTENT_ANALYSIS_ERROR",
                details={"exception": str(exc)},
            )

    # ------------------------------------------------------------------
    # Core Intent Parsing Pipeline
    # ------------------------------------------------------------------
    def analyze_intent(
        self,
        command: str,
        dataset_knowledge: Optional[DatasetKnowledge] = None,
    ) -> UserIntent:
        """
        Analyze command using hybrid LLM reasoning + deterministic semantic grounding.
        """
        text = command.strip()
        lowered = text.lower()

        # 1. Parse time expressions
        time_info = self._extract_time_range(lowered)

        # 2. Extract multi-step operations & capabilities
        capabilities, primary_intent, requested_out = self._detect_capabilities_and_intent(lowered)

        # 3. Extract aggregations, rankings, comparisons, and filters
        agg_type = self._extract_aggregation(lowered)
        ranking = self._extract_ranking(lowered)
        comparison = self._extract_comparison(lowered)
        filters = self._extract_filters(lowered, comparison)

        # 4. Extract & Cross-Reference Entities against DatasetKnowledge
        metrics, dimensions, ambiguities, needs_clarification = self._match_entities_with_dataset(
            lowered, dataset_knowledge
        )

        # 5. Determine confidence
        confidence = 0.95
        if not metrics and not dimensions and primary_intent == IntentType.UNKNOWN:
            confidence = 0.30
        elif needs_clarification:
            confidence = 0.50
        elif primary_intent == IntentType.UNKNOWN:
            confidence = 0.60
        elif not capabilities:
            confidence = 0.70

        # 6. Compose objective narrative
        objective = self._compose_objective(primary_intent, metrics, dimensions, time_info, ranking, comparison)

        return UserIntent(
            intent_type=primary_intent,
            objective=objective,
            entities={
                "metrics": metrics,
                "dimensions": dimensions,
                "filters": filters,
                "ranking": ranking,
                "comparison": comparison,
            },
            metrics=metrics,
            dimensions=dimensions,
            filters=filters,
            time_range=time_info,
            comparison=comparison,
            ranking=ranking,
            aggregation_type=agg_type,
            requested_output=requested_out,
            required_capabilities=capabilities,
            constraints={"ranking": ranking} if ranking else {},
            confidence=confidence,
            ambiguities=ambiguities,
            needs_clarification=needs_clarification,
            original_command=command,
            metadata={"parsed_at": datetime.now().isoformat()},
        )

    # ------------------------------------------------------------------
    # Time Expression Understanding
    # ------------------------------------------------------------------
    def _extract_time_range(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract structured time constraints without synthesizing fake dates."""
        # 1. Quarters (e.g. "Q1", "Q2", "Q3", "Q4", "Q3 2024", "last quarter", "this quarter")
        q_match = re.search(r"\b(q[1-4])\b(?:\s+(\d{4}))?", text)
        if q_match:
            q_name = q_match.group(1).upper()
            y_val = int(q_match.group(2)) if q_match.group(2) else None
            return {"type": "quarter", "quarter": q_name, "year": y_val, "raw": q_match.group(0)}

        # 2. Month + Year or Month name (e.g. "january 2026", "next month", "last month")
        month_match = re.search(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b(?:\s+(\d{4}))?",
            text,
        )
        if month_match:
            m_name = month_match.group(1).lower()
            y_val = int(month_match.group(2)) if month_match.group(2) else None
            return {"type": "month", "month": m_name, "year": y_val, "raw": month_match.group(0)}

        # 3. Relative offsets
        relative_patterns = {
            "last quarter": {"type": "relative_quarter", "period": "last_quarter", "raw": "last quarter"},
            "this quarter": {"type": "relative_quarter", "period": "this_quarter", "raw": "this quarter"},
            "last month": {"type": "relative_month", "period": "last_month", "raw": "last month"},
            "this month": {"type": "relative_month", "period": "this_month", "raw": "this month"},
            "next month": {"type": "relative_month", "period": "next_month", "raw": "next month"},
            "last year": {"type": "relative_year", "period": "last_year", "raw": "last year"},
            "this year": {"type": "relative_year", "period": "this_year", "raw": "this year"},
            "next year": {"type": "relative_year", "period": "next_year", "raw": "next year"},
            "today": {"type": "relative_day", "period": "today", "raw": "today"},
            "yesterday": {"type": "relative_day", "period": "yesterday", "raw": "yesterday"},
            "year over year": {"type": "comparison_period", "period": "yoy", "raw": "year over year"},
            "month over month": {"type": "comparison_period", "period": "mom", "raw": "month over month"},
        }
        for pat, struct in relative_patterns.items():
            if pat in text:
                return struct

        # 4. Standalone Year matching (e.g. 2024, 2025, 2026)
        year_match = re.search(r"\b(19\d\d|20\d\d)\b", text)
        if year_match:
            y = int(year_match.group(1))
            return {"type": "year", "year": y, "raw": year_match.group(0)}

        return None

    # ------------------------------------------------------------------
    # Intent & Multi-Step Capability Classification
    # ------------------------------------------------------------------
    def _detect_capabilities_and_intent(self, text: str) -> Tuple[List[str], IntentType, str]:
        """Extract multi-step required capabilities and determine the primary intent."""
        capabilities: List[str] = []
        intent_candidates: List[Tuple[IntentType, int]] = []
        requested_output = "analysis"

        # 1. Cleaning / Duplicates
        if any(w in text for w in ("clean", "preprocess", "sanitize", "impute")):
            capabilities.append("data_cleaning")
            intent_candidates.append((IntentType.DATA_CLEANING, 90))
        if any(w in text for w in ("remove duplicate", "remove duplicates", "duplicate", "dedup")):
            capabilities.append("duplicate_handling")
            if (IntentType.DATA_CLEANING, 90) not in intent_candidates:
                intent_candidates.append((IntentType.DATA_CLEANING, 85))

        # 2. Root Cause / Why
        if any(w in text for w in ("why", "root cause", "driver", "drivers", "reason for", "cause of")):
            capabilities.extend(["trend_analysis", "segmentation", "anomaly_detection", "relationship_analysis"])
            intent_candidates.append((IntentType.ROOT_CAUSE_ANALYSIS, 95))
            requested_output = "explanation"

        # 3. Comparison
        if any(w in text for w in ("compare", "versus", "vs", "difference between", "between")):
            capabilities.append("comparison")
            intent_candidates.append((IntentType.COMPARISON, 92))

        # 4. Forecasting / Prediction / ML
        if any(w in text for w in ("forecast", "future", "projection", "next month", "next quarter")):
            capabilities.append("forecasting")
            intent_candidates.append((IntentType.FORECASTING, 94))
            requested_output = "forecast"
        elif any(w in text for w in ("predict", "train", "model", "churn", "classify", "regression")):
            capabilities.extend(["feature_engineering", "model_training", "prediction"])
            if "churn" in text or "classify" in text or "classifier" in text:
                intent_candidates.append((IntentType.CLASSIFICATION, 92))
            else:
                intent_candidates.append((IntentType.PREDICTION, 90))
            requested_output = "model"

        # 5. Anomaly Detection
        if any(w in text for w in ("unusual", "anomaly", "anomalies", "outlier", "outliers", "suspicious")):
            capabilities.append("anomaly_detection")
            intent_candidates.append((IntentType.ANOMALY_DETECTION, 93))

        # 6. Aggregation / Ranking / Regional Analysis
        if any(w in text for w in ("by region", "regional", "by country", "by category", "by customer")):
            capabilities.append("regional_analysis" if "region" in text else "segmentation")
        if any(w in text for w in ("total", "sum", "average", "mean", "count", "top", "highest", "lowest")):
            capabilities.append("aggregation")
            if not intent_candidates:
                intent_candidates.append((IntentType.AGGREGATION, 80))

        # 7. Visualization / Report
        if any(w in text for w in ("report", "pdf", "slide", "deck", "summary presentation")):
            capabilities.append("reporting")
            intent_candidates.append((IntentType.REPORTING, 91))
            requested_output = "report"
        elif any(w in text for w in ("chart", "plot", "graph", "visualize", "histogram", "scatter", "bar chart")):
            capabilities.append("visualization")
            if not intent_candidates:
                intent_candidates.append((IntentType.VISUALIZATION, 85))
            requested_output = "chart"

        # 8. Recommendation / Decision Support
        if any(w in text for w in ("recommend", "recommendation", "what should i do",
                                   "what to do", "advise", "decision", "which action",
                                   "safest", "increase revenue while", "reducing risk")):
            capabilities.extend(["decision_support", "recommendation_generation"])
            intent_candidates.append((IntentType.RECOMMENDATION, 90))
            requested_output = "recommendation"
        elif any(w in text for w in ("should we", "should i", "trade-off", "option a", "option b")):
            capabilities.append("decision_support")
            intent_candidates.append((IntentType.RECOMMENDATION, 85))
            requested_output = "recommendation"

        # Multi-step determination
        if len(capabilities) >= 3:
            primary = IntentType.MULTI_STAGE_PIPELINE if len(intent_candidates) > 1 else intent_candidates[0][0]
        elif intent_candidates:
            intent_candidates.sort(key=lambda x: x[1], reverse=True)
            primary = intent_candidates[0][0]
        elif any(w in text for w in ("analyze", "analysis", "explore", "inspect", "dataset", "data", "summary", "stats", "tell me", "what is", "how is", "show", "give", "overview", "distribution", "breakdown", "performance", "patterns", "numbers", "insights", "metrics")):
            primary = IntentType.DATASET_ANALYSIS
            capabilities.append("exploratory_analysis")
        else:
            primary = IntentType.UNKNOWN

        return capabilities, primary, requested_output

    # ------------------------------------------------------------------
    # Operations (Aggregation, Ranking, Comparison)
    # ------------------------------------------------------------------
    def _extract_aggregation(self, text: str) -> Optional[str]:
        if any(w in text for w in ("average", "avg", "mean")):
            return "mean"
        if any(w in text for w in ("total", "sum", "overall sum", "aggregate")):
            return "sum"
        if any(w in text for w in ("count", "how many", "number of")):
            return "count"
        if any(w in text for w in ("max", "maximum", "highest", "largest")):
            return "max"
        if any(w in text for w in ("min", "minimum", "lowest", "smallest")):
            return "min"
        return None

    def _extract_ranking(self, text: str) -> Optional[Dict[str, Any]]:
        # Match "top 10", "top 5", "first 10", "bottom 3"
        top_match = re.search(r"\b(top|first|highest|largest)\s+(\d+)\b", text)
        if top_match:
            return {"type": "top", "limit": int(top_match.group(2)), "order": "desc"}
        bot_match = re.search(r"\b(bottom|lowest|smallest)\s+(\d+)\b", text)
        if bot_match:
            return {"type": "bottom", "limit": int(bot_match.group(2)), "order": "asc"}
        return None

    def _clean_entity_name(self, raw_entity: str) -> str:
        cleaned = re.sub(r"^(?:the|an|a)\s+", "", raw_entity.strip(), flags=re.IGNORECASE).strip()
        if cleaned.lower() in ("us", "usa", "uk", "uae"):
            return cleaned.upper()
        return cleaned.title()

    def _extract_comparison(self, text: str) -> Optional[Dict[str, Any]]:
        # Match "between X and Y"
        between_match = re.search(r"\bbetween\s+([a-zA-Z0-9_\s]+?)\s+and\s+([a-zA-Z0-9_\s]+?)(?:\s+in|\s+by|\s+for|$|\.)", text, flags=re.IGNORECASE)
        if between_match:
            left = self._clean_entity_name(between_match.group(1))
            right = self._clean_entity_name(between_match.group(2))
            return {"type": "between_entities", "entities": [left, right], "entity_a": left, "entity_b": right}
        if "vs" in text or "versus" in text:
            parts = re.split(r"\b(?:vs|versus)\b", text, flags=re.IGNORECASE)
            if len(parts) >= 2:
                left = self._clean_entity_name(parts[0])
                right = self._clean_entity_name(parts[1])
                return {"type": "between_entities", "entities": [left, right], "entity_a": left, "entity_b": right}
        return None

    def _extract_filters(self, text: str, comparison: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        filters: Dict[str, Any] = {}
        if comparison and comparison.get("type") == "between_entities":
            entities = comparison.get("entities", [])
            filters["entities"] = entities
            if len(entities) >= 2:
                filters["comparison_targets"] = entities

        # Status / active filter
        if "active" in text and "inactive" not in text:
            filters["status"] = "Active"
        elif "inactive" in text:
            filters["status"] = "Inactive"

        # Explicit equality or region matching (e.g. region == North or in North)
        reg_match = re.search(r"\b(?:in|region\s*==?)\s+(north|south|east|west|central|apac|emea|latam)\b", text)
        if reg_match:
            filters["region"] = reg_match.group(1).capitalize()

        return filters

    # ------------------------------------------------------------------
    # Entity Extraction & DatasetKnowledge Grounding
    # ------------------------------------------------------------------
    def _match_entities_with_dataset(
        self,
        text: str,
        dk: Optional[DatasetKnowledge],
    ) -> Tuple[List[str], List[str], List[str], bool]:
        """
        Cross-reference user mentions against DatasetKnowledge.
        Detects ambiguous column names when multiple close candidates match.
        """
        metrics: List[str] = []
        dimensions: List[str] = []
        ambiguities: List[str] = []
        needs_clarification = False

        if dk is not None:
            col_names = [c if isinstance(c, str) else c.column_name for c in dk.columns]
            num_cols = [c if isinstance(c, str) else c.column_name for c in getattr(dk, "numeric_columns", [])]
            cat_cols = [c if isinstance(c, str) else c.column_name for c in getattr(dk, "categorical_columns", [])]

            # 1. Direct column matching for all dataset columns
            norm_text = " " + re.sub(r"[^\w\s]", " ", text) + " "
            for col in col_names:
                col_clean = col.lower()
                col_spaced = col_clean.replace("_", " ").replace("-", " ")
                if col_clean in text or f" {col_spaced} " in norm_text:
                    if col in num_cols or not cat_cols or col not in cat_cols:
                        metrics.append(col)
                    else:
                        dimensions.append(col)

            # 2. Search for potential metric references with synonyms
            metric_candidates = ("revenue", "sales", "profit", "cost", "quantity", "units", "price", "discount", "margin", "salary", "amount", "spend", "charges", "value", "balance", "score")
            for term in metric_candidates:
                if term in text:
                    exact_match = [c for c in col_names if c.lower() == term]
                    sub_matches = [c for c in col_names if term in c.lower()]

                    if exact_match and exact_match[0] not in metrics:
                        metrics.append(exact_match[0])
                    elif len(sub_matches) == 1 and sub_matches[0] not in metrics:
                        metrics.append(sub_matches[0])
                    elif len(sub_matches) > 1:
                        needs_clarification = True
                        ambiguity_msg = f"I found multiple matching columns for '{term}' ({', '.join([repr(c) for c in sub_matches])}). Which should I analyze?"
                        ambiguities.append(ambiguity_msg)
                        metrics.extend(sub_matches)

            # 3. Search for potential dimension references
            dim_candidates = ("country", "region", "city", "customer", "customer_id", "product", "category", "segment", "status", "type", "gender", "tier", "channel")
            for term in dim_candidates:
                if term in text:
                    matches = [c for c in col_names if term in c.lower()]
                    if matches and matches[0] not in dimensions:
                        dimensions.append(matches[0])

            # 4. Country / Region inference from comparison entities (e.g. India vs US)
            country_indicators = ("india", "us", "usa", "uk", "germany", "france", "canada", "australia", "china", "japan")
            if any(cnt in text for cnt in country_indicators):
                for col in ("country", "region", "geo", "location", "territory"):
                    if col in col_names and col not in dimensions:
                        dimensions.append(col)

        else:
            # Fallback when DatasetKnowledge is not yet attached
            metric_terms = ("revenue", "profit", "sales", "cost", "quantity", "units", "price", "churn", "salary", "amount", "spend", "charges")
            for m in metric_terms:
                if m in text and m not in metrics:
                    metrics.append(m)

            dim_terms = ("region", "country", "city", "customer", "category", "product", "segment", "status", "tier")
            for d in dim_terms:
                if d in text and d not in dimensions:
                    dimensions.append(d)

        # Deduplicate
        metrics = list(dict.fromkeys(metrics))
        dimensions = list(dict.fromkeys(dimensions))

        return metrics, dimensions, ambiguities, needs_clarification

    # ------------------------------------------------------------------
    # Narrative Composition Helper
    # ------------------------------------------------------------------
    def _compose_objective(
        self,
        intent: IntentType,
        metrics: List[str],
        dimensions: List[str],
        time_info: Optional[Dict[str, Any]],
        ranking: Optional[Dict[str, Any]],
        comparison: Optional[Dict[str, Any]],
    ) -> str:
        parts = []
        if intent == IntentType.ROOT_CAUSE_ANALYSIS:
            parts.append(f"Perform root-cause analysis on {', '.join(metrics) or 'target metrics'}")
        elif intent == IntentType.FORECASTING:
            parts.append(f"Generate time-series forecast for {', '.join(metrics) or 'target metric'}")
        elif intent == IntentType.COMPARISON:
            parts.append(f"Compare {', '.join(metrics) or 'metrics'} across {', '.join(dimensions) or 'dimensions'}")
        elif intent == IntentType.DATA_CLEANING:
            parts.append("Execute data cleaning and quality sanitization")
        elif intent == IntentType.ANOMALY_DETECTION:
            parts.append(f"Identify unusual anomalies and outliers in {', '.join(metrics) or 'dataset'}")
        elif intent == IntentType.PREDICTION:
            parts.append(f"Train predictive model for {', '.join(metrics) or 'target variable'}")
        elif ranking:
            parts.append(f"Find {ranking.get('type')} {ranking.get('limit')} records sorted by {', '.join(metrics) or 'metric'}")
        else:
            parts.append("Perform exploratory data analysis and insight extraction")

        if time_info:
            parts.append(f"for {time_info.get('raw')}")

        return " ".join(parts) + "."


# ---------------------------------------------------------------------------
# Legacy IntentAnalyzer Implementation for Full Backward Compatibility
# ---------------------------------------------------------------------------

class IntentAnalyzer:
    """Deterministic, contextual intent extractor with semantic knowledge integration."""

    INTENT_KEYWORDS = {
        AnalyticalIntent.CLEANING: (
            "clean", "preprocess", "impute", "missing", "outlier removal",
            "dedup", "duplicates", "null values", "format data", "sanitize",
        ),
        AnalyticalIntent.FORECASTING: (
            "forecast", "future", "predict next", "time series", "horizon",
            "next month", "next quarter", "next year", "projection", "trend over time",
        ),
        AnalyticalIntent.CNN: (
            "cnn", "convolutional", "image", "spatial", "spectrogram", "signal image",
            "pixel", "grid data", "image classification",
        ),
        AnalyticalIntent.DEEP_LEARNING: (
            "deep learning", "ann", "neural network", "mlp", "multi-layer perceptron",
            "deep model", "pytorch",
        ),
        AnalyticalIntent.PREDICTION: (
            "train", "predict", "classifier", "regression", "model", "churn",
            "supervised", "forecast target", "random forest", "logistic regression",
            "xgboost", "fit model",
        ),
        AnalyticalIntent.EXPLANATION: (
            "explain", "driver", "drivers", "feature importance", "why",
            "top factors", "influence", "key indicators", "shap", "coefficients",
        ),
        AnalyticalIntent.VISUALIZATION: (
            "chart", "plot", "graph", "visualize", "bar chart", "line chart",
            "scatter", "histogram", "heatmap", "distribution plot",
        ),
        AnalyticalIntent.ANOMALIES: (
            "anomaly", "anomalies", "outlier", "outliers", "unusual", "deviations",
        ),
        AnalyticalIntent.REPORT: (
            "report", "summary report", "executive summary", "overview",
            "full analysis", "brief", "presentation",
        ),
        AnalyticalIntent.EDA: (
            "eda", "describe", "summary", "stats", "statistics", "correlations",
            "explore", "profile", "distribution", "overview",
        ),
    }

    def __init__(self):
        self.modern_agent = CommandIntelligenceAgent()

    def analyze(
        self,
        query: str,
        knowledge: Optional[Union[DatasetKnowledge, pd.DataFrame]] = None,
        dataframe: Optional[pd.DataFrame] = None,
        df: Optional[Any] = None,
    ) -> IntentClassificationResult:
        """Parse natural language query into a structured multi-intent profile."""
        if isinstance(knowledge, pd.DataFrame) and dataframe is None:
            dataframe = knowledge
            knowledge = None
        if df is not None:
            if isinstance(df, pd.DataFrame) and dataframe is None:
                dataframe = df
            elif isinstance(df, DatasetKnowledge) and knowledge is None:
                knowledge = df

        q_norm = query.strip().lower()
        reasoning: List[str] = []
        matched_intents: List[AnalyticalIntent] = []

        # 1. Match all intents mentioned in the query
        for intent, kw_list in self.INTENT_KEYWORDS.items():
            hits = [kw for kw in kw_list if kw in q_norm or f" {kw} " in f" {q_norm} "]
            if hits:
                matched_intents.append(intent)
                reasoning.append(f"Matched {intent.value} via keywords: {', '.join(hits)}")

        # Distinguish prediction vs forecasting vs deep learning
        primary = AnalyticalIntent.EDA
        secondary: List[AnalyticalIntent] = []

        if AnalyticalIntent.CNN in matched_intents:
            primary = AnalyticalIntent.CNN
        elif AnalyticalIntent.DEEP_LEARNING in matched_intents:
            primary = AnalyticalIntent.DEEP_LEARNING
        elif AnalyticalIntent.FORECASTING in matched_intents:
            primary = AnalyticalIntent.FORECASTING
        elif AnalyticalIntent.PREDICTION in matched_intents:
            primary = AnalyticalIntent.PREDICTION
        elif AnalyticalIntent.CLEANING in matched_intents:
            primary = AnalyticalIntent.CLEANING
        elif AnalyticalIntent.EXPLANATION in matched_intents:
            primary = AnalyticalIntent.EXPLANATION
        elif AnalyticalIntent.VISUALIZATION in matched_intents:
            primary = AnalyticalIntent.VISUALIZATION
        elif AnalyticalIntent.ANOMALIES in matched_intents:
            primary = AnalyticalIntent.ANOMALIES
        elif AnalyticalIntent.REPORT in matched_intents:
            primary = AnalyticalIntent.REPORT
        elif matched_intents:
            primary = matched_intents[0]

        secondary = [i for i in matched_intents if i != primary]

        # 2. Extract Top-K requirement
        top_k = None
        top_k_match = re.search(r"top\s+(\d+)", q_norm)
        if top_k_match:
            top_k = int(top_k_match.group(1))
            reasoning.append(f"Detected top-{top_k} limit")

        # 3. Extract time horizon
        horizon = None
        horizon_match = re.search(r"next\s+(\d+)\s*(?:month|day|week|quarter|period|year)?", q_norm)
        if horizon_match:
            horizon = int(horizon_match.group(1))
            reasoning.append(f"Detected time horizon of {horizon} periods")

        # 4. Extract Chart Type
        chart_type = None
        for ctype in ("bar", "line", "scatter", "histogram", "heatmap", "box", "pie"):
            if ctype in q_norm or f"{ctype} chart" in q_norm:
                chart_type = ctype
                reasoning.append(f"Detected requested chart type: {ctype}")
                break

        # 5. Extract Target Column and Features using DatasetKnowledge / DataFrame
        target_col = None
        feature_cols: List[str] = []
        group_by_col = None

        columns: List[str] = []
        if knowledge is not None and hasattr(knowledge, "columns"):
            columns = [c if isinstance(c, str) else c.column_name for c in knowledge.columns]
        elif dataframe is not None:
            columns = list(dataframe.columns)

        if len(columns) > 0:
            for col in columns:
                col_lower = col.lower()
                if col_lower in q_norm or col_lower.replace("_", " ") in q_norm:
                    if target_col is None and (
                        "predict" in q_norm or "target" in q_norm or "forecast" in q_norm
                    ):
                        target_col = col
                        reasoning.append(f"Found target column '{col}' in query")
                    else:
                        feature_cols.append(col)

            if target_col is None and knowledge is not None:
                for concept in ("churn", "revenue", "profit", "sales", "price", "salary"):
                    if concept in q_norm:
                        matched = knowledge.find_columns_by_concept(concept)
                        if matched:
                            target_col = matched[0]
                            reasoning.append(f"Mapped concept '{concept}' to column '{target_col}'")
                            break

            if target_col is None and primary in (
                AnalyticalIntent.PREDICTION,
                AnalyticalIntent.FORECASTING,
                AnalyticalIntent.DEEP_LEARNING,
            ):
                if knowledge is not None:
                    target_col = knowledge.get_primary_metric()
                    if target_col:
                        reasoning.append(f"Defaulted target to primary metric '{target_col}'")

        needs_cleaning = AnalyticalIntent.CLEANING in matched_intents or "clean" in q_norm
        needs_explanation = (
            AnalyticalIntent.EXPLANATION in matched_intents
            or "explain" in q_norm
            or "driver" in q_norm
            or top_k is not None
        )

        confidence = 0.95 if matched_intents else 0.70

        return IntentClassificationResult(
            primary_intent=primary,
            secondary_intents=secondary,
            confidence=confidence,
            target_column=target_col,
            feature_columns=feature_cols,
            group_by=group_by_col,
            time_horizon=horizon,
            chart_type=chart_type,
            top_k=top_k,
            needs_cleaning=needs_cleaning,
            needs_explanation=needs_explanation,
            raw_query=query,
            reasoning=reasoning,
        )
