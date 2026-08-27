"""
Universal Conversational Analytical Context & Session Memory Layer.

Single source of truth for:
- Multi-turn analytical context tracking (sessions, datasets, schemas, rows)
- Non-destructive, state-aware reference resolution ("it", "that", "the target", "the forecast", "cluster 2")
- Contextual intent routing & follow-up task generation
- Multi-dataset switching and session isolation
- Bounded result and metric history (zero massive DataFrame storage in context objects)
- Ambiguity detection and structured clarification requests
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import math
import re
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from agent.agent_result import (
    AgentError,
    AgentResult,
    AgentStatus,
    ClaimType,
    ErrorCategory,
    Evidence,
)
from agent.canonical_data_layer import CanonicalDataLayer, SemanticProfile


# ------------------------------------------------------------------------------
# Structured Context Models
# ------------------------------------------------------------------------------

class DatasetSnapshot(BaseModel):
    """Compact, bounded summary of a dataset within a session (no raw bulk rows)."""
    dataset_id: str
    dataset_name: str
    columns: List[str] = Field(default_factory=list)
    numeric_columns: List[str] = Field(default_factory=list)
    categorical_columns: List[str] = Field(default_factory=list)
    datetime_columns: List[str] = Field(default_factory=list)
    identifier_columns: List[str] = Field(default_factory=list)
    constant_columns: List[str] = Field(default_factory=list)
    original_rows: int = 0
    current_rows: int = 0
    preview_sample: List[Dict[str, Any]] = Field(default_factory=list) # max 5 rows
    quality_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "columns": self.columns,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "datetime_columns": self.datetime_columns,
            "identifier_columns": self.identifier_columns,
            "constant_columns": self.constant_columns,
            "original_rows": self.original_rows,
            "current_rows": self.current_rows,
            "preview_sample": self.preview_sample,
            "quality_score": round(self.quality_score, 4),
        }


class ExecutionRecord(BaseModel):
    """Bounded record of a past analytical execution."""
    execution_id: str
    turn_id: int
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    user_command: str
    resolved_command: str
    task_type: str
    intent: str
    target: Optional[str] = None
    features: List[str] = Field(default_factory=list)
    time_column: Optional[str] = None
    model_selected: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.85
    status: str = "success"
    summary: str = ""
    top_findings: List[str] = Field(default_factory=list)
    resolved_references: Dict[str, str] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "turn_id": self.turn_id,
            "timestamp": self.timestamp,
            "user_command": self.user_command,
            "resolved_command": self.resolved_command,
            "task_type": self.task_type,
            "intent": self.intent,
            "target": self.target,
            "features": self.features,
            "time_column": self.time_column,
            "model_selected": self.model_selected,
            "metrics": self.metrics,
            "confidence": round(self.confidence, 4),
            "status": self.status,
            "summary": self.summary,
            "top_findings": self.top_findings,
            "resolved_references": self.resolved_references,
        }


class AnalyticalContext(BaseModel):
    """Structured analytical session context."""
    session_id: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    last_active_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    active_dataset_id: Optional[str] = None
    datasets: Dict[str, DatasetSnapshot] = Field(default_factory=dict)
    active_target: Optional[str] = None
    active_features: List[str] = Field(default_factory=list)
    active_time_column: Optional[str] = None
    active_task: Optional[str] = None
    previous_task: Optional[str] = None
    current_intent: Optional[str] = None
    previous_intent: Optional[str] = None
    last_execution_id: Optional[str] = None
    latest_result_summary: Optional[Dict[str, Any]] = None
    latest_metrics: Dict[str, Any] = Field(default_factory=dict)
    latest_confidence: float = 0.85
    latest_model_name: Optional[str] = None
    latest_forecast_horizon: Optional[int] = None
    latest_cluster_count: Optional[int] = None
    latest_anomaly_count: Optional[int] = None
    latest_strongest_relationship: Optional[Dict[str, Any]] = None
    latest_second_strongest_relationship: Optional[Dict[str, Any]] = None
    execution_history: List[ExecutionRecord] = Field(default_factory=list)
    pending_clarification: Optional[Dict[str, Any]] = None
    assumptions: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_active_at": self.last_active_at,
            "active_dataset_id": self.active_dataset_id,
            "datasets": {k: v.to_dict() for k, v in self.datasets.items()},
            "active_target": self.active_target,
            "active_features": self.active_features,
            "active_time_column": self.active_time_column,
            "active_task": self.active_task,
            "previous_task": self.previous_task,
            "current_intent": self.current_intent,
            "previous_intent": self.previous_intent,
            "last_execution_id": self.last_execution_id,
            "latest_result_summary": self.latest_result_summary,
            "latest_metrics": self.latest_metrics,
            "latest_confidence": round(self.latest_confidence, 4),
            "latest_model_name": self.latest_model_name,
            "latest_forecast_horizon": self.latest_forecast_horizon,
            "latest_cluster_count": self.latest_cluster_count,
            "latest_anomaly_count": self.latest_anomaly_count,
            "latest_strongest_relationship": self.latest_strongest_relationship,
            "execution_history": [h.to_dict() for h in self.execution_history],
            "pending_clarification": self.pending_clarification,
            "assumptions": self.assumptions,
            "limitations": self.limitations,
            "warnings": self.warnings,
        }


class ContextualResolution(BaseModel):
    """Result of resolving a natural language command against conversational context."""
    user_command: str
    resolved_command: str
    detected_intent: str
    target: Optional[str] = None
    features: List[str] = Field(default_factory=list)
    time_column: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    is_follow_up: bool = False
    needs_clarification: bool = False
    clarification_reason: Optional[str] = None
    suggested_options: List[str] = Field(default_factory=list)
    resolved_references: Dict[str, str] = Field(default_factory=dict)
    dataset_id: Optional[str] = None


# ------------------------------------------------------------------------------
# Universal Reference Resolver
# ------------------------------------------------------------------------------

class UniversalReferenceResolver:
    """
    Resolves anaphoric expressions, pronouns, entity references, and follow-up
    modifiers using structured AnalyticalContext.
    """

    def resolve(
        self,
        command: str,
        context: Optional[AnalyticalContext],
        df: Optional[pd.DataFrame] = None,
    ) -> ContextualResolution:
        """
        Disambiguate and enrich a command using the active analytical context.
        """
        cmd_clean = command.strip()
        cmd_lower = cmd_clean.lower()

        if context is None:
            return ContextualResolution(
                user_command=cmd_clean,
                resolved_command=cmd_clean,
                detected_intent="eda",
                is_follow_up=False,
            )

        resolved_cmd = cmd_clean
        resolved_refs: Dict[str, str] = {}
        is_follow_up = False
        target = context.active_target
        features = list(context.active_features)
        time_column = context.active_time_column
        parameters: Dict[str, Any] = {}
        dataset_id = context.active_dataset_id

        active_ds = context.datasets.get(context.active_dataset_id) if context.active_dataset_id else None

        # 1. Dataset Reference & Switching: e.g. "go back to sales", "switch to dataset_2"
        switch_match = re.search(r"\b(switch to|go back to|use the)\s+([a-zA-Z0-9_\-\.]+)(\s+dataset)?\b", cmd_lower)
        if switch_match:
            requested_name = switch_match.group(2)
            for d_id, ds in context.datasets.items():
                if requested_name in d_id.lower() or requested_name in ds.dataset_name.lower() or ds.dataset_name.lower() in requested_name:
                    dataset_id = d_id
                    resolved_refs["dataset"] = ds.dataset_name
                    is_follow_up = True
                    break

        # 2. Check for Ambiguous References: e.g., "compare that with the other one"
        if ("compare that with the other" in cmd_lower or "compare it to the other" in cmd_lower) and len(context.execution_history) >= 2:
            options = [f"{h.task_type} (from turn {h.turn_id})" for h in context.execution_history[-2:]]
            return ContextualResolution(
                user_command=cmd_clean,
                resolved_command=cmd_clean,
                detected_intent="compare",
                is_follow_up=True,
                needs_clarification=True,
                clarification_reason="Multiple recent analytical results exist for comparison.",
                suggested_options=options,
            )

        # 3. Horizon Modification: e.g. "make it 12", "increase horizon to 10", "forecast next 12", "increase horizon to 8 periods"
        horizon_match = re.search(r"\b(make it|increase horizon to|set horizon to|horizon to|next)\s+(\d+)", cmd_lower)
        if horizon_match:
            new_horizon = int(horizon_match.group(2))
            parameters["periods"] = new_horizon
            parameters["horizon"] = new_horizon
            resolved_refs["horizon"] = str(new_horizon)
            is_follow_up = True
            if context.previous_task == "forecasting" or "forecast" in cmd_lower or "make it" in cmd_lower:
                resolved_cmd = f"forecast the next {new_horizon} periods for {target or 'target'}"
                return ContextualResolution(
                    user_command=cmd_clean,
                    resolved_command=resolved_cmd,
                    detected_intent="forecasting",
                    target=target,
                    time_column=time_column,
                    parameters=parameters,
                    is_follow_up=True,
                    resolved_references=resolved_refs,
                    dataset_id=dataset_id,
                )

        # 4. Target Pronoun Resolution: "predict it", "forecast it", "what predicts it", "the target", "same target", "which feature predicts it"
        if re.search(r"\b(predict it|forecast it|what predicts it|predicts it|the target|same target|for it)\b", cmd_lower) or "predicts it" in cmd_lower or "predict it" in cmd_lower or "the target" in cmd_lower:
            if target:
                resolved_refs["target"] = target
                is_follow_up = True
                resolved_cmd = re.sub(r"\b(predict it)\b", f"predict {target}", resolved_cmd, flags=re.IGNORECASE)
                resolved_cmd = re.sub(r"\b(forecast it)\b", f"forecast {target}", resolved_cmd, flags=re.IGNORECASE)
                resolved_cmd = re.sub(r"\b(what predicts it|which feature predicts it)\b", f"which feature predicts {target}", resolved_cmd, flags=re.IGNORECASE)
                resolved_cmd = re.sub(r"\b(the target|same target)\b", target, resolved_cmd, flags=re.IGNORECASE)
                resolved_cmd = re.sub(r"\bit\b", target, resolved_cmd, flags=re.IGNORECASE)
            elif active_ds and active_ds.numeric_columns:
                # If no active target was explicitly set, check if command is ambiguous
                if len(active_ds.numeric_columns) > 1 and "predict" in cmd_lower:
                    return ContextualResolution(
                        user_command=cmd_clean,
                        resolved_command=cmd_clean,
                        detected_intent="prediction",
                        is_follow_up=True,
                        needs_clarification=True,
                        clarification_reason="No target variable was previously established in context.",
                        suggested_options=active_ds.numeric_columns[:4],
                    )

        # 5. Relationship References: "the strongest relationship", "that correlation", "that relationship", "second strongest"
        if "second strongest" in cmd_lower and context.latest_second_strongest_relationship:
            rel = context.latest_second_strongest_relationship
            f1, f2 = rel.get("feature_1"), rel.get("feature_2")
            if f1 and f2:
                resolved_refs["second_strongest_relationship"] = f"{f1} and {f2}"
                features = [f1, f2]
                is_follow_up = True
                resolved_cmd = f"analyze statistical relationship between {f1} and {f2}"
        elif any(k in cmd_lower for k in ("strongest relationship", "strongest correlation", "most correlated", "that relationship", "that correlation")):
            if context.latest_strongest_relationship:
                rel = context.latest_strongest_relationship
                f1, f2 = rel.get("feature_1"), rel.get("feature_2")
                if f1 and f2:
                    resolved_refs["strongest_relationship"] = f"{f1} and {f2}"
                    features = [f1, f2]
                    is_follow_up = True
                    resolved_cmd = re.sub(
                        r"\b(the strongest relationship|the strongest correlation|that relationship|that correlation|strongest relationship)\b",
                        f"the relationship between {f1} and {f2}",
                        resolved_cmd,
                        flags=re.IGNORECASE,
                    )

        # 6. Feature References: "those features", "same features", "the features"
        if re.search(r"\b(those features|same features|the features|using those features)\b", cmd_lower) or "those features" in cmd_lower:
            if features:
                resolved_refs["features"] = ", ".join(features)
                is_follow_up = True
                resolved_cmd = re.sub(r"\b(those features|same features|the features)\b", ", ".join(features), resolved_cmd, flags=re.IGNORECASE)

        # 7. Cluster / Segment References: "cluster 2", "that cluster", "focus on cluster 3", "explain that group"
        clust_match = re.search(r"\b(cluster|segment|group)\s+(\d+)\b", cmd_lower)
        if clust_match:
            c_num = int(clust_match.group(2))
            parameters["cluster_id"] = c_num
            resolved_refs["cluster_id"] = str(c_num)
            is_follow_up = True
            resolved_cmd = f"explain and profile cluster {c_num}"

        # 8. Anomaly References: "those anomalies", "the anomalies", "the outliers", "show anomalies"
        if any(k in cmd_lower for k in ("those anomalies", "the anomalies", "the outliers", "show only anomalies", "why are they anomalies")):
            is_follow_up = True
            resolved_refs["anomalies"] = "detected outliers"
            if context.latest_anomaly_count is not None:
                resolved_refs["anomaly_count"] = str(context.latest_anomaly_count)

        # 9. Model References: "the model", "the previous model", "that model", "compare with previous model"
        if re.search(r"\b(the model|the previous model|that model|previous model)\b", cmd_lower) or "previous model" in cmd_lower:
            if context.latest_model_name:
                resolved_refs["model"] = context.latest_model_name
                is_follow_up = True
                resolved_cmd = re.sub(r"\b(the model|the previous model|that model|previous model)\b", context.latest_model_name, resolved_cmd, flags=re.IGNORECASE)

        # 10. General Deictics: "tell me more about that", "explain that", "why is that important", "show more"
        if any(k in cmd_lower for k in ("explain that", "tell me more about that", "why is that important", "show more", "why?")):
            is_follow_up = True
            if context.previous_task:
                resolved_refs["context_task"] = context.previous_task
                resolved_cmd = f"explain and synthesize findings for {context.previous_task}"

        # 11. Intent Mapping
        detected_intent = "eda"
        if any(w in cmd_lower for w in ("forecast", "future", "horizon", "periods")):
            detected_intent = "forecasting"
        elif any(w in cmd_lower for w in ("predict", "train", "model", "regression", "classification")):
            detected_intent = "prediction"
        elif any(w in cmd_lower for w in ("anomaly", "anomalies", "outlier", "outliers")):
            detected_intent = "anomaly_detection"
        elif any(w in cmd_lower for w in ("cluster", "clustering", "segment", "segmentation", "group")):
            detected_intent = "clustering"
        elif any(w in cmd_lower for w in ("correlation", "relationship", "correlated", "dependency", "association")):
            detected_intent = "statistical_analysis"
        elif any(w in cmd_lower for w in ("hypothesis", "significant", "t-test", "anova")):
            detected_intent = "hypothesis_testing"
        elif any(w in cmd_lower for w in ("clean", "transform", "impute", "preprocess")):
            detected_intent = "transformation"
        elif is_follow_up and context.previous_intent:
            detected_intent = context.previous_intent

        return ContextualResolution(
            user_command=cmd_clean,
            resolved_command=resolved_cmd,
            detected_intent=detected_intent,
            target=target,
            features=features,
            time_column=time_column,
            parameters=parameters,
            is_follow_up=is_follow_up,
            resolved_references=resolved_refs,
            dataset_id=dataset_id,
        )


# ------------------------------------------------------------------------------
# Session Context Manager
# ------------------------------------------------------------------------------

class SessionContextManager:
    """
    Thread-safe, session-isolated analytical context and session memory manager.
    Enforces bounded history, TTL expirations, and multi-dataset continuity.
    """

    def __init__(self, max_history_per_session: int = 10, ttl_seconds: int = 86400):
        self.max_history = max_history_per_session
        self.ttl_seconds = ttl_seconds
        self._contexts: Dict[str, AnalyticalContext] = {}
        # Scoped In-Memory DataFrames: (session_id, dataset_id) -> pd.DataFrame
        self._session_datasets: Dict[Tuple[str, str], pd.DataFrame] = {}
        self._lock = threading.RLock()
        self.resolver = UniversalReferenceResolver()

    # --------------------------------------------------------------------------
    # Session & Dataset Accessors
    # --------------------------------------------------------------------------

    def get_or_create_context(self, session_id: str) -> AnalyticalContext:
        """Get or initialize context for a given session."""
        with self._lock:
            if session_id not in self._contexts:
                self._contexts[session_id] = AnalyticalContext(session_id=session_id)
            ctx = self._contexts[session_id]
            ctx.last_active_at = datetime.now().isoformat()
            return ctx

    def get_context(self, session_id: str) -> Optional[AnalyticalContext]:
        """Retrieve existing context if present."""
        with self._lock:
            return self._contexts.get(session_id)

    def register_dataset(
        self,
        session_id: str,
        df: pd.DataFrame,
        dataset_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> DatasetSnapshot:
        """Register and cache a dataset for a session without storing bulk data inside context."""
        with self._lock:
            ctx = self.get_or_create_context(session_id)
            d_id = dataset_id or f"ds_{uuid.uuid4().hex[:8]}"
            d_name = dataset_name or f"dataset_{len(ctx.datasets) + 1}"

            # Ingest through CanonicalDataLayer to obtain SemanticProfile
            ingested = CanonicalDataLayer.ingest(df)
            prof = ingested.profile

            preview_records = df.head(5).to_dict(orient="records")

            total_missing = sum(m.get("count", 0) for m in prof.missing_stats.values()) if isinstance(prof.missing_stats, dict) else 0
            qs = round(1.0 - (total_missing / max(1, len(df) * max(1, len(df.columns)))), 4)

            snapshot = DatasetSnapshot(
                dataset_id=d_id,
                dataset_name=d_name,
                columns=list(df.columns),
                numeric_columns=prof.numeric_columns,
                categorical_columns=prof.categorical_columns,
                datetime_columns=prof.datetime_candidates,
                identifier_columns=prof.identifier_columns,
                constant_columns=prof.constant_columns,
                original_rows=len(df),
                current_rows=len(df),
                preview_sample=preview_records,
                quality_score=qs,
            )

            # Store in session context and cache dataframe in memory scoped to (session_id, d_id)
            ctx.datasets[d_id] = snapshot
            ctx.active_dataset_id = d_id
            if prof.datetime_candidates and not ctx.active_time_column:
                ctx.active_time_column = prof.datetime_candidates[0]

            self._session_datasets[(session_id, d_id)] = df.copy()
            return snapshot

    def get_dataset(self, session_id: str, dataset_id: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Retrieve cached DataFrame for a session."""
        with self._lock:
            ctx = self.get_context(session_id)
            if not ctx:
                return None
            target_id = dataset_id or ctx.active_dataset_id
            if not target_id:
                return None
            return self._session_datasets.get((session_id, target_id))

    def switch_dataset(self, session_id: str, dataset_id: str) -> bool:
        """Switch active dataset within a session and reset target/feature pointers."""
        with self._lock:
            ctx = self.get_context(session_id)
            if not ctx or dataset_id not in ctx.datasets:
                return False
            ctx.active_dataset_id = dataset_id
            # Reset target/features on dataset switch to prevent cross-dataset contamination
            ctx.active_target = None
            ctx.active_features = []
            ds = ctx.datasets[dataset_id]
            ctx.active_time_column = ds.datetime_columns[0] if ds.datetime_columns else None
            return True

    # --------------------------------------------------------------------------
    # Record Analytical Execution & Update Memory
    # --------------------------------------------------------------------------

    def record_execution(
        self,
        session_id: str,
        result: AgentResult,
        user_command: str,
        resolved_command: Optional[str] = None,
        target: Optional[str] = None,
        features: Optional[List[str]] = None,
        time_column: Optional[str] = None,
        resolved_references: Optional[Dict[str, str]] = None,
    ) -> None:
        """Update analytical context after an execution."""
        with self._lock:
            ctx = self.get_or_create_context(session_id)
            turn_id = len(ctx.execution_history) + 1

            # Extract task details
            task_type = result.task_type or "orchestration"
            status_str = result.status.value if isinstance(result.status, AgentStatus) else str(result.status)
            res_data = result.data if isinstance(result.data, dict) else (result.result if isinstance(result.result, dict) else {})

            # Update active pointers
            if target:
                ctx.active_target = target
            if features:
                ctx.active_features = list(features)
            if time_column:
                ctx.active_time_column = time_column

            ctx.previous_task = ctx.active_task
            ctx.active_task = task_type
            ctx.last_execution_id = result.execution_id
            ctx.latest_confidence = float(result.confidence)
            ctx.latest_metrics = result.metrics or {}
            ctx.assumptions = result.assumptions or []
            ctx.limitations = result.limitations or []
            ctx.warnings = result.warnings or []

            # Extract specific analytical state from result payload
            if "tasks" in res_data:
                tasks_dict = res_data["tasks"]
                # Forecast state
                if "forecasting" in tasks_dict and isinstance(tasks_dict["forecasting"], dict):
                    fc = tasks_dict["forecasting"]
                    ctx.latest_forecast_horizon = fc.get("horizon", 6)
                    ctx.latest_model_name = fc.get("model_selected") or fc.get("model_name")
                # Prediction state
                if "prediction" in tasks_dict and isinstance(tasks_dict["prediction"], dict):
                    pred = tasks_dict["prediction"]
                    ctx.latest_model_name = pred.get("selected_model") or pred.get("best_model")
                # Anomaly state
                if "anomaly_detection" in tasks_dict and isinstance(tasks_dict["anomaly_detection"], dict):
                    anom = tasks_dict["anomaly_detection"]
                    ctx.latest_anomaly_count = anom.get("anomaly_count", 0)
                # Clustering state
                if "clustering" in tasks_dict and isinstance(tasks_dict["clustering"], dict):
                    clust = tasks_dict["clustering"]
                    ctx.latest_cluster_count = clust.get("cluster_count", clust.get("n_clusters"))
                # Relationships state
                if "statistical_analysis" in tasks_dict and isinstance(tasks_dict["statistical_analysis"], dict):
                    stats = tasks_dict["statistical_analysis"]
                    ranked = stats.get("ranked_relationships") or stats.get("relationships", [])
                    if len(ranked) >= 1 and isinstance(ranked[0], dict):
                        ctx.latest_strongest_relationship = ranked[0]
                    if len(ranked) >= 2 and isinstance(ranked[1], dict):
                        ctx.latest_second_strongest_relationship = ranked[1]

            # Bounded execution record
            top_findings = []
            if "synthesis" in res_data and isinstance(res_data["synthesis"], dict):
                top_findings = res_data["synthesis"].get("important_findings", [])

            record = ExecutionRecord(
                execution_id=result.execution_id,
                turn_id=turn_id,
                user_command=user_command,
                resolved_command=resolved_command or user_command,
                task_type=task_type,
                intent=res_data.get("detected_intent", task_type),
                target=ctx.active_target,
                features=ctx.active_features,
                time_column=ctx.active_time_column,
                model_selected=ctx.latest_model_name,
                metrics=result.metrics or {},
                confidence=float(result.confidence),
                status=status_str,
                summary=res_data.get("summary", ""),
                top_findings=top_findings[:3],
                resolved_references=resolved_references or {},
            )

            ctx.execution_history.append(record)
            # Enforce bounded history
            if len(ctx.execution_history) > self.max_history:
                ctx.execution_history = ctx.execution_history[-self.max_history:]

    # --------------------------------------------------------------------------
    # Context Invalidation & Cleanup
    # --------------------------------------------------------------------------

    def invalidate_target(self, session_id: str) -> None:
        with self._lock:
            ctx = self.get_context(session_id)
            if ctx:
                ctx.active_target = None

    def invalidate_features(self, session_id: str) -> None:
        with self._lock:
            ctx = self.get_context(session_id)
            if ctx:
                ctx.active_features = []

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._contexts:
                del self._contexts[session_id]
            # Remove cached dataframes for this session
            keys_to_del = [k for k in self._session_datasets.keys() if k[0] == session_id]
            for k in keys_to_del:
                del self._session_datasets[k]


# Global singleton instance
DEFAULT_SESSION_CONTEXT_MANAGER = SessionContextManager()