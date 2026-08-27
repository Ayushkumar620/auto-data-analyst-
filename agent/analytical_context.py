"""
Universal Conversational Analytical Context & Session Memory Layer.
"""
from __future__ import annotations

import json
import math
import re
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from agent.agent_result import AgentError, AgentResult, AgentStatus, ClaimType, ErrorCategory, Evidence
from agent.canonical_data_layer import CanonicalDataLayer, SemanticProfile


class DatasetSnapshot(BaseModel):
    """Metadata-only footprint of a registered session dataset."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

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
    preview_sample: List[Dict[str, Any]] = Field(default_factory=list)
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
            "preview_sample": self.preview_sample[:5],
            "quality_score": round(float(self.quality_score), 4),
        }


class ExecutionHistoryItem(BaseModel):
    """Immutable audit record of an executed analytical turn."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    turn_id: int
    execution_id: str
    user_command: str
    resolved_command: str
    task_type: str
    status: str
    target: Optional[str] = None
    features: List[str] = Field(default_factory=list)
    time_column: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    model_name: Optional[str] = None
    cluster_count: Optional[int] = None
    anomaly_count: Optional[int] = None
    strongest_relationship: Optional[Dict[str, Any]] = None
    second_strongest_relationship: Optional[Dict[str, Any]] = None
    dataset_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "execution_id": self.execution_id,
            "user_command": self.user_command,
            "resolved_command": self.resolved_command,
            "task_type": self.task_type,
            "status": self.status,
            "target": self.target,
            "features": self.features,
            "time_column": self.time_column,
            "metrics": self.metrics,
            "model_name": self.model_name,
            "cluster_count": self.cluster_count,
            "anomaly_count": self.anomaly_count,
            "strongest_relationship": self.strongest_relationship,
            "second_strongest_relationship": self.second_strongest_relationship,
            "dataset_id": self.dataset_id,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
        }


ExecutionRecord = ExecutionHistoryItem


class SessionContext(BaseModel):
    """Session state carrying active entities, turn history, and reference memory."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str
    active_dataset_id: Optional[str] = None
    active_target: Optional[str] = None
    active_features: List[str] = Field(default_factory=list)
    active_time_column: Optional[str] = None
    datasets: Dict[str, DatasetSnapshot] = Field(default_factory=dict)
    execution_history: List[ExecutionHistoryItem] = Field(default_factory=list)
    previous_task: Optional[str] = None
    previous_intent: Optional[str] = None
    latest_model_name: Optional[str] = None
    latest_strongest_relationship: Optional[Dict[str, Any]] = None
    latest_second_strongest_relationship: Optional[Dict[str, Any]] = None
    latest_cluster_count: Optional[int] = None
    latest_anomaly_count: Optional[int] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def active_task(self) -> Optional[str]:
        return self.previous_task

    @property
    def last_execution_id(self) -> Optional[str]:
        return self.execution_history[-1].execution_id if self.execution_history else None

    @property
    def last_result(self) -> Optional[Dict[str, Any]]:
        return self.execution_history[-1].to_dict() if self.execution_history else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "active_dataset_id": self.active_dataset_id,
            "active_target": self.active_target,
            "active_features": self.active_features,
            "active_time_column": self.active_time_column,
            "datasets": {k: v.to_dict() for k, v in self.datasets.items()},
            "execution_history": [h.to_dict() for h in self.execution_history],
            "previous_task": self.previous_task,
            "previous_intent": self.previous_intent,
            "latest_model_name": self.latest_model_name,
            "latest_strongest_relationship": self.latest_strongest_relationship,
            "latest_second_strongest_relationship": self.latest_second_strongest_relationship,
            "latest_cluster_count": self.latest_cluster_count,
            "latest_anomaly_count": self.latest_anomaly_count,
            "turn_count": len(self.execution_history),
            "last_execution_id": self.last_execution_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


AnalyticalContext = SessionContext


class ContextualResolution(BaseModel):
    """Result of context-aware follow-up command resolution."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_command": self.user_command,
            "resolved_command": self.resolved_command,
            "detected_intent": self.detected_intent,
            "target": self.target,
            "features": self.features,
            "time_column": self.time_column,
            "parameters": self.parameters,
            "is_follow_up": self.is_follow_up,
            "needs_clarification": self.needs_clarification,
            "clarification_reason": self.clarification_reason,
            "suggested_options": self.suggested_options,
            "resolved_references": self.resolved_references,
            "dataset_id": self.dataset_id,
        }


class UniversalReferenceResolver:
    """Deterministic, domain-aware resolver for conversational analytical follow-up commands."""

    def resolve(
        self,
        command: str,
        context: Optional[SessionContext],
        df: Optional[pd.DataFrame] = None,
    ) -> ContextualResolution:
        cmd_clean = command.strip()
        cmd_lower = cmd_clean.lower()

        if not context:
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

        # 1. Dataset Reference & Switching: e.g. "go back to sales", "switch to dataset_2", "switch to sales_data dataset"
        switch_match = re.search(r"(?:switch to|go back to|use the)\s+([a-zA-Z0-9_\-\.]+)(?:\s+dataset)?", cmd_lower)
        if switch_match:
            requested_name = switch_match.group(1).lower()
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

        # 3. Horizon Modification: e.g. "make it 12", "increase horizon to 8 periods", "forecast next 12"
        horizon_match = re.search(r"(?:make it|increase horizon to|set horizon to|horizon to|next)\s+(\d+)", cmd_lower)
        if horizon_match:
            new_horizon = int(horizon_match.group(1))
            parameters["periods"] = new_horizon
            parameters["horizon"] = new_horizon
            resolved_refs["horizon"] = str(new_horizon)
            is_follow_up = True
            if context.previous_task == "forecasting" or "forecast" in cmd_lower or "make it" in cmd_lower or "horizon" in cmd_lower:
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

        # 4. Target Pronoun Resolution: "predict it", "forecast it", "which feature predicts it", "the target", "same target"
        if re.search(r"\b(predicts? it|forecasts? it|what predicts it|which feature predicts it|which variables influence the target|the target|same target|for it)\b", cmd_lower):
            if target:
                resolved_refs["target"] = target
                is_follow_up = True
                resolved_cmd = re.sub(r"\bpredicts? it\b", f"predicts {target}", resolved_cmd, flags=re.IGNORECASE)
                resolved_cmd = re.sub(r"\bforecasts? it\b", f"forecasts {target}", resolved_cmd, flags=re.IGNORECASE)
                resolved_cmd = re.sub(r"\bwhat predicts it\b", f"which feature predicts {target}", resolved_cmd, flags=re.IGNORECASE)
                resolved_cmd = re.sub(r"\bwhich feature predicts it\b", f"which feature predicts {target}", resolved_cmd, flags=re.IGNORECASE)
                resolved_cmd = re.sub(r"\bwhich variables influence the target\b", f"which variables influence {target}", resolved_cmd, flags=re.IGNORECASE)
                resolved_cmd = re.sub(r"\b(the target|same target)\b", target, resolved_cmd, flags=re.IGNORECASE)
                if "forecast it" in cmd_lower:
                    resolved_cmd = f"forecast {target}"
            elif active_ds and active_ds.numeric_columns:
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
                resolved_cmd = re.sub(
                    r"\b(the second strongest relationship|that relationship|that correlation|second strongest relationship)\b",
                    f"the relationship between {f1} and {f2}",
                    resolved_cmd,
                    flags=re.IGNORECASE,
                )
                if f1 not in resolved_cmd:
                    resolved_cmd = f"{resolved_cmd} between {f1} and {f2}"
        elif any(k in cmd_lower for k in ("strongest relationship", "strongest correlation", "most correlated", "that relationship", "that correlation")):
            is_follow_up = True
            if context.latest_strongest_relationship:
                rel = context.latest_strongest_relationship
                f1, f2 = rel.get("feature_1"), rel.get("feature_2")
                if f1 and f2:
                    resolved_refs["strongest_relationship"] = f"{f1} and {f2}"
                    features = [f1, f2]
                    resolved_cmd = re.sub(
                        r"\b(the strongest relationship|the strongest correlation|that relationship|that correlation|strongest relationship)\b",
                        f"the relationship between {f1} and {f2}",
                        resolved_cmd,
                        flags=re.IGNORECASE,
                    )
                    if f1 not in resolved_cmd:
                        resolved_cmd = f"{resolved_cmd} between {f1} and {f2}"

        # 6. Feature References: "those features", "same features", "the features"
        if re.search(r"\b(those features|same features|the features|using those features|with those features)\b", cmd_lower):
            if features:
                feat_str = ", ".join(features)
                resolved_refs["features"] = feat_str
                is_follow_up = True
                resolved_cmd = re.sub(r"\b(those features|same features|the features)\b", feat_str, resolved_cmd, flags=re.IGNORECASE)

        # 7. Cluster / Segment References: "cluster 2", "that cluster", "focus on cluster 3", "explain that group"
        clust_match = re.search(r"\b(?:cluster|segment|group)\s+(\d+)\b", cmd_lower)
        if clust_match:
            c_num = int(clust_match.group(1))
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
        if re.search(r"\b(the model|the previous model|that model|previous model)\b", cmd_lower):
            if context.latest_model_name:
                resolved_refs["model"] = context.latest_model_name
                is_follow_up = True
                resolved_cmd = re.sub(r"\b(the model|the previous model|that model|previous model)\b", context.latest_model_name, resolved_cmd, flags=re.IGNORECASE)

        # 10. General Deictics: "tell me more about that", "explain that", "why is that important", "show more"
        if any(k in cmd_lower for k in ("explain that", "tell me more about that", "why is that important", "show more", "why?")):
            is_follow_up = True
            if context.previous_task:
                resolved_refs["context_task"] = context.previous_task
                if context.previous_task not in resolved_cmd:
                    resolved_cmd = f"tell me more about {context.previous_task}"

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
        elif any(w in cmd_lower for w in ("explain", "explanation", "methodology", "how was this calculated", "show evidence", "why did you", "why is this")):
            detected_intent = "explanation"
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


class SessionContextManager:
    """Thread-safe state manager for conversational analytical sessions."""

    def __init__(self, max_history_turns: int = 50, max_history_per_session: Optional[int] = None):
        self._contexts: Dict[str, SessionContext] = {}
        self._dataset_store: Dict[Tuple[str, str], pd.DataFrame] = {}
        self._lock = threading.RLock()
        self.max_history_turns = max_history_per_session or max_history_turns
        self.resolver = UniversalReferenceResolver()

    def get_or_create_context(self, session_id: str) -> SessionContext:
        with self._lock:
            if session_id not in self._contexts:
                self._contexts[session_id] = SessionContext(session_id=session_id)
            return self._contexts[session_id]

    def get_context(self, session_id: str) -> Optional[SessionContext]:
        with self._lock:
            return self._contexts.get(session_id)

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

    def register_dataset(
        self,
        session_id: str,
        df: pd.DataFrame,
        dataset_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> DatasetSnapshot:
        with self._lock:
            ctx = self.get_or_create_context(session_id)
            d_id = dataset_id or f"ds_{uuid.uuid4().hex[:8]}"
            d_name = dataset_name or f"dataset_{len(ctx.datasets) + 1}"

            ingested = CanonicalDataLayer.ingest(df)
            prof = ingested.profile

            preview_records = df.head(5).to_dict(orient="records")
            total_cells = max(1, len(df) * max(1, len(df.columns)))
            total_missing = sum(m.get("missing_count", 0) for m in prof.missing_stats.values()) if isinstance(prof.missing_stats, dict) else 0
            q_score = round(max(0.0, min(1.0, 1.0 - (total_missing / total_cells))), 4)

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
                quality_score=q_score,
            )

            ctx.datasets[d_id] = snapshot
            ctx.active_dataset_id = d_id
            if prof.datetime_candidates and not ctx.active_time_column:
                ctx.active_time_column = prof.datetime_candidates[0]

            self._dataset_store[(session_id, d_id)] = df.copy(deep=True)
            ctx.updated_at = datetime.utcnow().isoformat()
            return snapshot

    def get_dataset(self, session_id: str, dataset_id: Optional[str] = None) -> Optional[pd.DataFrame]:
        with self._lock:
            ctx = self.get_context(session_id)
            if not ctx:
                return None
            target_d_id = dataset_id or ctx.active_dataset_id
            if not target_d_id:
                return None
            df = self._dataset_store.get((session_id, target_d_id))
            return df.copy(deep=True) if df is not None else None

    def record_execution(
        self,
        session_id: str,
        result: AgentResult,
        user_command: str = "",
        resolved_command: Optional[str] = None,
        target: Optional[str] = None,
        features: Optional[List[str]] = None,
        time_column: Optional[str] = None,
        resolved_references: Optional[Dict[str, str]] = None,
    ) -> ExecutionHistoryItem:
        with self._lock:
            ctx = self.get_or_create_context(session_id)
            turn_id = len(ctx.execution_history) + 1
            eff_cmd = resolved_command or user_command

            t_val = target or result.target or (result.data.get("target") if isinstance(result.data, dict) else None)
            f_val = features or (result.data.get("features") if isinstance(result.data, dict) else [])
            time_col = time_column or (result.data.get("time_column") if isinstance(result.data, dict) else None)

            if result.is_success:
                if t_val:
                    ctx.active_target = str(t_val)
                if f_val:
                    ctx.active_features = list(f_val)
                if time_col:
                    ctx.active_time_column = str(time_col)

                ctx.previous_task = result.task_type
                ctx.previous_intent = result.task_type

                model_name = None
                clust_count = None
                anom_count = None
                r1 = None
                r2 = None

                # Extract from data, result, output
                for container in (result.data, result.result, result.output):
                    if isinstance(container, dict):
                        if not model_name:
                            model_name = container.get("model_name") or container.get("model_selected")
                        if clust_count is None:
                            clust_count = container.get("cluster_count") or container.get("tasks", {}).get("clustering", {}).get("cluster_count")
                        if anom_count is None:
                            anom_count = container.get("anomaly_count") or container.get("tasks", {}).get("anomaly_detection", {}).get("anomaly_count")
                        if not r1:
                            ranked_rels = container.get("ranked_relationships") or container.get("relationships") or container.get("tasks", {}).get("statistical_analysis", {}).get("ranked_relationships") or []
                            if ranked_rels and len(ranked_rels) >= 1:
                                r1 = ranked_rels[0] if isinstance(ranked_rels[0], dict) else None
                            if ranked_rels and len(ranked_rels) >= 2:
                                r2 = ranked_rels[1] if isinstance(ranked_rels[1], dict) else None

                if model_name:
                    ctx.latest_model_name = str(model_name)
                if clust_count is not None:
                    ctx.latest_cluster_count = int(clust_count)
                if anom_count is not None:
                    ctx.latest_anomaly_count = int(anom_count)
                if r1:
                    ctx.latest_strongest_relationship = r1
                if r2:
                    ctx.latest_second_strongest_relationship = r2

            hist_item = ExecutionHistoryItem(
                turn_id=turn_id,
                execution_id=result.execution_id or f"exec_{uuid.uuid4().hex[:8]}",
                user_command=user_command,
                resolved_command=eff_cmd,
                task_type=result.task_type or "orchestration",
                status=result.status.value if hasattr(result.status, "value") else str(result.status),
                target=ctx.active_target,
                features=ctx.active_features,
                time_column=ctx.active_time_column,
                metrics=result.metrics or {},
                model_name=ctx.latest_model_name,
                cluster_count=ctx.latest_cluster_count,
                anomaly_count=ctx.latest_anomaly_count,
                strongest_relationship=ctx.latest_strongest_relationship,
                second_strongest_relationship=ctx.latest_second_strongest_relationship,
                dataset_id=ctx.active_dataset_id,
                duration_ms=result.duration_ms,
            )

            ctx.execution_history.append(hist_item)
            if len(ctx.execution_history) > self.max_history_turns:
                ctx.execution_history = ctx.execution_history[-self.max_history_turns:]

            ctx.updated_at = datetime.utcnow().isoformat()
            return hist_item

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._contexts:
                del self._contexts[session_id]
            for (s_id, d_id) in list(self._dataset_store.keys()):
                if s_id == session_id:
                    del self._dataset_store[(s_id, d_id)]


# Global default session manager singleton
DEFAULT_SESSION_CONTEXT_MANAGER = SessionContextManager()