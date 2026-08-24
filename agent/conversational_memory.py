"""Conversational Memory & Multi-Turn Context Resolution Engine.

Maintains stateful conversational memory across analysis turns, resolves anaphora /
pronouns ("it", "those", "that metric", "the previous model"), tracks active datasets,
dimensions, metrics, and models, and provides contextual disambiguation for natural
language queries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class ConversationTurn:
    """A single turn in the conversational data analysis session."""
    turn_id: int
    user_command: str
    resolved_command: str
    intent: str
    active_metric: Optional[str]
    active_dimension: Optional[str]
    active_target: Optional[str]
    active_model_type: Optional[str]
    summary_findings: List[str]
    evidence_count: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "user_command": self.user_command,
            "resolved_command": self.resolved_command,
            "intent": self.intent,
            "active_metric": self.active_metric,
            "active_dimension": self.active_dimension,
            "active_target": self.active_target,
            "active_model_type": self.active_model_type,
            "summary_findings": self.summary_findings,
            "evidence_count": self.evidence_count,
            "timestamp": self.timestamp,
        }


@dataclass
class SessionState:
    """Active state for a multi-turn user analysis session."""
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)
    active_dataset_name: Optional[str] = None
    active_metric: Optional[str] = None
    active_dimension: Optional[str] = None
    active_target: Optional[str] = None
    active_date_column: Optional[str] = None
    last_model_id: Optional[str] = None
    last_model_type: Optional[str] = None
    last_intent: Optional[str] = None
    last_entities: List[str] = field(default_factory=list)
    turns: List[ConversationTurn] = field(default_factory=list)

    def update_activity(self):
        self.last_active_at = time.time()


class ConversationalMemoryEngine:
    """Manages multi-turn conversation sessions and contextual command resolution."""

    # Anaphora and relative reference patterns
    PRONOUN_PATTERNS = [
        (re.compile(r"\b(why did it (decrease|drop|fall|increase|rise|change))\b", re.I), "metric"),
        (re.compile(r"\b(how is it trending|trend of it|show it over time)\b", re.I), "metric"),
        (re.compile(r"\b(compare it (with|to|against))\b", re.I), "metric"),
        (re.compile(r"\b(predict it|forecast it)\b", re.I), "metric"),
        (re.compile(r"\b(train (a |the )?model (on|for) it)\b", re.I), "target"),
        (re.compile(r"\b(what drives it|key drivers of it)\b", re.I), "metric"),
        (re.compile(r"\b(show (top|bottom) \d+ of (those|them|these))\b", re.I), "dimension"),
        (re.compile(r"\b(filter (by|for) (that|those|them))\b", re.I), "dimension"),
        (re.compile(r"\b(explain (its|the) (features|coefficients|weights|performance|accuracy))\b", re.I), "model"),
        (re.compile(r"\b(why\??$|why did that happen\??)\b", re.I), "metric"),
    ]

    def __init__(self, ttl_seconds: int = 86400):
        self.ttl_seconds = ttl_seconds
        self._sessions: Dict[str, SessionState] = {}

    def get_or_create_session(self, session_id: str) -> SessionState:
        """Retrieve existing session state or initialize a fresh one."""
        self._cleanup_expired()
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
        session = self._sessions[session_id]
        session.update_activity()
        return session

    def resolve_context(
        self,
        command: str,
        session_id: str,
        df: Optional[pd.DataFrame] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Disambiguate natural language commands using conversational session state.
        
        Returns:
            Tuple of (resolved_command_string, context_metadata_dict)
        """
        session = self.get_or_create_session(session_id)
        cmd_clean = command.strip()
        resolved = cmd_clean
        context_used = {}

        # If user refers to prior context
        has_pronoun = any(
            p.search(cmd_clean) for p, _ in self.PRONOUN_PATTERNS
        ) or re.search(r"\b(it|its|those|these|that metric|previous model|last model)\b", cmd_clean, re.I)

        # 1. Resolve Metric References ("it", "that metric")
        if session.active_metric:
            # Replace "why did it fall/decrease/rise" -> "why did <metric> fall/decrease/rise"
            resolved = re.sub(
                r"\bwhy did it\b",
                f"why did {session.active_metric}",
                resolved,
                flags=re.I
            )
            # Replace "compare it" -> "compare <metric>"
            resolved = re.sub(
                r"\bcompare it\b",
                f"compare {session.active_metric}",
                resolved,
                flags=re.I
            )
            # Replace "forecast it" / "predict it" -> "forecast <metric>"
            resolved = re.sub(
                r"\b(forecast|predict) it\b",
                f"\\1 {session.active_metric}",
                resolved,
                flags=re.I
            )
            # Replace "what drives it" / "drivers of it"
            resolved = re.sub(
                r"\b(drivers of|what drives) it\b",
                f"\\1 {session.active_metric}",
                resolved,
                flags=re.I
            )
            # Replace standalone "why?" -> "why did <metric> change?"
            if re.match(r"^why\??$", cmd_clean.strip(), re.I):
                resolved = f"why did {session.active_metric} change?"

            if resolved != cmd_clean:
                context_used["resolved_metric"] = session.active_metric

        # 2. Resolve ML Target References ("train model on it", "build model for it")
        target_candidate = session.active_target or session.active_metric
        if target_candidate:
            if re.search(r"\b(train|build|create) (a |the )?(best )?model (on|for) it\b", resolved, re.I):
                resolved = re.sub(
                    r"\b(train|build|create) (a |the )?(best )?model (on|for) it\b",
                    f"build the best model to predict {target_candidate}",
                    resolved,
                    flags=re.I
                )
                context_used["resolved_target"] = target_candidate

        # 3. Resolve Category / Dimension References ("show top 5 of those")
        if session.active_dimension:
            if re.search(r"\b(those|them|these)\b", resolved, re.I):
                resolved = re.sub(
                    r"\b(of|for) (those|them|these)\b",
                    f"by {session.active_dimension}",
                    resolved,
                    flags=re.I
                )
                context_used["resolved_dimension"] = session.active_dimension

        # 4. Resolve Model Explanations ("explain its features", "explain previous model")
        if session.last_model_type:
            if re.search(r"\b(its|the|previous|last) model\b", resolved, re.I) or re.search(r"\bexplain its (features|performance|accuracy)\b", resolved, re.I):
                context_used["resolved_model"] = session.last_model_type

        # 5. Extract and infer column context if df provided
        if df is not None and not df.empty:
            self._introspect_dataframe_context(session, df)

        context_metadata = {
            "session_id": session_id,
            "turn_count": len(session.turns),
            "context_modified": resolved != cmd_clean,
            "original_command": cmd_clean,
            "resolved_command": resolved,
            "active_state": {
                "active_metric": session.active_metric,
                "active_dimension": session.active_dimension,
                "active_target": session.active_target,
                "last_intent": session.last_intent,
                "last_model_type": session.last_model_type,
            },
            "context_used": context_used,
        }

        return resolved, context_metadata

    def record_turn(
        self,
        session_id: str,
        user_command: str,
        resolved_command: str,
        intent: str,
        active_metric: Optional[str] = None,
        active_dimension: Optional[str] = None,
        active_target: Optional[str] = None,
        active_model_type: Optional[str] = None,
        summary_findings: Optional[List[str]] = None,
        evidence_count: int = 0
    ) -> ConversationTurn:
        """Record the outcome of an analytical turn into conversational memory."""
        session = self.get_or_create_session(session_id)
        turn_id = len(session.turns) + 1

        # Update active state
        if active_metric:
            session.active_metric = active_metric
        if active_dimension:
            session.active_dimension = active_dimension
        if active_target:
            session.active_target = active_target
        if active_model_type:
            session.last_model_type = active_model_type
        if intent:
            session.last_intent = intent

        turn = ConversationTurn(
            turn_id=turn_id,
            user_command=user_command,
            resolved_command=resolved_command,
            intent=intent,
            active_metric=session.active_metric,
            active_dimension=session.active_dimension,
            active_target=session.active_target,
            active_model_type=session.last_model_type,
            summary_findings=summary_findings or [],
            evidence_count=evidence_count,
        )

        session.turns.append(turn)
        session.update_activity()
        return turn

    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve chronological history of turns for a session."""
        session = self._sessions.get(session_id)
        if not session:
            return []
        return [t.to_dict() for t in session.turns]

    def clear_session(self, session_id: str):
        """Clear memory for a specific session."""
        if session_id in self._sessions:
            del self._sessions[session_id]

    def _introspect_dataframe_context(self, session: SessionState, df: pd.DataFrame):
        """Infer probable active metrics and dimensions from DataFrame."""
        if not session.active_metric:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Prioritize standard financial/volume metrics
            for col in num_cols:
                if any(m in col.lower() for m in ["profit", "revenue", "sales", "price", "amount", "total"]):
                    session.active_metric = col
                    break
            if not session.active_metric and num_cols:
                session.active_metric = num_cols[0]

        if not session.active_dimension:
            cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
            for col in cat_cols:
                if any(d in col.lower() for d in ["country", "region", "category", "segment", "type", "state"]):
                    session.active_dimension = col
                    break
            if not session.active_dimension and cat_cols:
                session.active_dimension = cat_cols[0]

    def _cleanup_expired(self):
        """Evict sessions that have exceeded TTL."""
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if (now - s.last_active_at) > self.ttl_seconds
        ]
        for sid in expired:
            del self._sessions[sid]


# Global singleton instance for app-wide conversational memory
global_conversational_memory = ConversationalMemoryEngine()

