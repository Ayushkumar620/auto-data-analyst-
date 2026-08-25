"""
Schemas for Conversational Analyst and Evidence-Based Report Generation.

Provides Pydantic models for:
- ReportType & ConversationalIntent
- ConversationTurn & DatasetContext
- ConversationSession & ConversationSummary
- ReportSection & GeneratedReport
"""
from __future__ import annotations

from enum import Enum
import time
from typing import Any, Dict, List, Optional, Union
import uuid

from pydantic import BaseModel, ConfigDict, Field

from agent.autonomous_analysis_schemas import Insight
from agent.schemas import ClaimType, Evidence


# ==============================================================================
# 1. Enums
# ==============================================================================

class ReportType(str, Enum):
    """Type / audience of the generated analytical report."""
    QUICK_SUMMARY = "quick_summary"        # Concise 1-page overview with top findings
    ANALYST_REPORT = "analyst_report"      # Full detailed multi-section analytical report
    EXECUTIVE_REPORT = "executive_report"  # High-level business strategy briefing
    TECHNICAL_REPORT = "technical_report"  # ML/statistical deep dive with architecture & validation


class ConversationalIntent(str, Enum):
    """Analytical intent behind a conversational user turn."""
    ANALYZE = "analyze"
    EXPLAIN = "explain"
    COMPARE = "compare"
    FILTER = "filter"
    SUMMARIZE = "summarize"
    DRILL_DOWN = "drill_down"
    INVESTIGATE = "investigate"
    PREDICT = "predict"
    FORECAST = "forecast"
    DETECT_ANOMALY = "detect_anomaly"
    MONITOR = "monitor"
    RECOMMEND = "recommend"
    GENERATE_REPORT = "generate_report"
    CLARIFICATION = "clarification"


# ==============================================================================
# 2. Session & Turn Models
# ==============================================================================

class DatasetContext(BaseModel):
    """Lightweight metadata describing the active session dataset."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_id: str = Field(default_factory=lambda: f"ds_{uuid.uuid4().hex[:8]}")
    dataset_name: str = "active_dataset"
    row_count: int = 0
    column_count: int = 0
    numeric_columns: List[str] = Field(default_factory=list)
    categorical_columns: List[str] = Field(default_factory=list)
    date_columns: List[str] = Field(default_factory=list)
    primary_metric: Optional[str] = None
    primary_dimension: Optional[str] = None
    primary_target: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class ConversationTurn(BaseModel):
    """A single traceable turn in the analytical conversation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    turn_id: str = Field(default_factory=lambda: f"turn_{uuid.uuid4().hex[:8]}")
    session_id: str
    user_message: str
    resolved_intent: ConversationalIntent = ConversationalIntent.ANALYZE
    referenced_entities: List[str] = Field(default_factory=list)
    execution_plan: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    evidence: List[Evidence] = Field(default_factory=list)
    assistant_response: str = ""
    timestamp: float = Field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "user_message": self.user_message,
            "resolved_intent": self.resolved_intent.value,
            "referenced_entities": self.referenced_entities,
            "execution_plan": self.execution_plan,
            "result": self.result,
            "evidence": [e.to_dict() for e in self.evidence],
            "assistant_response": self.assistant_response,
            "timestamp": round(self.timestamp, 3),
        }


class ConversationSession(BaseModel):
    """Stateful context for a multi-turn user analysis session."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str = Field(default_factory=lambda: f"sess_{uuid.uuid4().hex[:10]}")
    user_id: Optional[str] = None
    dataset_context: Optional[DatasetContext] = None
    active_dataset: Optional[Any] = None  # In-memory DataFrame or reference
    active_analysis: Optional[str] = None
    active_model: Optional[str] = None
    previous_results: List[Dict[str, Any]] = Field(default_factory=list)
    previous_insights: List[Insight] = Field(default_factory=list)
    previous_questions: List[str] = Field(default_factory=list)
    turns: List[ConversationTurn] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    def update_timestamp(self):
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "dataset_context": self.dataset_context.to_dict() if self.dataset_context else None,
            "active_analysis": self.active_analysis,
            "active_model": self.active_model,
            "turns_count": len(self.turns),
            "insights_count": len(self.previous_insights),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class ConversationSummary(BaseModel):
    """Condensed summary of analytical decisions and key findings."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str
    active_dataset: Optional[str] = None
    user_objective: Optional[str] = None
    important_findings: List[str] = Field(default_factory=list)
    important_metrics: Dict[str, Any] = Field(default_factory=dict)
    current_model: Optional[str] = None
    unresolved_questions: List[str] = Field(default_factory=list)
    important_decisions: List[str] = Field(default_factory=list)


# ==============================================================================
# 3. Evidence-Based Report Models
# ==============================================================================

class ReportSection(BaseModel):
    """A structured chapter or subsection of an executive report."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    section_id: str = Field(default_factory=lambda: f"sec_{uuid.uuid4().hex[:6]}")
    title: str
    content: str
    evidence_refs: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    subsections: List[ReportSection] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_id": self.section_id,
            "title": self.title,
            "content": self.content,
            "evidence_refs": self.evidence_refs,
            "metrics": self.metrics,
            "subsections": [s.to_dict() for s in self.subsections],
        }


class GeneratedReport(BaseModel):
    """Complete multi-section analytical report with verified evidence chains."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    report_id: str = Field(default_factory=lambda: f"rep_{uuid.uuid4().hex[:8]}")
    title: str
    report_type: ReportType = ReportType.ANALYST_REPORT
    executive_summary: str
    sections: List[ReportSection] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    markdown_content: str = ""
    created_at: float = Field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "title": self.title,
            "report_type": self.report_type.value,
            "executive_summary": self.executive_summary,
            "sections": [s.to_dict() for s in self.sections],
            "recommendations": self.recommendations,
            "limitations": self.limitations,
            "evidence": [e.to_dict() for e in self.evidence],
            "markdown_content": self.markdown_content,
            "created_at": round(self.created_at, 3),
        }
