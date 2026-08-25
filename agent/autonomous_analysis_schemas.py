"""
Schemas for Autonomous Data Analysis and Insight Generation Engine.

Provides standardized Pydantic models for:
- AnalysisDepth & InsightCategory & InsightSeverity
- AnalysisCandidate
- AutonomousAnalysisRequest
- Insight
- AutonomousAnalysisResult
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import uuid

from pydantic import BaseModel, ConfigDict, Field

from agent.intent import UserIntent
from agent.schemas import ClaimType, Evidence


# ==============================================================================
# 1. Enums & Categories
# ==============================================================================

class AnalysisDepth(str, Enum):
    """Depth of autonomous analytical exploration."""
    QUICK = "quick"          # Top 3-4 high-level analyses
    STANDARD = "standard"    # Top 6-8 comprehensive analyses (default)
    DEEP = "deep"            # Full exploration (up to max_analysis_steps)


class InsightCategory(str, Enum):
    """Domain category of the generated analytical finding."""
    TREND = "trend"
    ANOMALY = "anomaly"
    PERFORMANCE = "performance"
    OPPORTUNITY = "opportunity"
    RISK = "risk"
    DATA_QUALITY = "data_quality"
    RELATIONSHIP = "relationship"
    CONCENTRATION = "concentration"
    FORECAST = "forecast"
    MODEL = "model"
    COMPARISON = "comparison"


class InsightSeverity(str, Enum):
    """Significance / urgency of the insight."""
    INFORMATIONAL = "INFORMATIONAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ==============================================================================
# 2. Candidate & Request Models
# ==============================================================================

class AnalysisCandidate(BaseModel):
    """A prioritized analytical capability identified as suitable for dataset & intent."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    analysis_type: str  # e.g. "descriptive_statistics", "trend_analysis", "segmentation", "correlation_analysis", "anomaly_detection", "concentration_analysis"
    objective: str
    required_inputs: List[str] = Field(default_factory=list)
    priority: int = 1  # 1 = highest
    expected_value: float = Field(default=0.8, ge=0.0, le=1.0)
    computational_cost: str = "low"  # "low", "medium", "high"
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_type": self.analysis_type,
            "objective": self.objective,
            "required_inputs": self.required_inputs,
            "priority": self.priority,
            "expected_value": round(float(self.expected_value), 2),
            "computational_cost": self.computational_cost,
            "confidence": round(float(self.confidence), 2),
            "reason": self.reason,
        }


class AutonomousAnalysisRequest(BaseModel):
    """Specification payload for an autonomous analysis run."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset: Any  # pd.DataFrame, dict, or records
    user_intent: Optional[UserIntent] = None
    analysis_depth: AnalysisDepth = AnalysisDepth.STANDARD
    time_constraints: Optional[float] = None
    business_objective: Optional[str] = None
    max_analysis_steps: int = Field(default=10, ge=1, le=25)
    confidence_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    requested_output: Optional[str] = None
    constraints: Dict[str, Any] = Field(default_factory=dict)


# ==============================================================================
# 3. Insight Model
# ==============================================================================

class Insight(BaseModel):
    """Structured, evidence-backed analytical finding."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    insight_id: str = Field(default_factory=lambda: f"ins_{uuid.uuid4().hex[:8]}")
    title: str
    summary: str
    category: InsightCategory
    claim_type: ClaimType = ClaimType.FACT
    severity: InsightSeverity = InsightSeverity.INFORMATIONAL
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    evidence: Evidence
    affected_columns: List[str] = Field(default_factory=list)
    affected_segments: List[str] = Field(default_factory=list)
    calculation: Dict[str, Any] = Field(default_factory=dict)
    limitations: List[str] = Field(default_factory=list)
    recommended_action: Optional[str] = None
    source_analysis: str = "autonomous_analyst"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "insight_id": self.insight_id,
            "title": self.title,
            "summary": self.summary,
            "category": self.category.value,
            "claim_type": self.claim_type.value,
            "severity": self.severity.value,
            "importance": round(float(self.importance), 2),
            "confidence": round(float(self.confidence), 2),
            "evidence": self.evidence.to_dict(),
            "affected_columns": self.affected_columns,
            "affected_segments": self.affected_segments,
            "calculation": self.calculation,
            "limitations": self.limitations,
            "recommended_action": self.recommended_action,
            "source_analysis": self.source_analysis,
        }


# ==============================================================================
# 4. Master Autonomous Analysis Result
# ==============================================================================

class AutonomousAnalysisResult(BaseModel):
    """Comprehensive output of the Autonomous Data Analysis Engine."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    status: str = "success"  # "success", "partial", "failed"
    summary: str
    insights: List[Insight] = Field(default_factory=list)
    key_metrics: Dict[str, Any] = Field(default_factory=dict)
    analyses_performed: List[str] = Field(default_factory=list)
    analyses_skipped: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    confidence: float = 1.0
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "summary": self.summary,
            "insights": [i.to_dict() for i in self.insights],
            "key_metrics": self.key_metrics,
            "analyses_performed": self.analyses_performed,
            "analyses_skipped": self.analyses_skipped,
            "warnings": self.warnings,
            "limitations": self.limitations,
            "recommendations": self.recommendations,
            "evidence": [e.to_dict() for e in self.evidence],
            "confidence": round(float(self.confidence), 2),
            "execution_time": round(float(self.execution_time), 4),
        }

