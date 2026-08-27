"""
Universal Analytical Explanation & Evidence Traceability Schemas.

Defines standardized Pydantic v2 data contracts for explainability,
evidence tracing, methodology breakdown, metric interpretation, and uncertainty modeling.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class ExplanationSection(BaseModel):
    """Structured section within an analytical explanation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    title: str
    content: str
    evidence_refs: List[str] = Field(default_factory=list, description="IDs of supporting Evidence objects")
    importance: float = Field(default=1.0, ge=0.0, le=1.0, description="Relative importance score")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "evidence_refs": self.evidence_refs,
            "importance": round(float(self.importance), 4),
            "metadata": self.metadata,
        }


class MetricExplanation(BaseModel):
    """Interpretation and benchmark context for a single analytical metric."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    metric_name: str
    value: Optional[Union[float, int, str]] = None
    interpretation: str
    validation_status: str = Field(default="validated", description="validated, warning, unvalidated")
    benchmark: Optional[str] = None
    confidence_interval: Optional[Tuple[float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "interpretation": self.interpretation,
            "validation_status": self.validation_status,
            "benchmark": self.benchmark,
        }


class EvidenceTrace(BaseModel):
    """Traceable link between an analytical claim and underlying computation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    evidence_id: str = Field(default_factory=lambda: f"evi_{uuid.uuid4().hex[:8]}")
    claim: str
    source: str
    method: str
    columns: List[str] = Field(default_factory=list)
    rows_analyzed: Optional[int] = None
    calculation: Optional[str] = None
    result: Optional[Union[float, int, str, Dict[str, Any], List[Any]]] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    claim_type: str = "observation"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "claim": self.claim,
            "source": self.source,
            "method": self.method,
            "columns": self.columns,
            "rows_analyzed": self.rows_analyzed,
            "calculation": self.calculation,
            "result": self.result,
            "confidence": round(float(self.confidence), 4),
            "claim_type": self.claim_type,
        }


class AnalyticalExplanation(BaseModel):
    """Canonical analytical explanation contract with full evidence traceability."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    explanation_id: str = Field(default_factory=lambda: f"exp_{uuid.uuid4().hex[:8]}")
    task_type: str = "general"
    summary: str
    findings: List[ExplanationSection] = Field(default_factory=list)
    methodology: List[ExplanationSection] = Field(default_factory=list)
    metrics: List[MetricExplanation] = Field(default_factory=list)
    evidence: List[EvidenceTrace] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    uncertainty: Dict[str, Any] = Field(
        default_factory=lambda: {
            "statistical_confidence": None,
            "model_validation_score": None,
            "prediction_interval_level": None,
            "epistemic_confidence": 1.0,
            "practical_effect_size": None,
            "notes": [],
        }
    )
    provenance: Dict[str, Any] = Field(default_factory=dict)
    recommended_next_steps: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "explanation_id": self.explanation_id,
            "task_type": self.task_type,
            "summary": self.summary,
            "findings": [f.to_dict() for f in self.findings],
            "methodology": [m.to_dict() for m in self.methodology],
            "metrics": [me.to_dict() for me in self.metrics],
            "evidence": [e.to_dict() for e in self.evidence],
            "assumptions": self.assumptions,
            "limitations": self.limitations,
            "uncertainty": self.uncertainty,
            "provenance": self.provenance,
            "recommended_next_steps": self.recommended_next_steps,
        }

    def to_markdown(self) -> str:
        """Format the complete explanation as a formatted markdown document."""
        lines = [
            f"# Analytical Explanation: {self.task_type.replace('_', ' ').title()}",
            "",
            "## Summary",
            self.summary,
            "",
        ]

        if self.findings:
            lines.append("## Key Findings & Evidence")
            for f in self.findings:
                lines.append(f"### {f.title}")
                lines.append(f.content)
                if f.evidence_refs:
                    lines.append(f"*Supporting Evidence:* {', '.join(f.evidence_refs)}")
                lines.append("")

        if self.metrics:
            lines.append("## Validated Metrics")
            for m in self.metrics:
                lines.append(f"- **{m.metric_name}**: `{m.value}` — {m.interpretation}")
            lines.append("")

        if self.methodology:
            lines.append("## Methodology & Calculation")
            for meth in self.methodology:
                lines.append(f"### {meth.title}")
                lines.append(meth.content)
                lines.append("")

        if self.uncertainty:
            lines.append("## Uncertainty & Reliability")
            for k, v in self.uncertainty.items():
                if v is not None and k != "notes":
                    lines.append(f"- **{k.replace('_', ' ').title()}**: `{v}`")
            for note in self.uncertainty.get("notes", []):
                lines.append(f"- *Note:* {note}")
            lines.append("")

        if self.limitations:
            lines.append("## Limitations & Causal Boundaries")
            for lim in self.limitations:
                lines.append(f"- {lim}")
            lines.append("")

        if self.recommended_next_steps:
            lines.append("## Recommended Next Questions")
            for step in self.recommended_next_steps:
                lines.append(f"- {step}")
            lines.append("")

        return "\n".join(lines)