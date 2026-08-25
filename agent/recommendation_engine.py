"""
Autonomous Decision & Recommendation Engine (Milestone 5, Task 4).

Pipeline:
    Observed Data
      -> Insights  ->  Forecasts  ->  What-If Scenarios  ->  Risks
      -> Constraints  ->  RECOMMENDATION ENGINE  ->  Ranked Actions
      -> Expected Impact  ->  Risks  ->  Confidence  ->  Evidence-backed Recommendation

Design guarantees:
- STRICT separation of facts, predictions, recommendations, and assumptions.
- Every recommendation is generated from a single DecisionContext and is tied
  to Evidence. A recommendation cannot exist without evidence.
- The LLM boundary: all numerical claims come from deterministic calculations;
  the LLM may only explain / summarize / translate, and NEVER invents impact,
  risk, constraints, metrics, or causality.
- Recommendations are advisory (human approval required). Nothing is executed.
- When the user provides no objective, the engine returns evidence-backed
  observations and asks which objective to optimize for.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from agent.recommendation_schemas import (
    ActionCandidate,
    ActionType,
    AuditRecord,
    DecisionConstraint,
    DecisionContext,
    ExpectedImpact,
    Fact,
    ImpactAvailability,
    OpportunityAssessment,
    Prediction,
    PriorityLevel,
    Recommendation,
    RecommendationObjective,
    RecommendationRequest,
    RecommendationResult,
    RiskAssessment,
    RiskSeverity,
    ScoringFactors,
    TradeOff,
)
from agent.schemas import ClaimType, Evidence


# ==============================================================================
# Internal normalization helpers (accept objects OR plain dicts)
# ==============================================================================

def _insight_normalize(insight: Any) -> Optional[Dict[str, Any]]:
    """Normalize an Insight object or dict into a stable internal dict."""
    if insight is None:
        return None
    if isinstance(insight, dict):
        return insight
    # Assume an object with attributes (e.g. autonomous_analysis_schemas.Insight)
    try:
        category = getattr(insight, "category", None)
        claim_type = getattr(insight, "claim_type", None)
        severity = getattr(insight, "severity", None)
        return {
            "insight_id": getattr(insight, "insight_id", None) or f"ins_{uuid.uuid4().hex[:6]}",
            "title": getattr(insight, "title", ""),
            "summary": getattr(insight, "summary", ""),
            "category": category.value if hasattr(category, "value") else str(category or ""),
            "claim_type": claim_type.value if hasattr(claim_type, "value") else str(claim_type or ""),
            "severity": severity.value if hasattr(severity, "value") else str(severity or ""),
            "importance": getattr(insight, "importance", 0.5),
            "confidence": getattr(insight, "confidence", 0.9),
            "evidence": getattr(insight, "evidence", None),
            "affected_segments": list(getattr(insight, "affected_segments", []) or []),
            "affected_columns": list(getattr(insight, "affected_columns", []) or []),
            "calculation": getattr(insight, "calculation", {}) or {},
            "recommended_action": getattr(insight, "recommended_action", None),
            "source_analysis": getattr(insight, "source_analysis", "insight"),
        }
    except Exception:
        return None


def _to_evidence_list(evidence: Any) -> List[Evidence]:
    """Coerce a single Evidence, list of Evidence, or None into a list."""
    if evidence is None:
        return []
    if isinstance(evidence, Evidence):
        return [evidence]
    if isinstance(evidence, (list, tuple)):
        out: List[Evidence] = []
        for e in evidence:
            if isinstance(e, Evidence):
                out.append(e)
            elif isinstance(e, dict):
                try:
                    out.append(Evidence(**e))
                except Exception:
                    pass
        return out
    return []


def _evidence_id(ev: Evidence) -> str:
    """Derive a stable evidence id for audit trail / traceability."""
    base = ev.source_reference or ev.source or ev.method or ev.operation or "evidence"
    return f"EV-{abs(hash(base)) % 100000:05d}"
