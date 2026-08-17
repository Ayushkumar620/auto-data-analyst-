"""Optional LLM interpretation layer for insights.

The AI never computes statistics. All numerical facts are produced by Python
data-processing code (FactAnalyzer / EDA) and are passed to the model as a
read-only evidence block. The model writes business explanations that reference
only those facts. When no API key is configured, or on any LLM failure, the
system transparently falls back to the deterministic rule-based descriptions so
the endpoint always returns a useful, evidence-backed result.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from backend.app.config import settings
from .prompts import build_interpretation_prompt
from .schemas import Insight

logger = logging.getLogger(__name__)

_ALLOWED_TYPES: set[str] = {
    "key_finding", "trend", "anomaly", "risk", "opportunity", "recommendation",
}


def _llm_available() -> bool:
    return bool(getattr(settings, "openai_api_key", "") or "")


def _fallback_descriptions(
    insights: List[Insight], facts: Dict[str, Any]
) -> List[Insight]:
    """Attach conservative, evidence-backed descriptions when no LLM is present.

    This keeps the insight stream useful and truthful without inventing facts.
    """
    enriched: List[Insight] = []
    for insight in insights:
        description = insight.description or _default_description(insight.type, facts)
        enriched.append(Insight(
            type=insight.type,
            title=insight.title,
            description=description,
            severity=insight.severity,
            confidence=insight.confidence,
            evidence=insight.evidence,
            recommendation=insight.recommendation,
            source="rule",
        ))
    return enriched


def _default_description(insight_type: str, facts: Dict[str, Any]) -> str:
    if insight_type == "key_finding":
        return "A notable pattern was identified in the available evidence."
    if insight_type == "trend":
        return "A measurable change was observed during the analyzed period."
    if insight_type == "anomaly":
        return "Unusual values were detected that may warrant review."
    if insight_type == "risk":
        return "A condition that could limit reliability or performance was identified."
    if insight_type == "opportunity":
        return "An area with potential for further focus was identified."
    return "Review the supporting evidence for more detail."