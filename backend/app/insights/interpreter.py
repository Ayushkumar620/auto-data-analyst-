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

class InsightInterpreter:
    """Writes business narratives for deterministic insights via the LLM."""

    def __init__(self) -> None:
        self.enabled = _llm_available()

    def enrich(
        self, dataset_name: str, facts: Dict[str, Any], insights: List[Insight]
    ) -> List[Insight]:
        if not self.enabled:
            return _fallback_descriptions(insights, facts)
        if not insights:
            return insights

        try:
            narrative = self._call_llm(dataset_name, facts, insights)
            return self._merge(narrative, insights)
        except Exception as exc:  # Fail open to deterministic descriptions
            logger.warning("LLM interpretation failed (%s); using deterministic fallback", exc)
            return _fallback_descriptions(insights, facts)

    def _call_llm(self, dataset_name: str, facts: Dict[str, Any], insights: List[Insight]) -> Dict[str, str]:
        import urllib.request

        evidence = {
            "facts": facts,
            "insights": [i.to_dict() for i in insights],
        }
        prompt = build_interpretation_prompt(dataset_name, evidence)
        url = getattr(settings, "openai_base_url", "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        payload = json.dumps({
            "model": settings.openai_model,
            "messages": [
                {"role": "system", "content": (
                    "You produce concise business explanations for data insights. "
                    "You MUST only reference facts supplied in the evidence. "
                    "Never invent numbers, never compute new statistics, and never "
                    "claim causation from correlation. Mark any recommendation as "
                    "conditional. Respond with a JSON object mapping each insight "
                    "title to a short explanation text."
                )},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }).encode("utf-8")

        req = urllib.request.Request(url, data=payload, method="POST")
        req.add_header("Authorization", f"Bearer {settings.openai_api_key}")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        return _extract_json_object(content)

    def _merge(self, narrative: Dict[str, str], insights: List[Insight]) -> List[Insight]:
        merged: List[Insight] = []
        for insight in insights:
            text = narrative.get(insight.title) or narrative.get(str(insight.title))
            description = text if text and not _looks_like_new_facts(text) else (
                insight.description or _default_description(insight.type, {})
            )
            merged.append(Insight(
                type=insight.type,
                title=insight.title,
                description=description,
                severity=insight.severity,
                confidence=insight.confidence,
                evidence=insight.evidence,
                recommendation=insight.recommendation,
                source="llm" if text else "rule",
            ))
        return merged


def _looks_like_new_facts(text: str) -> bool:
    """Heuristic guard: reject LLM text that introduces statistics not in evidence.

    We cannot verify arbitrary numbers, so if the text appears to invent figures we
    keep the deterministic description instead of surfacing unverified claims.
    """
    lowered = text.lower()
    suspicious = [
        "increased by", "decreased by", "grew by", "dropped by",
        "rose by", "fell by", "growth of", "decline of", "% to",
    ]
    return any(token in lowered for token in suspicious)


def _extract_json_object(content: str) -> Dict[str, str]:
    """Extract a JSON object from an LLM response (handles fences/trailing text)."""
    if not content:
        return {}
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): str(v) for k, v in parsed.items() if isinstance(v, (str, int, float))}