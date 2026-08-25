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
# ==============================================================================
# DecisionContextBuilder
# ==============================================================================

class DecisionContextBuilder:
    """Assembles a DecisionContext (facts, insights, predictions, constraints,
    assumptions, uncertainties) from a RecommendationRequest."""

    @staticmethod
    def _normalize_constraint(c: Any) -> Optional[DecisionConstraint]:
        if isinstance(c, DecisionConstraint):
            return c
        if isinstance(c, dict):
            try:
                return DecisionConstraint(**c)
            except Exception:
                pass
        return None

    @classmethod
    def build(cls, request: RecommendationRequest) -> DecisionContext:
        facts: List[Fact] = []
        assumptions: List[str] = []
        uncertainties: List[str] = []
        insights: List[Dict[str, Any]] = []

        # ---- 1. Facts & insights from the insights list ----
        for raw in request.insights:
            ins = _insight_normalize(raw)
            if not ins:
                continue
            insights.append(ins)
            for lim in ins.get("limitations", []) or []:
                if lim and lim not in uncertainties:
                    uncertainties.append(str(lim))
            claim = str(ins.get("claim_type", "")).lower()
            if claim in ("fact", "observation"):
                calc = ins.get("calculation") or {}
                facts.append(Fact(
                    statement=ins.get("summary") or ins.get("title") or "",
                    claim_type=ClaimType.FACT if claim == "fact" else ClaimType.OBSERVATION,
                    category=ins.get("category") or "fact",
                    value=cls._extract_primary_value(calc),
                    affected_metric=cls._pick_metric(calc, ins.get("affected_columns") or []),
                    affected_segment=calc.get("top_segment") or calc.get("dimension")
                                   or (ins.get("affected_segments") or [None])[0],
                    evidence=_to_evidence_list(ins.get("evidence")),
                    confidence=float(ins.get("confidence", 0.9) or 0.9),
                ))

        # ---- 2. Predictions from forecasts ----
        predictions: List[Prediction] = []
        for fc in request.forecasts:
            pred = cls._forecast_to_prediction(fc)
            if pred:
                predictions.append(pred)
            assumptions.extend(cls._forecast_assumptions(fc))
            uncertainties.extend(cls._forecast_uncertainties(fc))

        # ---- 3. Scenario assumptions / uncertainties ----
        for sc in request.scenarios:
            scd = cls._scenario_to_dict(sc)
            if scd:
                assumptions.extend(scd.get("assumptions", []) or [])
                uncertainties.extend(scd.get("limitations", []) or [])

        # ---- 4. Constraints ----
        constraints: List[DecisionConstraint] = []
        for bc in request.business_constraints:
            c = cls._normalize_constraint(bc)
            if c:
                constraints.append(c)

        # ---- 5. Dataset knowledge based uncertainties ----
        dk = request.dataset_context
        if dk is not None:
            try:
                dq = getattr(dk, "quality", None)
                if dq is not None:
                    qs = getattr(dq, "quality_score", None)
                    warnings = getattr(dq, "warnings", []) or []
                    if isinstance(qs, (int, float)) and qs < 80:
                        uncertainties.append(f"Dataset quality score is {qs} (below 80).")
                    uncertainties.extend([str(w) for w in warnings if w])
            except Exception:
                pass

        return DecisionContext(
            facts=facts,
            insights=insights,
            predictions=predictions,
            constraints=constraints,
            assumptions=list(dict.fromkeys(assumptions)),
            uncertainties=list(dict.fromkeys(uncertainties)),
            available_actions=[],
        )


    # ---- helpers ------------------------------------------------------------
    @staticmethod
    def _extract_primary_value(calc: Dict[str, Any]) -> Optional[float]:
        for key in ("overall_growth_pct", "top_share_pct", "top_20_percent_share",
                    "top_3_entities_share", "total_metric", "top_value"):
            v = calc.get(key)
            if isinstance(v, (int, float)):
                return float(v)
        return None

    @staticmethod
    def _pick_metric(calc: Dict[str, Any], cols: List[str]) -> Optional[str]:
        m = calc.get("metric") or calc.get("target")
        if m:
            return str(m)
        for c in cols:
            if c and any(t in str(c).lower() for t in ("rev", "sales", "profit", "spend", "cost", "retention")):
                return str(c)
        return cols[0] if cols else None

    @staticmethod
    def _scenario_to_dict(sc: Any) -> Optional[Dict[str, Any]]:
        if sc is None:
            return None
        if isinstance(sc, dict):
            return sc
        try:
            return {
                "scenario_name": getattr(sc, "scenario_name", ""),
                "target_metric": getattr(sc, "target_metric", None),
                "baseline_value": getattr(sc, "baseline_value", None),
                "scenario_value": getattr(sc, "scenario_value", None),
                "absolute_difference": getattr(sc, "absolute_difference", None),
                "percentage_difference": getattr(sc, "percentage_difference", None),
                "assumptions": list(getattr(sc, "assumptions", []) or []),
                "limitations": list(getattr(sc, "limitations", []) or []),
                "evidence": _to_evidence_list(getattr(sc, "evidence", None)),
                "scenario_id": None,
            }
        except Exception:
            return None

    @staticmethod
    def _forecast_to_prediction(fc: Any) -> Optional[Prediction]:
        if fc is None:
            return None
        if isinstance(fc, dict):
            preds = fc.get("predictions") or []
            return DecisionContextBuilder._prediction_from_points(
                preds, fc.get("target"), fc.get("model_id"), fc.get("forecast_horizon"),
                _to_evidence_list(fc.get("evidence")))
        try:
            preds = list(getattr(fc, "predictions", []) or [])
            return DecisionContextBuilder._prediction_from_points(
                preds, getattr(fc, "target", None), getattr(fc, "model_id", None),
                getattr(fc, "forecast_horizon", None),
                _to_evidence_list(getattr(fc, "evidence", None)))
        except Exception:
            return None

    @staticmethod
    def _prediction_from_points(preds: List[Any], target: Optional[str],
                                model_id: Optional[str], horizon: Optional[int],
                                evidence: List[Evidence]) -> Optional[Prediction]:
        if not preds:
            return None
        vals: List[float] = []
        for p in preds:
            v = p.get("prediction") if isinstance(p, dict) else getattr(p, "prediction", None)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if not vals:
            return None
        first, last = vals[0], vals[-1]
        change = ((last - first) / first * 100.0) if first else 0.0
        direction = "up" if change > 2.0 else ("down" if change < -2.0 else "stable")
        metric = target or "target"
        statement = (f"{metric} is forecast to {direction} approximately "
                     f"{abs(change):.1f}% over the forecast horizon.")
        return Prediction(
            statement=statement,
            metric=str(metric),
            direction=direction,
            change_percent=round(change, 4),
            horizon=horizon,
            model_id=model_id,
            evidence=evidence,
            confidence=0.8,
        )

    @staticmethod
    def _forecast_assumptions(fc: Any) -> List[str]:
        if isinstance(fc, dict):
            return [str(a) for a in (fc.get("assumptions") or []) if a]
        try:
            return [str(a) for a in (getattr(fc, "assumptions", []) or []) if a]
        except Exception:
            return []

    @staticmethod
    def _forecast_uncertainties(fc: Any) -> List[str]:
        if isinstance(fc, dict):
            lims = [str(a) for a in (fc.get("limitations") or []) if a]
            if not lims:
                lims = [f"Forecast for {fc.get('target', 'metric')} carries uncertainty."]
            return lims
        try:
            lims = [str(a) for a in (getattr(fc, "limitations", []) or []) if a]
            if not lims:
                lims = [f"Forecast for {getattr(fc, 'target', 'metric')} carries uncertainty."]
            return lims
        except Exception:
            return []

