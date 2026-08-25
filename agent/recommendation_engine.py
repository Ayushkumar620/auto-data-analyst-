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

# ==============================================================================
# Normalization helpers
# ==============================================================================

def _monitoring_to_dict(m: Any) -> Optional[Dict[str, Any]]:
    """Normalize a MonitoringResult (object or dict) into a dict."""
    if m is None:
        return None
    if isinstance(m, dict):
        return m
    try:
        dd = getattr(m, "data_drift", None)
        dd = dd.to_dict() if dd is not None and hasattr(dd, "to_dict") else dd
        pdd = getattr(m, "prediction_drift", None)
        pdd = pdd.to_dict() if pdd is not None and hasattr(pdd, "to_dict") else pdd
        pd = getattr(m, "performance_drift", None)
        pd = pd.to_dict() if pd is not None and hasattr(pd, "to_dict") else pd
        return {
            "model_id": getattr(m, "model_id", None),
            "overall_severity": getattr(m, "overall_severity", None),
            "data_drift": dd,
            "prediction_drift": pdd,
            "performance_drift": pd,
            "evidence": _to_evidence_list(getattr(m, "evidence", None)),
            "recommendations": list(getattr(m, "recommendations", []) or []),
        }
    except Exception:
        return None


def _metric_label(metric: Optional[str]) -> str:
    return metric or "the metric"


class RiskEngine:
    """Deterministic, evidence-backed risk assessment. Only creates risks that
    are supported by evidence. Never assigns CRITICAL simply because something
    looks unusual."""

    SEVERITY_WEIGHT = {
        RiskSeverity.LOW: 0.10,
        RiskSeverity.MEDIUM: 0.25,
        RiskSeverity.HIGH: 0.50,
        RiskSeverity.CRITICAL: 0.80,
    }

    @staticmethod
    def _severity_for_share(share: float) -> RiskSeverity:
        if share >= 90.0:
            return RiskSeverity.CRITICAL
        if share >= 75.0:
            return RiskSeverity.HIGH
        if share >= 55.0:
            return RiskSeverity.MEDIUM
        return RiskSeverity.LOW

    def assess(self, context: DecisionContext,
               monitoring: Optional[List[Dict[str, Any]]] = None) -> List[RiskAssessment]:
        risks: List[RiskAssessment] = []

        # ---- Concentration risks ----
        for ins in context.insights:
            calc = ins.get("calculation") or {}
            ev = _to_evidence_list(ins.get("evidence"))
            cat = str(ins.get("category", "")).lower()
            if cat == "concentration":
                top20 = calc.get("top_20_percent_share")
                top3 = calc.get("top_3_entities_share")
                share = float(top20) if isinstance(top20, (int, float)) else (
                    float(top3) if isinstance(top3, (int, float)) else 0.0)
                if share > 0:
                    risks.append(RiskAssessment(
                        risk=(f"Customer/revenue concentration: {calc.get('dimension') or 'entities'} "
                              f"shows high concentration (top ~{share:.0f}% of "
                              f"{_metric_label(calc.get('metric'))})."),
                        severity=self._severity_for_share(share),
                        probability=round(min(1.0, share / 100.0 + 0.15), 3),
                        impact="Revenue dependency on a small set of entities.",
                        evidence=ev,
                        evidence_ids=[_evidence_id(e) for e in ev],
                        mitigation="Review retention plans and diversify the customer base.",
                        confidence=max(float(ins.get("confidence", 0.9)), 0.7),
                    ))
            elif cat == "performance":
                share = calc.get("top_share_pct")
                if isinstance(share, (int, float)) and share >= 55.0 and (calc.get("segment_count") or 1) >= 3:
                    risks.append(RiskAssessment(
                        risk=(f"Segment concentration: '{calc.get('top_segment')}' contributes "
                              f"{share:.0f}% of {calc.get('metric')}, creating dependency."),
                        severity=self._severity_for_share(share),
                        probability=round(min(1.0, share / 100.0), 3),
                        impact="Over-reliance on a single segment.",
                        evidence=ev,
                        evidence_ids=[_evidence_id(e) for e in ev],
                        mitigation="Diversify across segments and monitor the dominant segment.",
                        confidence=float(ins.get("confidence", 0.9)),
                    ))


        # ---- Forecast uncertainty ----
        if context.predictions:
            pred = context.predictions[0]
            ev_pred = pred.evidence or [Evidence(
                source="recommendation_engine.forecast", method="forecast_uncertainty",
                confidence=0.8, claim_type=ClaimType.INFERENCE)]
            risks.append(RiskAssessment(
                risk=(f"Forecast uncertainty: the forward estimate for '{pred.metric}' "
                      f"({pred.direction}) is a projection, not a certainty."),
                severity=RiskSeverity.MEDIUM if pred.direction != "stable" else RiskSeverity.LOW,
                probability=0.6,
                impact="Decisions based on the forecast may miss the actual outcome.",
                evidence=ev_pred,
                evidence_ids=[_evidence_id(e) for e in ev_pred],
                mitigation="Treat forecast ranges as guidance and revisit assumptions.",
                confidence=0.8,
            ))

        # ---- Model monitoring (drift / degradation) ----
        for mon in (monitoring or []):
            risks.extend(self._monitoring_risks(mon))
        return risks

    def _monitoring_risks(self, mon: Dict[str, Any]) -> List[RiskAssessment]:
        out: List[RiskAssessment] = []
        ev = mon.get("evidence") or []
        dd = mon.get("data_drift")
        if isinstance(dd, dict):
            sev_raw = dd.get("severity")
            sev_key = str(sev_raw).upper() if sev_raw else ("HIGH" if dd.get("overall_drift") else "NONE")
            if sev_key in ("HIGH", "CRITICAL") or dd.get("overall_drift"):
                sev = RiskSeverity.CRITICAL if sev_key == "CRITICAL" else RiskSeverity.HIGH
                out.append(RiskAssessment(
                    risk=("Data drift detected: incoming data distribution differs "
                          "significantly from the reference, so model predictions may "
                          "no longer be reliable."),
                    severity=sev,
                    probability=0.7,
                    impact="Predictions may degrade under distribution change.",
                    evidence=ev,
                    evidence_ids=[_evidence_id(e) for e in ev],
                    mitigation=("Validate incoming data and investigate the source of "
                                "distribution changes before relying on predictions."),
                    confidence=0.85,
                ))
        pd = mon.get("performance_drift")
        if isinstance(pd, dict) and pd.get("degradation_detected"):
            out.append(RiskAssessment(
                risk=("Model performance degradation detected against the reference "
                      "baseline metric."),
                severity=RiskSeverity.HIGH,
                probability=0.7,
                impact="Model quality below acceptable level for production decisions.",
                evidence=ev,
                evidence_ids=[_evidence_id(e) for e in ev],
                mitigation=("Evaluate model recalibration or retraining after "
                            "investigating the cause."),
                confidence=0.85,
            ))
        pdrift = mon.get("prediction_drift")
        if isinstance(pdrift, dict) and pdrift.get("prediction_drift_detected"):
            out.append(RiskAssessment(
                risk=("Prediction drift detected: the distribution of model outputs "
                      "has shifted."),
                severity=RiskSeverity.MEDIUM,
                probability=0.6,
                impact="Model outputs may be miscalibrated.",
                evidence=ev,
                evidence_ids=[_evidence_id(e) for e in ev],
                mitigation="Investigate input changes and monitor output calibration.",
                confidence=0.8,
            ))
        return out

class OpportunityEngine:
    """Evidence-backed opportunity detection. Never labels something an
    opportunity without measurable supporting evidence."""

    def detect(self, context: DecisionContext,
               scenarios: Optional[List[Dict[str, Any]]] = None) -> List[OpportunityAssessment]:
        opportunities: List[OpportunityAssessment] = []
        seen = set()

        for ins in context.insights:
            calc = ins.get("calculation") or {}
            ev = _to_evidence_list(ins.get("evidence"))
            cat = str(ins.get("category", "")).lower()
            metric = calc.get("metric") or (ins.get("affected_columns") or [None])[-1] or None
            segment = calc.get("top_segment") or calc.get("segment") or (ins.get("affected_segments") or [None])[0]

            if cat == "opportunity":
                key = ("opp", ins.get("title", ""))
                if key not in seen:
                    seen.add(key)
                    opportunities.append(OpportunityAssessment(
                        opportunity=ins.get("summary") or ins.get("title", ""),
                        evidence=ev,
                        evidence_ids=[_evidence_id(e) for e in ev],
                        affected_segment=segment,
                        affected_metric=metric,
                        confidence=float(ins.get("confidence", 0.8)),
                        assumptions=[],
                    ))
            elif cat == "trend":
                growth = calc.get("overall_growth_pct")
                if isinstance(growth, (int, float)) and growth >= 10.0:
                    key = ("trend", metric)
                    if key not in seen:
                        seen.add(key)
                        opportunities.append(OpportunityAssessment(
                            opportunity=(f"Strong {growth:.0f}% growth in '{metric}' "
                                         f"represents a momentum opportunity."),
                            evidence=ev,
                            evidence_ids=[_evidence_id(e) for e in ev],
                            affected_segment=segment,
                            affected_metric=metric,
                            confidence=float(ins.get("confidence", 0.8)),
                            assumptions=["Historical trend is assumed to continue; this is a projection, not a guarantee."],
                            risks=["Trend may not persist."],
                        ))
            elif cat == "performance":
                share = calc.get("top_share_pct")
                disparity = calc.get("disparity_ratio")
                if isinstance(share, (int, float)) and share >= 40.0:
                    key = ("high_value", calc.get("top_segment"))
                    if key not in seen:
                        seen.add(key)
                        opportunities.append(OpportunityAssessment(
                            opportunity=(f"High-value segment '{calc.get('top_segment')}' "
                                         f"contributes {share:.0f}% of {metric}."),
                            evidence=ev,
                            evidence_ids=[_evidence_id(e) for e in ev],
                            affected_segment=calc.get("top_segment"),
                            affected_metric=metric,
                            confidence=float(ins.get("confidence", 0.85)),
                            assumptions=[],
                            risks=["High segment concentration creates dependency risk."],
                        ))
                if disparity and isinstance(disparity, (int, float)) and disparity >= 3.0:
                    key = ("open", calc.get("bottom_segment"))
                    if key not in seen:
                        seen.add(key)
                        opportunities.append(OpportunityAssessment(
                            opportunity=(f"Underpenetrated segment '{calc.get('bottom_segment')}' "
                                         f"({disparity}x gap vs. leader) offers upside."),
                            evidence=ev,
                            evidence_ids=[_evidence_id(e) for e in ev],
                            affected_segment=calc.get("bottom_segment"),
                            affected_metric=metric,
                            confidence=float(ins.get("confidence", 0.6)),
                            assumptions=["Gap is addressable with the available levers."],
                            risks=["Lagging segment may have structural constraints."],
                        ))
        return opportunities

class ExpectedImpactEstimator:
    """Estimates expected impact strictly from real scenario/model calculations.
    If impact cannot be quantified, availability = 'unavailable'."""

    def estimate(self, candidate: ActionCandidate,
                 scenarios: Optional[List[Dict[str, Any]]]) -> ExpectedImpact:
        metric = candidate.affected_metric
        if not scenarios:
            return ExpectedImpact()
        matches = [s for s in scenarios if s and s.get("target_metric") == metric]
        if not matches:
            for s in scenarios:
                name = (s.get("scenario_name") or "").lower()
                if any(k in name for k in (candidate.objective.value.replace("_", " "),
                                           candidate.action_type.value, str(metric or "").lower())):
                    matches.append(s)
        if not matches:
            return ExpectedImpact()
        best = max(matches, key=lambda s: abs(float(s.get("absolute_difference") or 0.0)))
        baseline = best.get("baseline_value")
        scenario_val = best.get("scenario_value")
        abs_diff = best.get("absolute_difference")
        pct_diff = best.get("percentage_difference")
        if baseline is None or scenario_val is None:
            return ExpectedImpact()
        return ExpectedImpact(
            availability=ImpactAvailability.AVAILABLE,
            metric=metric,
            baseline=float(baseline),
            scenario=float(scenario_val),
            estimated_impact=float(abs_diff) if abs_diff is not None else None,
            estimated_impact_pct=float(pct_diff) if pct_diff is not None else None,
            scenario_name=best.get("scenario_name"),
            scenario_id=best.get("scenario_id"),
            assumptions=list(best.get("assumptions", []) or []),
            uncertainties=list(best.get("limitations", []) or []),
        )

class ConstraintValidator:
    """Rejects recommendations that violate explicit user constraints.

    Deterministic policy:
      - spending_limit / budget_limit: reject actions that imply increased spend
        (INCREASE, EXPAND, REALLOCATE).
      - region_filter / segment_filter: reject actions whose affected segment is
        outside the allowed set.
      - retention_min: reject actions that could lower retention (DECREASE).
    """

    @staticmethod
    def _allowed_regions(constraint: DecisionConstraint) -> Optional[List[str]]:
        v = constraint.value
        if v is None:
            return None
        if isinstance(v, str):
            return [v]
        if isinstance(v, (list, tuple, set)):
            return [str(x) for x in v]
        if isinstance(v, dict):
            regions = v.get("regions") or v.get("allowed") or v.get("segment")
            if isinstance(regions, str):
                return [regions]
            if isinstance(regions, (list, tuple, set)):
                return [str(x) for x in regions]
            if v.get("region"):
                return [str(v["region"])]
        return None

    def validate(self, candidate: ActionCandidate,
                 constraints: List[DecisionConstraint]) -> Tuple[bool, List[DecisionConstraint]]:
        passed = True
        satisfied: List[DecisionConstraint] = []
        for c in constraints:
            ctype = str(c.type).lower()
            allowed = True
            if ctype in ("spending_limit", "budget_limit", "spending", "budget", "do not increase spending"):
                if candidate.action_type in (ActionType.INCREASE, ActionType.EXPAND, ActionType.REALLOCATE):
                    allowed = False
            elif ctype in ("region_filter", "region", "only_focus_on", "segment_filter", "segment"):
                regions = self._allowed_regions(c)
                if regions and candidate.affected_segment:
                    seg_norm = str(candidate.affected_segment).lower()
                    if not any(str(r).lower() in seg_norm or seg_norm in str(r).lower() for r in regions):
                        allowed = False
            elif ctype in ("retention_min", "retention", "keep_retention"):
                if candidate.action_type == ActionType.DECREASE or "reduce retention" in candidate.action.lower():
                    allowed = False
            if not allowed:
                passed = False
                c.validation_status = "rejected"
            else:
                satisfied.append(c)
                c.validation_status = "satisfied"
        return passed, satisfied

class ActionGenerator:
    """Discovers evidence-tied ActionCandidates from insights, forecasts,
    monitoring results, and opportunities. Generates actions only from evidence."""

    def __init__(self) -> None:
        self.validator = ConstraintValidator()
        self.impact = ExpectedImpactEstimator()

    def generate(self, context: DecisionContext,
                 objectives: List[RecommendationObjective],
                 scenarios: Optional[List[Dict[str, Any]]],
                 monitoring: Optional[List[Dict[str, Any]]],
                 risk_tolerance: Optional[str] = None) -> List[ActionCandidate]:
        candidates: List[ActionCandidate] = []
        seen_keys = set()

        # ---- 1. Insight-driven actions ----
        for ins in context.insights:
            calc = ins.get("calculation") or {}
            ev = _to_evidence_list(ins.get("evidence"))
            cat = str(ins.get("category", "")).lower()
            metric = calc.get("metric") or None
            segment = calc.get("top_segment") or None

            if cat == "concentration":
                key = ("concentration", metric)
                if key not in seen_keys:
                    seen_keys.add(key)
                    candidates.append(ActionCandidate(
                        action="Review retention plans for high-value customers and assess dependency risk.",
                        action_type=ActionType.REVIEW,
                        objective=self._pick_objective(objectives, [RecommendationObjective.REDUCE_RISK, RecommendationObjective.IMPROVE_RETENTION]),
                        affected_metric=metric,
                        affected_segment=calc.get("dimension"),
                        rationale=("Customer concentration is high; a small number of entities "
                                   "drive a large share of revenue."),
                        evidence=ev,
                        evidence_ids=[_evidence_id(e) for e in ev],
                        assumptions=["Current customer mix persists without intervention."],
                        risks=["Revenue concentration creates dependency risk."],
                        confidence=float(ins.get("confidence", 0.85)),
                    ))
            elif cat == "performance":
                share = calc.get("top_share_pct")
                if isinstance(share, (int, float)) and share >= 40.0:
                    key = ("retain", calc.get("top_segment"))
                    if key not in seen_keys:
                        seen_keys.add(key)
                        candidates.append(ActionCandidate(
                            action=(f"Prioritize retention and expansion of high-value "
                                    f"'{calc.get('top_segment')}' {calc.get('dimension')} customers."),
                            action_type=ActionType.RETAIN,
                            objective=self._pick_objective(objectives, [RecommendationObjective.MAXIMIZE_REVENUE, RecommendationObjective.IMPROVE_RETENTION]),
                            affected_metric=metric,
                            affected_segment=calc.get("top_segment"),
                            rationale=(f"{calc.get('top_segment')} contributes {share:.0f}% of "
                                       f"{metric}, so protecting it protects revenue."),
                            evidence=ev,
                            evidence_ids=[_evidence_id(e) for e in ev],
                            assumptions=["High-value segment remains serviceable."],
                            risks=["High segment concentration creates dependency risk."],
                            confidence=float(ins.get("confidence", 0.85)),
                        ))
                bottom = calc.get("bottom_segment")
                if bottom and isinstance(share, (int, float)) and share >= 40.0:
                    key = ("inv", bottom)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        candidates.append(ActionCandidate(
                            action=(f"Investigate '{bottom}' {calc.get('dimension')} conversion "
                                    f"and product mix before increasing investment."),
                            action_type=ActionType.INVESTIGATE,
                            objective=self._pick_objective(objectives, [RecommendationObjective.MAXIMIZE_REVENUE, RecommendationObjective.IMPROVE_CONVERSION]),
                            affected_metric=metric,
                            affected_segment=bottom,
                            rationale=(f"'{bottom}' lags the top segment in {metric}; understand "
                                       f"why before committing spend."),
                            evidence=ev,
                            evidence_ids=[_evidence_id(e) for e in ev],
                            assumptions=["The gap is addressable with the available levers."],
                            risks=[],
                            confidence=float(ins.get("confidence", 0.7)),
                        ))

            elif cat == "trend":
                growth = calc.get("overall_growth_pct")
                if isinstance(growth, (int, float)):
                    key = ("trend", metric, "up" if growth >= 0 else "down")
                    if key not in seen_keys:
                        seen_keys.add(key)
                        if growth > 5.0:
                            candidates.append(ActionCandidate(
                                action=(f"Continue investing in rising '{metric}' and scale the "
                                        f"strongest-performing segments."),
                                action_type=ActionType.EXPAND,
                                objective=self._pick_objective(objectives, [RecommendationObjective.MAXIMIZE_REVENUE]),
                                affected_metric=metric,
                                affected_segment=segment,
                                rationale=f"'{metric}' grew {growth:.0f}%, so expansion builds on momentum.",
                                evidence=ev,
                                evidence_ids=[_evidence_id(e) for e in ev],
                                assumptions=["Historical trend continues (a projection, not a guarantee)."],
                                risks=["Trend may not persist."],
                                confidence=float(ins.get("confidence", 0.8)),
                            ))
                        elif growth < -5.0:
                            candidates.append(ActionCandidate(
                                action=(f"Investigate drivers of declining '{metric}' before "
                                        f"adjusting investment."),
                                action_type=ActionType.INVESTIGATE,
                                objective=self._pick_objective(objectives, [RecommendationObjective.REDUCE_RISK, RecommendationObjective.MAXIMIZE_REVENUE]),
                                affected_metric=metric,
                                affected_segment=segment,
                                rationale=f"'{metric}' declined {abs(growth):.0f}%; diagnose before acting.",
                                evidence=ev,
                                evidence_ids=[_evidence_id(e) for e in ev],
                                assumptions=[],
                                risks=["Decline could accelerate if unaddressed."],
                                confidence=float(ins.get("confidence", 0.8)),
                            ))

        # ---- 2. Forecast-driven actions ----
        for pred in context.predictions:
            if pred.direction == "down":
                candidates.append(ActionCandidate(
                    action="Review demand drivers and consider reducing inventory exposure.",
                    action_type=ActionType.REVIEW,
                    objective=self._pick_objective(objectives, [RecommendationObjective.REDUCE_RISK, RecommendationObjective.REDUCE_COST]),
                    affected_metric=pred.metric,
                    affected_segment=None,
                    rationale=(f"Forecast expects '{pred.metric}' to decline "
                               f"({pred.change_percent:+.1f}%)."),
                    evidence=pred.evidence,
                    evidence_ids=[_evidence_id(e) for e in pred.evidence],
                    assumptions=["Forecast assumptions hold (a projection, not a certainty)."],
                    risks=["Forecast uncertainty."],
                    confidence=0.8,
                ))
            elif pred.direction == "up":
                candidates.append(ActionCandidate(
                    action=(f"Monitor '{pred.metric}' upside and be ready to capture "
                            f"the projected {pred.change_percent:+.1f}% growth."),
                    action_type=ActionType.MONITOR,
                    objective=self._pick_objective(objectives, [RecommendationObjective.MAXIMIZE_REVENUE]),
                    affected_metric=pred.metric,
                    affected_segment=None,
                    rationale=f"Forecast projects '{pred.metric}' growth ({pred.change_percent:+.1f}%).",
                    evidence=pred.evidence,
                    evidence_ids=[_evidence_id(e) for e in pred.evidence],
                    assumptions=["Forecast assumptions hold (a projection, not a certainty)."],
                    risks=["Forecast uncertainty."],
                    confidence=0.8,
                ))

        # ---- 3. Monitoring-driven actions ----
        for mon in (monitoring or []):
            dd = mon.get("data_drift")
            if isinstance(dd, dict) and (dd.get("overall_drift") or
                                         str(dd.get("severity", "")).upper() in ("HIGH", "CRITICAL")):
                candidates.append(ActionCandidate(
                    action="Validate incoming data and investigate the source of distribution changes before relying on predictions.",
                    action_type=ActionType.INVESTIGATE,
                    objective=self._pick_objective(objectives, [RecommendationObjective.REDUCE_RISK]),
                    affected_metric=None,
                    affected_segment=None,
                    rationale="Model monitoring detected significant data drift.",
                    evidence=mon.get("evidence") or [],
                    evidence_ids=[_evidence_id(e) for e in (mon.get("evidence") or [])],
                    assumptions=[],
                    risks=["Data drift reduces prediction reliability."],
                    confidence=0.85,
                ))
            pd = mon.get("performance_drift")
            if isinstance(pd, dict) and pd.get("degradation_detected"):
                candidates.append(ActionCandidate(
                    action="Evaluate model recalibration or retraining after investigating the cause.",
                    action_type=ActionType.REVIEW,
                    objective=self._pick_objective(objectives, [RecommendationObjective.REDUCE_RISK]),
                    affected_metric=None,
                    affected_segment=None,
                    rationale="Model performance degradation was detected by monitoring.",
                    evidence=mon.get("evidence") or [],
                    evidence_ids=[_evidence_id(e) for e in (mon.get("evidence") or [])],
                    assumptions=["Cause must be understood before retraining."],
                    risks=["Retraining without root-cause could mask drift."],
                    confidence=0.85,
                ))

        # ---- 4. Opportunity-driven actions (only when evidence-backed) ----
        opp_engine = OpportunityEngine()
        for opp in opp_engine.detect(context, scenarios):
            if opp.affected_metric and opp.confidence >= 0.6 and opp.opportunity:
                key = ("opp_action", opp.affected_segment or opp.opportunity[:40])
                if key not in seen_keys:
                    seen_keys.add(key)
                    candidates.append(ActionCandidate(
                        action=(f"Prioritize '{opp.affected_segment or 'the opportunity'}' "
                                f"to capture the identified opportunity in {opp.affected_metric}."),
                        action_type=ActionType.PRIORITIZE,
                        objective=self._pick_objective(objectives, [RecommendationObjective.MAXIMIZE_REVENUE]),
                        affected_metric=opp.affected_metric,
                        affected_segment=opp.affected_segment,
                        rationale=opp.opportunity,
                        evidence=opp.evidence,
                        evidence_ids=opp.evidence_ids,
                        assumptions=opp.assumptions,
                        risks=opp.risks,
                        confidence=opp.confidence,
                    ))

        # ---- 5. Constraint validation & impact estimation ----
        accepted: List[ActionCandidate] = []
        for cand in candidates:
            ok, satisfied = self.validator.validate(cand, context.constraints)
            if not ok:
                continue
            cand.constraints = satisfied
            cand.expected_impact = self.impact.estimate(cand, scenarios)
            accepted.append(cand)
        return accepted

    @staticmethod
    def _pick_objective(requested: List[RecommendationObjective],
                        preferred: List[RecommendationObjective]) -> RecommendationObjective:
        """Choose the candidate's driving objective: use a requested objective when
        one matches the preferred set, otherwise the most impactful preferred one,
        otherwise UNSPECIFIED (never invent an objective)."""
        if not requested:
            return preferred[0] if preferred else RecommendationObjective.UNSPECIFIED
        for p in preferred:
            if p in requested:
                return p
        return preferred[0] if preferred else RecommendationObjective.UNSPECIFIED

class RecommendationScorer:
    """Transparent, documented recommendation scoring.

    Formulas (normalized to ~[0,1] before aggregation unless stated):
      relevance          = 1.0 if objective matched, 0.3-0.4 otherwise,
                           0.5 (neutral) when no objective was supplied.
      evidence_strength  = 0.4 + 0.12 * min(len(evidence), 5) (+0.1 confidence bonus).
      expected_impact    = 0.5 + clamp(pct_change / 20%, -0.5, 0.5); 0.5 when unavailable.
      feasibility        = 1.0 for constraint-satisfying actions.
      risk_penalty       = sum of severity weights of associated context risks (capped).
      urgency            = max severity weight of associated context risks.
    final_score = evidence_strength + expected_impact + relevance + feasibility - risk_penalty
    """

    @staticmethod
    def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, v))

    def score(self, candidate: ActionCandidate, context: DecisionContext,
              objectives: List[RecommendationObjective],
              risk_tolerance: Optional[str] = None) -> Tuple[ScoringFactors, float]:
        # relevance
        if not objectives:
            relevance = 0.5
        elif candidate.objective in objectives:
            relevance = 1.0
        elif candidate.objective == RecommendationObjective.UNSPECIFIED:
            relevance = 0.4
        else:
            relevance = 0.3

        # evidence_strength
        n_ev = len(candidate.evidence)
        evidence_strength = self._clamp(0.4 + 0.12 * min(n_ev, 5) + 0.1 * float(candidate.confidence))

        # expected_impact
        impact = candidate.expected_impact
        if impact is None or impact.availability != ImpactAvailability.AVAILABLE:
            expected_impact = 0.5
        elif impact.estimated_impact_pct is not None:
            expected_impact = self._clamp(0.5 + float(impact.estimated_impact_pct) / 20.0)
        elif impact.estimated_impact is not None and impact.baseline:
            pct = float(impact.estimated_impact) / float(impact.baseline) * 100.0
            expected_impact = self._clamp(0.5 + pct / 20.0)
        else:
            expected_impact = 0.5

        feasibility = 1.0

        # associated risk penalty / urgency
        associated = self._associated_risks(candidate, context.risks)
        penalty = sum(RiskEngine.SEVERITY_WEIGHT[r.severity] for r in associated)
        penalty = min(penalty, 1.0)
        urgency = max([RiskEngine.SEVERITY_WEIGHT[r.severity] for r in associated], default=0.0)

        # risk tolerance filter adjusts penalty up when user is low-tolerance
        if risk_tolerance and str(risk_tolerance).upper() == "LOW":
            penalty = self._clamp(penalty * 1.5, 0.0, 1.0)

        factors = ScoringFactors(
            evidence_strength=round(evidence_strength, 4),
            expected_impact=round(expected_impact, 4),
            relevance=round(relevance, 4),
            feasibility=round(feasibility, 4),
            risk_penalty=round(penalty, 4),
            urgency=round(urgency, 4),
        )
        final = (evidence_strength + expected_impact + relevance + feasibility - penalty)
        factors.final_score = round(final, 4)
        return factors, final

    @staticmethod
    def _associated_risks(candidate: ActionCandidate,
                          risks: List[RiskAssessment]) -> List[RiskAssessment]:
        cand_ev_ids = set(candidate.evidence_ids)
        seg = str(candidate.affected_segment or "").lower()
        out = []
        for r in risks:
            r_ids = set(r.evidence_ids)
            if cand_ev_ids & r_ids:
                out.append(r)
            elif seg and r.risk and seg in r.risk.lower():
                out.append(r)
        return out

    @staticmethod
    def priority_for(score: float) -> PriorityLevel:
        if score >= 3.0:
            return PriorityLevel.HIGH
        if score >= 2.0:
            return PriorityLevel.MEDIUM
        return PriorityLevel.LOW

class RecommendationEngine:
    """Orchestrates the full evidence-backed recommendation pipeline."""

    def __init__(self) -> None:
        self.builder = DecisionContextBuilder()
        self.risk_engine = RiskEngine()
        self.opportunity_engine = OpportunityEngine()
        self.action_generator = ActionGenerator()
        self.scorer = RecommendationScorer()

        def generate(self, request: RecommendationRequest) -> RecommendationResult:
        start = time.perf_counter()
        objectives = RecommendationObjective.normalize(request.optimization_objective)
        # Only parse the user intent for an objective when the user actually
        # supplied a natural-language intent. A programmatic call that simply
        # passes insights without an objective still produces evidence-backed
        # recommendations (with UNSPECIFIED objective) rather than blocking on
        # clarification. Clarification is requested only in the conversational
        # case where a user intent was provided but no objective could be parsed.
        parsed_from_intent: List[RecommendationObjective] = []
        has_user_intent = bool(request.user_intent)
        if not objectives and has_user_intent:
            parsed_from_intent = RecommendationObjective.parse_objectives(request.user_intent)

        context = self.builder.build(request)

        monitoring = [m for m in (_monitoring_to_dict(x) for x in request.monitoring_results) if m]
        scenarios = [s for s in (DecisionContextBuilder._scenario_to_dict(x) for x in request.scenarios) if s]

        risks = self.risk_engine.assess(context, monitoring=monitoring)
        opportunities = self.opportunity_engine.detect(context, scenarios)
        context.risks = risks
        context.opportunities = opportunities

        candidates = self.action_generator.generate(
            context, objectives or parsed_from_intent, scenarios, monitoring,
            risk_tolerance=request.risk_tolerance)
        context.available_actions = candidates

        recommendations: List[Recommendation] = []
        for cand in candidates:
            factors, score = self.scorer.score(
                cand, context, objectives or parsed_from_intent, request.risk_tolerance)
            recommendations.append(self._to_recommendation(cand, factors, score))

        recommendations.sort(key=lambda r: r.score, reverse=True)
        limited = recommendations[: request.max_recommendations]

        trade_offs = self._build_trade_offs(limited, objectives or parsed_from_intent)
        audit = self._build_audit(limited)

        all_evidence: List[Evidence] = []
        for r in limited:
            all_evidence.extend(r.evidence)
        for r in risks:
            all_evidence.extend(r.evidence)
        for o in opportunities:
            all_evidence.extend(o.evidence)

        expected_impacts = [r.expected_impact for r in limited
                            if r.expected_impact and r.expected_impact.availability == ImpactAvailability.AVAILABLE]

        # Clarification is only needed for the conversational case: the user
        # supplied a natural-language intent but it contained no recognizable
        # business objective.
        needs_objective = has_user_intent and not objectives and not parsed_from_intent
        confidence = self._overall_confidence(limited, risks)
        summary = self._executive_summary(limited, objectives or parsed_from_intent, risks, needs_objective)
        elapsed = round(time.perf_counter() - start, 4)

        return RecommendationResult(
            status="needs_objective" if needs_objective else "success",
            executive_summary=summary,
            recommendations=limited,
            risks=risks,
            opportunities=opportunities,
            expected_impacts=expected_impacts,
            assumptions=context.assumptions,
            limitations=context.uncertainties,
            evidence=all_evidence,
            confidence=confidence,
            execution_time=elapsed,
            objectives=objectives or parsed_from_intent,
            trade_offs=trade_offs,
            audit_trail=audit,
            needs_objective_clarification=needs_objective,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _to_recommendation(cand: ActionCandidate, factors: ScoringFactors,
                           score: float) -> Recommendation:
        why_parts = [cand.rationale] if cand.rationale else []
        if cand.expected_impact and cand.expected_impact.availability == ImpactAvailability.AVAILABLE:
            ei = cand.expected_impact
            if ei.estimated_impact_pct is not None:
                why_parts.append(f"Expected impact from scenario '{ei.scenario_name}': "
                                 f"{ei.estimated_impact:+,.2f} ({ei.estimated_impact_pct:+.1f}%).")
            else:
                why_parts.append(f"Expected impact: {ei.estimated_impact:+,.2f}.")
        why = " ".join(why_parts) if why_parts else cand.action
        rec_confidence = max(0.1, min(1.0, float(cand.confidence) * 0.95))
        return Recommendation(
            recommendation_id=f"REC-{uuid.uuid4().hex[:8].upper()}",
            action=cand.action,
            why=why,
            action_type=cand.action_type,
            objective=cand.objective,
            affected_metric=cand.affected_metric,
            affected_segment=cand.affected_segment,
            evidence=cand.evidence,
            evidence_ids=cand.evidence_ids,
            expected_impact=cand.expected_impact,
            risks=cand.risks,
            assumptions=cand.assumptions,
            limitations=[],
            confidence=round(rec_confidence, 3),
            priority=RecommendationScorer.priority_for(score),
            score=round(score, 4),
            scoring=factors,
            human_approval_required=True,
        )

    @staticmethod
    def _overall_confidence(recommendations: List[Recommendation],
                            risks: List[RiskAssessment]) -> float:
        if not recommendations:
            base = 0.5
        else:
            base = sum(r.confidence for r in recommendations) / len(recommendations)
        for r in risks:
            if r.severity in (RiskSeverity.HIGH, RiskSeverity.CRITICAL):
                base -= 0.05
        return round(max(0.0, min(1.0, base)), 3)

    def _executive_summary(self, recommendations: List[Recommendation],
                           objectives: List[RecommendationObjective],
                           risks: List[RiskAssessment],
                           needs_objective: bool) -> str:
        if needs_objective:
            top = "; ".join(f"'{r.action}' ({r.priority.value})" for r in recommendations[:3])
            return ("Evidence-backed observations generated. No business objective was specified. "
                    "What business objective should I optimize the recommendations for "
                    "(e.g. revenue, cost, retention, risk, conversion, profit, efficiency)? "
                    + (f"Top observations: {top}." if top else ""))
        obj_label = ", ".join(RecommendationObjective.label(o) for o in objectives) or "unspecified"
        summary = f"Optimizing for: {obj_label}. "
        if recommendations:
            top = recommendations[0]
            summary += (f"Top recommendation: {top.action}. Supported by "
                        f"{len(top.evidence)} evidence item(s) with {top.confidence:.0%} "
                        f"confidence ({top.priority.value} priority).")
        else:
            summary += "No evidence-backed recommendation could be generated from the current context."
        risk_high = [r for r in risks if r.severity in (RiskSeverity.HIGH, RiskSeverity.CRITICAL)]
        if risk_high:
            summary += " Key risks to monitor: " + "; ".join(r.risk for r in risk_high[:2]) + "."
        return summary

    def _build_trade_offs(self, recommendations: List[Recommendation],
                          objectives: List[RecommendationObjective]) -> List[TradeOff]:
        if len(objectives) <= 1 or len(recommendations) < 2:
            return []
        a, b = recommendations[0], recommendations[1]
        a_imp = (a.expected_impact.estimated_impact if a.expected_impact else None)
        b_imp = (b.expected_impact.estimated_impact if b.expected_impact else None)
        a_risk = "yes" if (a.risks or a.scoring.risk_penalty >= 0.25) else "low"
        b_risk = "yes" if (b.risks or b.scoring.risk_penalty >= 0.25) else "low"
        note = (f"Option A ('{a.action}') expected impact "
                f"{a_imp if a_imp is not None else 'unavailable'} with risk {a_risk}; "
                f"Option B ('{b.action}') expected impact "
                f"{b_imp if b_imp is not None else 'unavailable'} with risk {b_risk}. "
                f"Choosing the higher-impact option may increase risk. Both require human approval.")
        return [
            TradeOff(option="A", expected_impact=a.expected_impact, risk=a_risk, note=note),
            TradeOff(option="B", expected_impact=b.expected_impact, risk=b_risk,
                     note=f"Option B: {b.action}."),
        ]

    def _build_audit(self, recommendations: List[Recommendation]) -> List[AuditRecord]:
        records = []
        for r in recommendations:
            scenario_ids = []
            if r.expected_impact and r.expected_impact.scenario_id:
                scenario_ids.append(r.expected_impact.scenario_id)
            records.append(AuditRecord(
                recommendation_id=r.recommendation_id,
                input_evidence_ids=list(r.evidence_ids),
                analysis_ids=[],
                model_ids=[],
                scenario_ids=scenario_ids,
                assumptions=list(r.assumptions),
                scoring_factors=r.scoring.to_dict(),
                final_score=r.score,
            ))
        return records

