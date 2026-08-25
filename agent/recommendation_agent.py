"""
Master Autonomous Decision & Recommendation Agent (Milestone 5, Task 4).

Wraps the deterministic RecommendationEngine as a standardized BaseAgent and adds
conversational decision-support integration:

    User: "What should I do?"              -> generate recommendations from context
    User: "How can I increase revenue?"    -> optimize for revenue
    User: "What is the safest option?"     -> rank by risk
    User: "What has the highest expected impact?" -> rank by estimated impact
    User: "Why are you recommending this?" -> explain evidence
    User: "What are the risks?"            -> return RiskAssessment(s)

The agent is fully deterministic and requires no external LLM/API keys. The LLM
boundary is respected: it only translates structured, evidence-backed results and
never invents impact, risk, constraints, metrics, or causality.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from agent.base import BaseAgent
from agent.recommendation_engine import RecommendationEngine
from agent.recommendation_schemas import (
    ImpactAvailability,
    RecommendationObjective,
    RecommendationRequest,
    RecommendationResult,
)
from agent.schemas import AgentResult


class RecommendationAgent(BaseAgent):
    """Standardized BaseAgent exposing the recommendation engine + conversation."""

    name = "Recommendation Agent"
    role = "decision_support_advisor"
    description = ("Generates evidence-backed, ranked business recommendations with "
                   "risks, opportunities, expected impact, confidence, and audit trail. "
                   "Advisory only; never executes real-world actions.")

    def __init__(self, data: Optional[Any] = None):
        super().__init__(data=data)
        self.engine = RecommendationEngine()

    # ------------------------------------------------------------------
    # Core generation API
    # ------------------------------------------------------------------
    def generate(self, request: RecommendationRequest) -> RecommendationResult:
        return self.engine.generate(request)

    def generate_from_data(self,
                           user_intent: Optional[str] = None,
                           insights: Optional[List[Any]] = None,
                           forecasts: Optional[List[Any]] = None,
                           scenarios: Optional[List[Any]] = None,
                           monitoring_results: Optional[List[Any]] = None,
                           business_constraints: Optional[List[Any]] = None,
                           objective: Optional[Any] = None,
                           dataset_context: Optional[Any] = None,
                           max_recommendations: int = 5,
                           risk_tolerance: Optional[str] = None) -> RecommendationResult:
        request = RecommendationRequest(
            user_intent=user_intent,
            dataset_context=dataset_context,
            insights=insights or [],
            forecasts=forecasts or [],
            scenarios=scenarios or [],
            monitoring_results=monitoring_results or [],
            business_constraints=business_constraints or [],
            optimization_objective=objective,
            risk_tolerance=risk_tolerance,
            max_recommendations=max_recommendations,
        )
        return self.engine.generate(request)

    # ------------------------------------------------------------------
    # Conversational integration
    # ------------------------------------------------------------------
    def understand_intent(self, text: str) -> Dict[str, Any]:
        """Classify a conversational query into a decision-support operation."""
        lowered = " ".join(str(text).lower().split())
        query_type = "recommend"
        if any(k in lowered for k in ("why", "explain", "reason")):
            query_type = "explain"
        elif any(k in lowered for k in ("highest impact", "biggest impact", "largest impact",
                                        "best expected impact")):
            query_type = "highest_impact"
        elif any(k in lowered for k in ("safest", "least risk", "lowest risk", "safest option")):
            query_type = "safest"
        elif any(k in lowered for k in ("risk", "safe", "danger")):
            query_type = "risks"
        objectives = RecommendationObjective.parse_objectives(text)
        return {"query_type": query_type, "objectives": list(objectives)}

    def handle_query(self, text: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a decision-support conversational turn deterministically."""
        info = self.understand_intent(text)
        gen_data = dict(request_data)
        if info["objectives"]:
            gen_data["optimization_objective"] = info["objectives"]
        request = RecommendationRequest(**gen_data)
        result = self.engine.generate(request)
        q = info["query_type"]
        if q == "risks":
            return self._respond_risks(result)
        if q == "explain":
            return self._respond_explain(result)
        if q == "safest":
            return self._respond_rank(result, by="risk")
        if q == "highest_impact":
            return self._respond_rank(result, by="impact")
        return self._respond_recommend(result)
    # ------------------------------------------------------------------
    # Response composers
    # ------------------------------------------------------------------
    @staticmethod
    def _respond_recommend(result: RecommendationResult) -> Dict[str, Any]:
        lines = ["Decision support results:"]
        if result.needs_objective_clarification:
            lines.append(result.executive_summary)
        for r in result.recommendations:
            lines.append(f"- [{r.priority.value}] {r.action} (confidence {r.confidence:.0%})")
        return {"response": "\n".join(lines), "result": result.to_dict(),
                "query_type": "recommend"}

    @staticmethod
    def _respond_risks(result: RecommendationResult) -> Dict[str, Any]:
        lines = ["Assessed risks (evidence-backed):"]
        if not result.risks:
            lines.append("No material risks identified from current evidence.")
        for risk in result.risks:
            lines.append(f"- [{risk.severity.value}] {risk.risk} "
                         f"(confidence {risk.confidence:.0%})")
        return {"response": "\n".join(lines),
                "risks": [r.to_dict() for r in result.risks],
                "result": result.to_dict(), "query_type": "risks"}

    @staticmethod
    def _respond_explain(result: RecommendationResult) -> Dict[str, Any]:
        lines = ["Recommendation explanations (WHAT / WHY / EVIDENCE / IMPACT / RISKS / CONFIDENCE):"]
        for r in result.recommendations:
            lines.append(f"WHAT: {r.action}")
            lines.append(f"WHY: {r.why}")
            ev = "; ".join(r.evidence_ids) if r.evidence_ids else "none"
            lines.append(f"EVIDENCE: {len(r.evidence)} item(s) {ev}")
            imp = (f"{r.expected_impact.estimated_impact:+,.2f}" if
                   r.expected_impact and r.expected_impact.availability == ImpactAvailability.AVAILABLE
                   else "unavailable")
            lines.append(f"EXPECTED IMPACT: {imp}")
            lines.append(f"RISKS: {', '.join(r.risks) if r.risks else 'none'}")
            lines.append(f"CONFIDENCE: {r.confidence:.0%}")
            lines.append("---")
        return {"response": "\n".join(lines), "query_type": "explain",
                "result": result.to_dict()}

    @staticmethod
    def _respond_rank(result: RecommendationResult, by: str) -> Dict[str, Any]:
        recs = list(result.recommendations)
        if by == "risk":
            recs.sort(key=lambda r: r.scoring.risk_penalty)
            label = "lowest risk"
        else:
            def imp(r):
                if r.expected_impact and r.expected_impact.estimated_impact is not None:
                    return r.expected_impact.estimated_impact
                return -1e18
            recs.sort(key=imp, reverse=True)
            label = "highest expected impact"
        lines = [f"Ranked by {label}:"]
        for r in recs:
            v = (f"{r.expected_impact.estimated_impact:+,.2f}" if
                 r.expected_impact and r.expected_impact.estimated_impact is not None
                 else "unavailable")
            lines.append(f"- {r.action} | impact: {v} | risk penalty: {r.scoring.risk_penalty:.2f}")
        return {"response": "\n".join(lines), "query_type": "rank",
                "result": result.to_dict(), "ranked_by": by}

    # ------------------------------------------------------------------
    # Standardized BaseAgent run()
    # ------------------------------------------------------------------
    def run(self, task: Union[str, Dict[str, Any]]) -> AgentResult:
        self._start()
        try:
            if isinstance(task, str):
                task = {"command": task}
            task = dict(task or {})
            mode = task.get("mode", task.get("query_type", "recommend"))
            command = task.get("command") or task.get("user_intent") or task.get("query") or ""

            data_map: Dict[str, Any] = {
                "user_intent": command or None,
                "insights": task.get("insights") or [],
                "forecasts": task.get("forecasts") or [],
                "scenarios": task.get("scenarios") or [],
                "monitoring_results": task.get("monitoring_results") or [],
                "business_constraints": task.get("business_constraints") or [],
                "optimization_objective": task.get("objective") or task.get("optimization_objective"),
                "dataset_context": (task.get("dataset_context") or
                                    task.get("dataset_knowledge") or task.get("knowledge")),
                "max_recommendations": int(task.get("max_recommendations", 5)),
                "risk_tolerance": task.get("risk_tolerance"),
            }

            if mode in ("explain", "risks", "safest", "highest_impact", "rank") and command:
                resp = self.handle_query(command, data_map)
                conf = float(resp.get("result", {}).get("confidence", 0.8)) if isinstance(resp.get("result"), dict) else 0.8
                return self._finish(
                    resp,
                    message="RecommendationAgent processed conversational decision query.",
                    evidence=[],
                    confidence=conf,
                    metadata={"query_type": resp.get("query_type", mode)},
                )

            result = self.generate_from_data(**data_map)
            return self._finish(
                result.to_dict(),
                message=(f"{self.name} generated {len(result.recommendations)} evidence-backed "
                         f"recommendation(s). Status: {result.status}."),
                evidence=result.evidence,
                confidence=result.confidence,
                metadata={"status": result.status,
                          "needs_objective": result.needs_objective_clarification},
            )
        except Exception as exc:
            return self._error(
                f"Recommendation generation failed: {str(exc).splitlines()[-1]}",
                suggested_fix=("Ensure insights/forecasts/scenarios are in the "
                               "expected format."),
            )

