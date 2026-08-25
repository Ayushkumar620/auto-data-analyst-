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
        elif any(k in lowered for k in ("risk", "safe", "danger")):
            query_type = "risks"
        elif any(k in lowered for k in ("highest impact", "biggest impact", "largest impact",
                                        "best expected impact")):
            query_type = "highest_impact"
        elif any(k in lowered for k in ("safest", "least risk", "lowest risk")):
            query_type = "safest"
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
