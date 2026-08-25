"""
Schemas for the Autonomous Decision & Recommendation Engine (Milestone 5, Task 4).

Type-safe Pydantic v2 contracts for RecommendationRequest, DecisionContext,
RiskAssessment, OpportunityAssessment, ActionCandidate, Recommendation,
ScoringFactors, TradeOff, AuditRecord and RecommendationResult.

The engine enforces STRICT separation between facts, predictions, recommendations
and assumptions — these models never merge those categories.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import uuid

from pydantic import BaseModel, ConfigDict, Field

from agent.schemas import ClaimType, Evidence


class RecommendationObjective(str, Enum):
    """Business objectives the engine can optimize toward.

    The engine NEVER invents an objective: if the user has not supplied one it
    returns evidence-backed observations and asks for the objective.
    """
    MAXIMIZE_REVENUE = "maximize_revenue"
    REDUCE_COST = "reduce_cost"
    IMPROVE_RETENTION = "improve_retention"
    REDUCE_RISK = "reduce_risk"
    IMPROVE_CONVERSION = "improve_conversion"
    INCREASE_PROFIT = "increase_profit"
    IMPROVE_OPERATIONAL_EFFICIENCY = "improve_operational_efficiency"
    UNSPECIFIED = "unspecified"

    @classmethod
    def from_text(cls, text: Optional[str]) -> Optional["RecommendationObjective"]:
        """Resolve a natural-language objective phrase. Returns None when no match."""
        if not text:
            return None
        lowered = " ".join(str(text).lower().split())
        for phrase, obj in OBJECTIVE_ALIASES.items():
            if phrase in lowered:
                return obj
        return None

    @classmethod
    def parse_objectives(cls, text: Optional[str]) -> List["RecommendationObjective"]:
        """Parse one or more objectives from freeform text."""
        if not text:
            return []
        lowered = " ".join(str(text).lower().split())
        found: List["RecommendationObjective"] = []
        for phrase, obj in OBJECTIVE_ALIASES.items():
            if phrase in lowered and obj not in found:
                found.append(obj)
        order = {o: i for i, o in enumerate(cls)}
        found.sort(key=lambda o: order[o])
        return found

    @classmethod
    def normalize(cls, value: Any) -> List["RecommendationObjective"]:
        """Normalize an objective (enum, str, list) into a list of objectives."""
        if value is None or value == "":
            return []
        if isinstance(value, cls):
            return [] if value == cls.UNSPECIFIED else [value]
        if isinstance(value, str):
            try:
                obj = cls(value)
                return [] if obj == cls.UNSPECIFIED else [obj]
            except ValueError:
                return cls.parse_objectives(value)
        if isinstance(value, (list, tuple, set)):
            order = {o: i for i, o in enumerate(cls)}
            deduped: List["RecommendationObjective"] = []
            for item in value:
                for obj in cls.normalize(item):
                    if obj not in deduped:
                        deduped.append(obj)
            deduped.sort(key=lambda o: order[o])
            return deduped
        return []

    @classmethod
    def label(cls, objective: Optional["RecommendationObjective"]) -> str:
        """Human readable label for display / explanation."""
        if objective is None:
            return "Not specified"
        return {
            cls.MAXIMIZE_REVENUE: "Revenue growth",
            cls.REDUCE_COST: "Cost reduction",
            cls.IMPROVE_RETENTION: "Customer retention",
            cls.REDUCE_RISK: "Risk reduction",
            cls.IMPROVE_CONVERSION: "Conversion improvement",
            cls.INCREASE_PROFIT: "Profitability",
            cls.IMPROVE_OPERATIONAL_EFFICIENCY: "Operational efficiency",
            cls.UNSPECIFIED: "Not specified",
        }[objective]



# Natural-language -> objective alias map (module scope so the enum stays clean).
OBJECTIVE_ALIASES: Dict[str, "RecommendationObjective"] = {
    "revenue": RecommendationObjective.MAXIMIZE_REVENUE,
    "grow revenue": RecommendationObjective.MAXIMIZE_REVENUE,
    "increase revenue": RecommendationObjective.MAXIMIZE_REVENUE,
    "maximize revenue": RecommendationObjective.MAXIMIZE_REVENUE,
    "sales": RecommendationObjective.MAXIMIZE_REVENUE,
    "increase sales": RecommendationObjective.MAXIMIZE_REVENUE,
    "cost": RecommendationObjective.REDUCE_COST,
    "reduce cost": RecommendationObjective.REDUCE_COST,
    "cut cost": RecommendationObjective.REDUCE_COST,
    "retention": RecommendationObjective.IMPROVE_RETENTION,
    "improve retention": RecommendationObjective.IMPROVE_RETENTION,
    "keep customers": RecommendationObjective.IMPROVE_RETENTION,
    "risk": RecommendationObjective.REDUCE_RISK,
    "reduce risk": RecommendationObjective.REDUCE_RISK,
    "safer": RecommendationObjective.REDUCE_RISK,
    "safest": RecommendationObjective.REDUCE_RISK,
    "conversion": RecommendationObjective.IMPROVE_CONVERSION,
    "improve conversion": RecommendationObjective.IMPROVE_CONVERSION,
    "profit": RecommendationObjective.INCREASE_PROFIT,
    "profitability": RecommendationObjective.INCREASE_PROFIT,
    "increase profit": RecommendationObjective.INCREASE_PROFIT,
    "improve profitability": RecommendationObjective.INCREASE_PROFIT,
    "efficiency": RecommendationObjective.IMPROVE_OPERATIONAL_EFFICIENCY,
    "operational efficiency": RecommendationObjective.IMPROVE_OPERATIONAL_EFFICIENCY,
    "efficient": RecommendationObjective.IMPROVE_OPERATIONAL_EFFICIENCY,
}



class ActionType(str, Enum):
    INVESTIGATE = "investigate"
    PRIORITIZE = "prioritize"
    INCREASE = "increase"
    DECREASE = "decrease"
    RETAIN = "retain"
    EXPAND = "expand"
    OPTIMIZE = "optimize"
    MONITOR = "monitor"
    TEST = "test"
    SEGMENT = "segment"
    REALLOCATE = "reallocate"
    REVIEW = "review"


class RiskSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class PriorityLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ImpactAvailability(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class ConstraintType(str, Enum):
    SPENDING_LIMIT = "spending_limit"
    BUDGET_LIMIT = "budget_limit"
    REGION_FILTER = "region_filter"
    RETENTION_MIN = "retention_min"
    SEGMENT_FILTER = "segment_filter"
    OTHER = "other"

class Fact(BaseModel):
    """A deterministic, evidence-backed observed true statement about the data.

    THIS IS NOT A RECOMMENDATION. It is an observed fact (e.g. 'Revenue increased 18%').
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    fact_id: str = Field(default_factory=lambda: f"FACT-{uuid.uuid4().hex[:8].upper()}")
    statement: str
    claim_type: ClaimType = ClaimType.FACT
    category: str = "fact"
    value: Optional[float] = None
    affected_metric: Optional[str] = None
    affected_segment: Optional[str] = None
    evidence: List[Evidence] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "statement": self.statement,
            "claim_type": self.claim_type.value,
            "category": self.category,
            "value": round(float(self.value), 4) if self.value is not None else None,
            "affected_metric": self.affected_metric,
            "affected_segment": self.affected_segment,
            "evidence": [e.to_dict() for e in self.evidence],
            "confidence": round(float(self.confidence), 3),
        }


class Prediction(BaseModel):
    """A forward-looking estimate produced by a deterministic forecasting model.

    THIS IS A PREDICTION, NOT A FACT AND NOT A RECOMMENDATION.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prediction_id: str = Field(default_factory=lambda: f"PRED-{uuid.uuid4().hex[:8].upper()}")
    statement: str
    metric: str
    direction: str = "stable"  # "up" | "down" | "stable"
    change_percent: Optional[float] = None
    horizon: Optional[int] = None
    model_id: Optional[str] = None
    evidence: List[Evidence] = Field(default_factory=list)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "statement": self.statement,
            "metric": self.metric,
            "direction": self.direction,
            "change_percent": round(float(self.change_percent), 4) if self.change_percent is not None else None,
            "horizon": self.horizon,
            "model_id": self.model_id,
            "evidence": [e.to_dict() for e in self.evidence],
            "confidence": round(float(self.confidence), 3),
        }


class ExpectedImpact(BaseModel):
    """A numerical impact estimate. The number MUST come from a real model/scenario
    computation. When it cannot be quantified, availability is 'unavailable' and the
    engine never invents an estimate.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    availability: ImpactAvailability = ImpactAvailability.UNAVAILABLE
    metric: Optional[str] = None
    baseline: Optional[float] = None
    scenario: Optional[float] = None
    estimated_impact: Optional[float] = None  # absolute impact where computable
    estimated_impact_pct: Optional[float] = None  # relative impact where computable
    scenario_name: Optional[str] = None
    scenario_id: Optional[str] = None
    assumptions: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "availability": self.availability.value,
            "metric": self.metric,
            "baseline": round(float(self.baseline), 4) if self.baseline is not None else None,
            "scenario": round(float(self.scenario), 4) if self.scenario is not None else None,
            "estimated_impact": round(float(self.estimated_impact), 4) if self.estimated_impact is not None else None,
            "estimated_impact_pct": round(float(self.estimated_impact_pct), 4) if self.estimated_impact_pct is not None else None,
            "scenario_name": self.scenario_name,
            "scenario_id": self.scenario_id,
            "assumptions": self.assumptions,
            "uncertainties": self.uncertainties,
        }


class DecisionConstraint(BaseModel):
    """An explicit constraint the user provided. Recommendations violating an
    explicit constraint MUST be rejected.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    constraint: str
    type: Union[ConstraintType, str] = ConstraintType.OTHER
    value: Any = None
    source: str = "user"
    validation_status: str = "active"  # "active" | "satisfied" | "rejected"

    def to_dict(self) -> Dict[str, Any]:
        ct = self.type.value if isinstance(self.type, ConstraintType) else str(self.type)
        return {
            "constraint": self.constraint,
            "type": ct,
            "value": self.value,
            "source": self.source,
            "validation_status": self.validation_status,
        }


class RiskAssessment(BaseModel):
    """An evidence-backed risk. Only created when supporting evidence exists."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    risk_id: str = Field(default_factory=lambda: f"RISK-{uuid.uuid4().hex[:8].upper()}")
    risk: str
    severity: RiskSeverity = RiskSeverity.MEDIUM
    probability: Optional[float] = None  # 0..1 where estimable
    impact: Optional[str] = None
    evidence: List[Evidence] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    mitigation: Optional[str] = None
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_id": self.risk_id,
            "risk": self.risk,
            "severity": self.severity.value,
            "probability": round(float(self.probability), 3) if self.probability is not None else None,
            "impact": self.impact,
            "evidence": [e.to_dict() for e in self.evidence],
            "evidence_ids": self.evidence_ids,
            "mitigation": self.mitigation,
            "confidence": round(float(self.confidence), 3),
        }


class OpportunityAssessment(BaseModel):
    """An evidence-backed opportunity (growing product, underpenetrated segment,
    high-value group, improving channel, strong regional growth, ...). Never
    labeled an opportunity without measurable evidence.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    opportunity: str
    evidence: List[Evidence] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    expected_impact: Optional[ExpectedImpact] = None
    affected_segment: Optional[str] = None
    affected_metric: Optional[str] = None
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    assumptions: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "opportunity": self.opportunity,
            "evidence": [e.to_dict() for e in self.evidence],
            "evidence_ids": self.evidence_ids,
            "expected_impact": self.expected_impact.to_dict() if self.expected_impact else None,
            "affected_segment": self.affected_segment,
            "affected_metric": self.affected_metric,
            "confidence": round(float(self.confidence), 3),
            "assumptions": self.assumptions,
            "risks": self.risks,
        }


class ActionCandidate(BaseModel):
    """A discovered, evidence-tied course of action. Never executed automatically."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    action_id: str = Field(default_factory=lambda: f"ACT-{uuid.uuid4().hex[:8].upper()}")
    action: str
    action_type: ActionType = ActionType.REVIEW
    objective: RecommendationObjective = RecommendationObjective.UNSPECIFIED
    affected_metric: Optional[str] = None
    affected_segment: Optional[str] = None
    rationale: str = ""
    expected_impact: Optional[ExpectedImpact] = None
    evidence: List[Evidence] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    constraints: List[DecisionConstraint] = Field(default_factory=list)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action": self.action,
            "action_type": self.action_type.value,
            "objective": self.objective.value,
            "affected_metric": self.affected_metric,
            "affected_segment": self.affected_segment,
            "rationale": self.rationale,
            "expected_impact": self.expected_impact.to_dict() if self.expected_impact else None,
            "evidence": [e.to_dict() for e in self.evidence],
            "evidence_ids": self.evidence_ids,
            "assumptions": self.assumptions,
            "risks": self.risks,
            "constraints": [c.to_dict() for c in self.constraints],
            "confidence": round(float(self.confidence), 3),
        }


class ScoringFactors(BaseModel):
    """Transparent, documented scoring factors behind a recommendation score.

    Formulae (all normalized to ~[0,1] before aggregation unless stated):
      relevance    = how well the action matches the requested objective(s)
      evidence_strength = strength/volume of supporting evidence
      expected_impact   = normalized expected impact magnitude (0.5 when unavailable)
      feasibility       = whether explicit constraints are satisfied
      urgency           = derived from the severity of related risks
      risk_penalty      = aggregate penalty from the candidate's risks
    final_score = evidence_strength + expected_impact + relevance + feasibility - risk_penalty
    """
    evidence_strength: float = 0.0
    expected_impact: float = 0.0
    relevance: float = 0.0
    feasibility: float = 0.0
    risk_penalty: float = 0.0
    urgency: float = 0.0
    final_score: float = 0.0
    formula: str = "evidence_strength + expected_impact + relevance + feasibility - risk_penalty"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_strength": round(float(self.evidence_strength), 4),
            "expected_impact": round(float(self.expected_impact), 4),
            "relevance": round(float(self.relevance), 4),
            "feasibility": round(float(self.feasibility), 4),
            "risk_penalty": round(float(self.risk_penalty), 4),
            "urgency": round(float(self.urgency), 4),
            "final_score": round(float(self.final_score), 4),
            "formula": self.formula,
        }


class Recommendation(BaseModel):
    """A fully-formed recommendation. Answers WHAT / WHY / EVIDENCE /
    EXPECTED IMPACT / RISKS / ASSUMPTIONS / CONFIDENCE.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    recommendation_id: str = Field(default_factory=lambda: f"REC-{uuid.uuid4().hex[:8].upper()}")
    action: str
    why: str
    action_type: ActionType = ActionType.REVIEW
    objective: RecommendationObjective = RecommendationObjective.UNSPECIFIED
    affected_metric: Optional[str] = None
    affected_segment: Optional[str] = None
    evidence: List[Evidence] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    expected_impact: Optional[ExpectedImpact] = None
    risks: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    priority: PriorityLevel = PriorityLevel.LOW
    score: float = 0.0
    scoring: ScoringFactors = Field(default_factory=ScoringFactors)
    human_approval_required: bool = True  # advisory: never auto-executed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommendation_id": self.recommendation_id,
            "action": self.action,
            "why": self.why,
            "action_type": self.action_type.value,
            "objective": self.objective.value,
            "affected_metric": self.affected_metric,
            "affected_segment": self.affected_segment,
            "evidence": [e.to_dict() for e in self.evidence],
            "evidence_ids": self.evidence_ids,
            "expected_impact": self.expected_impact.to_dict() if self.expected_impact else None,
            "risks": self.risks,
            "assumptions": self.assumptions,
            "limitations": self.limitations,
            "confidence": round(float(self.confidence), 3),
            "priority": self.priority.value,
            "score": round(float(self.score), 4),
            "scoring": self.scoring.to_dict(),
            "human_approval_required": self.human_approval_required,
        }


class TradeOff(BaseModel):
    """A transparent trade-off between competing options for multiple objectives."""
    option: str
    expected_impact: Optional[ExpectedImpact] = None
    risk: str = ""
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "option": self.option,
            "expected_impact": self.expected_impact.to_dict() if self.expected_impact else None,
            "risk": self.risk,
            "note": self.note,
        }


class AuditRecord(BaseModel):
    """Recommendation explainability / audit trail."""
    recommendation_id: str
    input_evidence_ids: List[str] = Field(default_factory=list)
    analysis_ids: List[str] = Field(default_factory=list)
    model_ids: List[str] = Field(default_factory=list)
    scenario_ids: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    scoring_factors: Dict[str, Any] = Field(default_factory=dict)
    final_score: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommendation_id": self.recommendation_id,
            "input_evidence_ids": self.input_evidence_ids,
            "analysis_ids": self.analysis_ids,
            "model_ids": self.model_ids,
            "scenario_ids": self.scenario_ids,
            "assumptions": self.assumptions,
            "scoring_factors": self.scoring_factors,
            "final_score": round(float(self.final_score), 4),
            "timestamp": self.timestamp.isoformat(),
        }


class DecisionContext(BaseModel):
    """The single source of truth from which every recommendation is generated.

    Holds facts, insights, predictions, risks, opportunities, constraints,
    assumptions, uncertainties, and available actions in strict categories.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    facts: List[Fact] = Field(default_factory=list)
    insights: List[Any] = Field(default_factory=list)
    predictions: List[Prediction] = Field(default_factory=list)
    risks: List[RiskAssessment] = Field(default_factory=list)
    opportunities: List[OpportunityAssessment] = Field(default_factory=list)
    constraints: List[DecisionConstraint] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    available_actions: List[ActionCandidate] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": [f.to_dict() for f in self.facts],
            "insights": [i.to_dict() if hasattr(i, "to_dict") else i for i in self.insights],
            "predictions": [p.to_dict() for p in self.predictions],
            "risks": [r.to_dict() for r in self.risks],
            "opportunities": [o.to_dict() for o in self.opportunities],
            "constraints": [c.to_dict() for c in self.constraints],
            "assumptions": self.assumptions,
            "uncertainties": self.uncertainties,
            "available_actions": [a.to_dict() for a in self.available_actions],
        }


class RecommendationRequest(BaseModel):
    """Specification for a recommendation-generation request."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_intent: Optional[str] = None
    dataset_context: Optional[Any] = None                     # DatasetKnowledge
    insights: List[Any] = Field(default_factory=list)         # Insight objects / dicts
    forecasts: List[Any] = Field(default_factory=list)        # ForecastResult objects / dicts
    scenarios: List[Any] = Field(default_factory=list)        # ScenarioResult objects / dicts
    model_results: List[Any] = Field(default_factory=list)
    monitoring_results: List[Any] = Field(default_factory=list)  # MonitoringResult objects / dicts
    business_constraints: List[Any] = Field(default_factory=list)  # DecisionConstraint / dict
    optimization_objective: Optional[Any] = None              # str | RecommendationObjective | list
    risk_tolerance: Optional[str] = None                      # "LOW" | "MEDIUM" | "HIGH"
    max_recommendations: int = Field(default=5, ge=1, le=20)


class RecommendationResult(BaseModel):
    """Complete output of the recommendation engine."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    status: str = "success"  # "success" | "needs_objective" | "partial" | "failed"
    executive_summary: str = ""
    recommendations: List[Recommendation] = Field(default_factory=list)
    risks: List[RiskAssessment] = Field(default_factory=list)
    opportunities: List[OpportunityAssessment] = Field(default_factory=list)
    expected_impacts: List[ExpectedImpact] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    execution_time: float = 0.0
    objectives: List[RecommendationObjective] = Field(default_factory=list)
    trade_offs: List[TradeOff] = Field(default_factory=list)
    audit_trail: List[AuditRecord] = Field(default_factory=list)
    needs_objective_clarification: bool = False
    human_approval_required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "executive_summary": self.executive_summary,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "risks": [r.to_dict() for r in self.risks],
            "opportunities": [o.to_dict() for o in self.opportunities],
            "expected_impacts": [e.to_dict() for e in self.expected_impacts],
            "assumptions": self.assumptions,
            "limitations": self.limitations,
            "evidence": [e.to_dict() for e in self.evidence],
            "confidence": round(float(self.confidence), 3),
            "execution_time": round(float(self.execution_time), 4),
            "objectives": [o.value for o in self.objectives],
            "trade_offs": [t.to_dict() for t in self.trade_offs],
            "audit_trail": [a.to_dict() for a in self.audit_trail],
            "needs_objective_clarification": self.needs_objective_clarification,
            "human_approval_required": self.human_approval_required,
        }

