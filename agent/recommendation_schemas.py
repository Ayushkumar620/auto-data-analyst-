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

    _ALIASES = {
        "revenue": MAXIMIZE_REVENUE, "grow revenue": MAXIMIZE_REVENUE,
        "increase revenue": MAXIMIZE_REVENUE, "maximize revenue": MAXIMIZE_REVENUE,
        "sales": MAXIMIZE_REVENUE, "increase sales": MAXIMIZE_REVENUE,
        "cost": REDUCE_COST, "reduce cost": REDUCE_COST, "cut cost": REDUCE_COST,
        "retention": IMPROVE_RETENTION, "improve retention": IMPROVE_RETENTION,
        "keep customers": IMPROVE_RETENTION,
        "risk": REDUCE_RISK, "reduce risk": REDUCE_RISK, "safer": REDUCE_RISK, "safest": REDUCE_RISK,
        "conversion": IMPROVE_CONVERSION, "improve conversion": IMPROVE_CONVERSION,
        "profit": INCREASE_PROFIT, "profitability": INCREASE_PROFIT,
        "increase profit": INCREASE_PROFIT, "improve profitability": INCREASE_PROFIT,
        "efficiency": IMPROVE_OPERATIONAL_EFFICIENCY,
        "operational efficiency": IMPROVE_OPERATIONAL_EFFICIENCY,
        "efficient": IMPROVE_OPERATIONAL_EFFICIENCY,
    }

    @classmethod
    def from_text(cls, text: Optional[str]) -> Optional["RecommendationObjective"]:
        """Resolve a natural-language objective phrase. Returns None when no match."""
        if not text:
            return None
        lowered = " ".join(str(text).lower().split())
        for phrase, obj in cls._ALIASES.items():
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
        for phrase, obj in cls._ALIASES.items():
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
