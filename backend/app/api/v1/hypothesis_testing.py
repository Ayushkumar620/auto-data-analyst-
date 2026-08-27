"""
Universal Hypothesis Testing & Statistical Significance FastAPI Router.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from agent.agent_result import AgentResult, AgentStatus
from agent.hypothesis_testing_agent import HypothesisTestingAgent

router = APIRouter(tags=["Hypothesis Testing"])


class HypothesisTestingRequest(BaseModel):
    dataset: Optional[List[Dict[str, Any]]] = Field(None, description="Tabular dataset records")
    feature: Optional[str] = Field(None, description="Primary numeric or categorical feature name")
    group: Optional[str] = Field(None, description="Grouping column or secondary categorical variable name")
    feature_2: Optional[str] = Field(None, description="Optional second numeric feature for paired comparisons")
    features: Optional[List[str]] = Field(None, description="Optional feature list to evaluate")
    target: Optional[str] = Field(None, description="Alias for grouping column")
    alpha: Optional[float] = Field(0.05, description="Significance threshold (0, 1)")
    paired: Optional[bool] = Field(False, description="Set True for paired observations")
    preferred_test: Optional[str] = Field(None, description="Optional preferred test method identifier")


@router.post("/hypothesis-testing")
@router.post("/hypothesis-testing/run")
def run_hypothesis_testing(req: HypothesisTestingRequest) -> Dict[str, Any]:
    """Execute autonomous data-driven hypothesis testing and statistical significance analysis."""
    if not req.dataset or len(req.dataset) == 0:
        raise HTTPException(
            status_code=400,
            detail="Dataset records are required for hypothesis testing. Provided dataset is empty.",
        )

    df = pd.DataFrame(req.dataset)

    agent = HypothesisTestingAgent()
    task = {
        "data": df,
        "feature": req.feature,
        "group": req.group or req.target,
        "feature_2": req.feature_2,
        "features": req.features,
        "alpha": req.alpha if req.alpha is not None and 0.0 < req.alpha < 1.0 else 0.05,
        "paired": req.paired,
        "preferred_test": req.preferred_test,
    }
    result: AgentResult = agent.run(task)

    if not result.is_success:
        err_msg = result.error_message or "Hypothesis testing failed."
        raise HTTPException(
            status_code=400 if result.status == AgentStatus.VALIDATION_FAILED else 422,
            detail=err_msg,
        )

    return result.to_dict()