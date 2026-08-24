"""Safe Isolated Sandbox FastAPI Router."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from backend.app.core.sandbox_runtime import (
    SafeIsolatedExecutionSandbox,
    SandboxExecutionResult,
    global_sandbox,
)

router = APIRouter(prefix="/sandbox", tags=["Isolated Sandbox"])


class SandboxExecuteRequest(BaseModel):
    code: str = Field(..., description="Python & Pandas script to execute")
    dataset: Optional[List[Dict[str, Any]]] = Field(None, description="In-memory records to populate into 'df'")


@router.post("/execute")
def execute_sandboxed_code(req: SandboxExecuteRequest) -> Dict[str, Any]:
    """Execute dynamic Python code safely inside AST-restricted sandbox."""
    df = pd.DataFrame(req.dataset) if req.dataset else None
    result: SandboxExecutionResult = global_sandbox.execute_code(
        code_str=req.code,
        dataframe=df,
    )
    return result.to_dict()
