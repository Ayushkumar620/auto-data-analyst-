"""Multi-Modal Computer Vision Feature Extraction FastAPI Router."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import numpy as np

from backend.app.core.vision_engine import (
    ComputerVisionFeatureEngine,
    VisionExtractionReport,
    global_vision_engine,
)

router = APIRouter(prefix="/vision", tags=["Computer Vision"])


class VisionExtractBatchRequest(BaseModel):
    # List of 2D or 3D numeric grids represented as nested lists
    images: List[List[List[float]]] = Field(..., description="Batch of image intensity matrices")
    labels: Optional[List[str]] = Field(None, description="Optional class labels")


@router.post("/extract")
def extract_visual_features(req: VisionExtractBatchRequest) -> Dict[str, Any]:
    """Extract spatial gradients, textures, color moments, and quality scores from image batch."""
    if not req.images:
        raise HTTPException(status_code=400, detail="Images list cannot be empty.")

    try:
        np_images = [np.array(img, dtype=np.float32) for img in req.images]
        report: VisionExtractionReport = global_vision_engine.extract_batch_to_dataframe(
            images=np_images,
            labels=req.labels,
        )
        return report.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Vision extraction failed: {str(e)}")
