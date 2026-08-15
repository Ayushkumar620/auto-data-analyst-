from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.config import UPLOAD_DIR
from backend.app.services.dataset_service import DatasetService

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("/")
def list_datasets() -> dict[str, Any]:
    return {"datasets": []}


@router.post("/upload")
def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    service = DatasetService(upload_folder=str(UPLOAD_DIR))
    try:
        result = service.upload_dataset(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc

    metadata = result["metadata"]
    payload = {
        "status": "uploaded",
        "dataset": result["dataset"],
        "rows": result["dataset"]["rows"],
        "columns": result["dataset"]["columns"],
        "file_type": result["dataset"]["file_type"],
        "column_names": metadata["column_names"],
        "data_types": metadata["data_types"],
        "missing_values": metadata["missing_values"],
        "duplicate_rows": metadata["duplicate_rows"],
        "memory_usage": metadata["memory_usage"],
        "preview": result["preview"],
    }
    return payload
