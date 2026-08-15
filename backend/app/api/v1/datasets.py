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
        dataframe = service._read_dataframe(str(UPLOAD_DIR / result["dataset"]["name"]))
        profile = service.profile_dataset(
            dataframe=dataframe,
            filename=result["dataset"]["name"],
            file_type=result["dataset"]["file_type"],
            file_size=result["metadata"]["file_size"].split(" ")[0] if result["metadata"]["file_size"] else 0,
        )
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
        "profile": profile["profile"],
        "column_analysis": profile["column_analysis"],
        "numeric_analysis": profile["numeric_analysis"],
        "categorical_analysis": profile["categorical_analysis"],
        "preview": profile["preview"],
        "recommendations": profile["recommendations"],
    }
    return payload
