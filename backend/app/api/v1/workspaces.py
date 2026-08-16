from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from backend.app.services.workspace_service import WorkspaceService

router = APIRouter(prefix="/workspaces", tags=["workspaces"])

_workspace_service = WorkspaceService()


@router.post("")
def create_workspace(name: str, owner: str) -> dict[str, Any]:
    try:
        result = _workspace_service.create_workspace(name=name, owner=owner)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Workspace creation failed: {exc}") from exc

    return result


@router.post("/{workspace_id}/projects")
def create_project(workspace_id: str, name: str) -> dict[str, Any]:
    try:
        result = _workspace_service.create_project(workspace_id=workspace_id, name=name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Project creation failed: {exc}") from exc

    return result
