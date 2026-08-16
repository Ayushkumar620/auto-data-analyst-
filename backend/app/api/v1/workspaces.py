from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.services.workspace_service import WorkspaceService

router = APIRouter(prefix="/workspaces", tags=["workspaces"])

_workspace_service = WorkspaceService()


class WorkspaceCreate(BaseModel):
    name: str
    owner: str


class ProjectCreate(BaseModel):
    name: str


@router.post("")
def create_workspace(workspace: WorkspaceCreate) -> dict[str, Any]:
    try:
        result = _workspace_service.create_workspace(name=workspace.name, owner=workspace.owner)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Workspace creation failed: {exc}") from exc

    return result


@router.post("/{workspace_id}/projects")
def create_project(workspace_id: str, project: ProjectCreate) -> dict[str, Any]:
    try:
        result = _workspace_service.create_project(workspace_id=workspace_id, name=project.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Project creation failed: {exc}") from exc

    return result
