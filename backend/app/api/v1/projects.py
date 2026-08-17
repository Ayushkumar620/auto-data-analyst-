"""Project and dataset management backed by PostgreSQL (Phase M)."""

from __future__ import annotations

from typing import Annotated
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.auth.router import CurrentUser
from backend.app.database import get_db
from backend.app.models import Dataset, Project, User

router = APIRouter(prefix="/projects", tags=["projects"])

Db = Annotated[Session, Depends(get_db)]


class ProjectCreate(BaseModel):
    name: str
    description: str | None = None


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None


def _project_out(project: Project) -> dict[str, Any]:
    return {
        "id": project.id,
        "name": project.name,
        "description": project.description,
        "created_at": project.created_at,
        "datasets": [_dataset_out(d) for d in project.datasets],
    }


def _dataset_out(dataset: Dataset) -> dict[str, Any]:
    return {
        "id": dataset.id,
        "name": dataset.name,
        "file_type": dataset.file_type,
        "rows": dataset.rows,
        "columns": dataset.columns,
        "created_at": dataset.created_at,
    }


def _get_owned_project(db: Session, owner: User, project_id: int) -> Project:
    project = db.get(Project, project_id)
    if project is None or (project.owner_id is not None and project.owner_id != owner.id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found.")
    return project


@router.get("")
def list_projects(db: Db, current: CurrentUser) -> dict[str, Any]:
    projects = db.query(Project).filter(
        (Project.owner_id == current.id) | (Project.owner_id.is_(None))
    ).all()
    return {"projects": [_project_out(p) for p in projects]}


@router.post("", status_code=status.HTTP_201_CREATED)
def create_project(payload: ProjectCreate, db: Db, current: CurrentUser) -> dict[str, Any]:
    project = Project(
        name=payload.name,
        description=payload.description,
        owner_id=current.id,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return _project_out(project)


@router.get("/{project_id}")
def get_project(project_id: int, db: Db, current: CurrentUser) -> dict[str, Any]:
    return _project_out(_get_owned_project(db, current, project_id))


@router.patch("/{project_id}")
def update_project(project_id: int, payload: ProjectUpdate, db: Db, current: CurrentUser) -> dict[str, Any]:
    project = _get_owned_project(db, current, project_id)
    if payload.name is not None:
        project.name = payload.name
    if payload.description is not None:
        project.description = payload.description
    db.commit()
    db.refresh(project)
    return _project_out(project)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: int, db: Db, current: CurrentUser) -> None:
    project = _get_owned_project(db, current, project_id)
    db.delete(project)
    db.commit()