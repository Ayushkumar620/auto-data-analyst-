from __future__ import annotations

from typing import Any

from backend.app.services.workspace_service import get_workspace_service


def resolve_context(workspace_id: str | None, project_id: str | None) -> tuple[str | None, str | None]:
    workspace_service = get_workspace_service()
    resolved_workspace_id = workspace_id

    if project_id:
        project = workspace_service.get_project(project_id)
        if project is None:
            raise ValueError("Project not found")
        if resolved_workspace_id and project["workspace_id"] != resolved_workspace_id:
            raise ValueError("Project does not belong to workspace")
        resolved_workspace_id = project["workspace_id"]

    if resolved_workspace_id and not workspace_service.has_workspace(resolved_workspace_id):
        raise ValueError("Workspace not found")

    return resolved_workspace_id, project_id


def link_uploaded_dataset(uploaded: dict[str, Any], workspace_id: str | None, project_id: str | None) -> dict[str, Any] | None:
    if not workspace_id or not project_id:
        return None

    workspace_service = get_workspace_service()
    return workspace_service.create_dataset(
        project_id=project_id,
        workspace_id=workspace_id,
        name=uploaded["dataset"]["name"],
        rows=uploaded["dataset"]["rows"],
        columns=uploaded["dataset"]["columns"],
        schema=uploaded["metadata"]["column_names"],
        stats={
            "missing_values": uploaded["metadata"]["missing_values"],
            "duplicate_rows": uploaded["metadata"]["duplicate_rows"],
            "memory_usage": uploaded["metadata"]["memory_usage"],
        },
    )
