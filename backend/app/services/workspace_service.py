from dataclasses import dataclass, field
from typing import Dict, List, Any
import uuid


@dataclass
class Workspace:
    id: str
    name: str
    owner: str
    projects: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Project:
    id: str
    workspace_id: str
    name: str
    datasets: List[Dict[str, Any]] = field(default_factory=list)
    sessions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Dataset:
    id: str
    project_id: str
    workspace_id: str
    name: str
    rows: int
    columns: int
    schema: List[str]
    stats: Dict[str, Any] = field(default_factory=dict)
    charts: List[Dict[str, Any]] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    reports: List[Dict[str, Any]] = field(default_factory=list)
    embeddings: List[Dict[str, Any]] = field(default_factory=list)
    chat_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = ""


class WorkspaceService:
    def __init__(self):
        self.workspaces: Dict[str, Workspace] = {}
        self.projects: Dict[str, Project] = {}
        self.datasets: Dict[str, Dataset] = {}

    def create_workspace(self, name: str, owner: str) -> Dict[str, Any]:
        workspace = Workspace(id=str(uuid.uuid4())[:8], name=name, owner=owner)
        self.workspaces[workspace.id] = workspace
        return self._workspace_to_dict(workspace)

    def has_workspace(self, workspace_id: str) -> bool:
        return workspace_id in self.workspaces

    def get_project(self, project_id: str) -> Dict[str, Any] | None:
        project = self.projects.get(project_id)
        if not project:
            return None
        return self._project_to_dict(project)

    def create_project(self, workspace_id: str, name: str) -> Dict[str, Any]:
        workspace = self.workspaces.get(workspace_id)
        if not workspace:
            raise ValueError("Workspace not found")

        project = Project(id=str(uuid.uuid4())[:8], workspace_id=workspace_id, name=name)
        self.projects[project.id] = project
        workspace.projects.append(self._project_to_dict(project))
        return self._project_to_dict(project)

    def create_dataset(self, project_id: str, workspace_id: str, name: str, rows: int, columns: int, schema: List[str], stats: Dict[str, Any]) -> Dict[str, Any]:
        project = self.projects.get(project_id)
        if not project:
            raise ValueError("Project not found")
        if workspace_id not in self.workspaces:
            raise ValueError("Workspace not found")
        if project.workspace_id != workspace_id:
            raise ValueError("Project does not belong to workspace")

        dataset = Dataset(
            id=str(uuid.uuid4())[:8],
            project_id=project_id,
            workspace_id=workspace_id,
            name=name,
            rows=rows,
            columns=columns,
            schema=schema,
            stats=stats,
            created_at="now",
        )
        self.datasets[dataset.id] = dataset
        project.datasets.append(self._dataset_to_dict(dataset))
        return self._dataset_to_dict(dataset)

    def _workspace_to_dict(self, workspace: Workspace) -> Dict[str, Any]:
        return {"id": workspace.id, "name": workspace.name, "owner": workspace.owner, "projects": workspace.projects}

    def _project_to_dict(self, project: Project) -> Dict[str, Any]:
        return {"id": project.id, "workspace_id": project.workspace_id, "name": project.name, "datasets": project.datasets, "sessions": project.sessions}

    def _dataset_to_dict(self, dataset: Dataset) -> Dict[str, Any]:
        return {
            "id": dataset.id,
            "project_id": dataset.project_id,
            "workspace_id": dataset.workspace_id,
            "name": dataset.name,
            "rows": dataset.rows,
            "columns": dataset.columns,
            "schema": dataset.schema,
            "stats": dataset.stats,
            "charts": dataset.charts,
            "insights": dataset.insights,
            "reports": dataset.reports,
            "embeddings": dataset.embeddings,
            "chat_history": dataset.chat_history,
            "created_at": dataset.created_at,
        }


_workspace_service_singleton = WorkspaceService()


def get_workspace_service() -> WorkspaceService:
    return _workspace_service_singleton
