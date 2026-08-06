from flask import Blueprint, jsonify, request

from backend.app.services.workspace_service import WorkspaceService

workspace_bp = Blueprint("workspace_bp", __name__, url_prefix="/api")
workspace_service = WorkspaceService()


@workspace_bp.route("/workspaces", methods=["POST"])
def create_workspace():
    payload = request.get_json(silent=True) or {}
    name = payload.get("name", "Untitled Workspace")
    owner = payload.get("owner", "demo")
    return jsonify(workspace_service.create_workspace(name=name, owner=owner))


@workspace_bp.route("/workspaces/<workspace_id>/projects", methods=["POST"])
def create_project(workspace_id: str):
    payload = request.get_json(silent=True) or {}
    name = payload.get("name", "Untitled Project")
    try:
        return jsonify(workspace_service.create_project(workspace_id=workspace_id, name=name))
    except ValueError as exc:
        return jsonify({"type": "error", "message": str(exc)}), 404
