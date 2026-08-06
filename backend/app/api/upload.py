from flask import Blueprint, jsonify, request

from backend.app.services.dataset_service import DatasetService
from backend.app.services.workspace_service import WorkspaceService

upload_bp = Blueprint("upload_bp", __name__, url_prefix="/api")


@upload_bp.route("/upload", methods=["POST"])
def upload_dataset():
    file_storage = request.files.get("file")
    dataset_service = DatasetService(upload_folder="uploads")
    workspace_service = WorkspaceService()

    try:
        result = dataset_service.upload_dataset(file_storage)
        workspace = workspace_service.create_workspace(name="Default Workspace", owner="demo")
        project = workspace_service.create_project(workspace_id=workspace["id"], name="Imported Project")
        dataset = workspace_service.create_dataset(
            project_id=project["id"],
            workspace_id=workspace["id"],
            name=result["dataset"]["name"],
            rows=result["dataset"]["rows"],
            columns=result["dataset"]["columns"],
            schema=result["metadata"]["column_names"],
            stats={
                "missing_values": result["metadata"]["missing_values"],
                "duplicate_rows": result["metadata"]["duplicate_rows"],
                "memory_usage": result["metadata"]["memory_usage"],
            },
        )
        result["dataset_id"] = dataset["id"]
        result["project_id"] = project["id"]
        result["workspace_id"] = workspace["id"]
        result["dataset_name"] = result["dataset"]["name"]
        result["rows"] = result["dataset"]["rows"]
        result["columns"] = result["dataset"]["columns"]
        result["column_names"] = result["metadata"]["column_names"]
        result["missing_values"] = result["metadata"]["missing_values"]
        result["duplicates"] = result["metadata"]["duplicate_rows"]
        result["data_types"] = result["metadata"]["data_types"]
        result["memory_usage"] = result["metadata"]["memory_usage"]
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Upload failed: {exc}"}), 500

    return jsonify(result)
