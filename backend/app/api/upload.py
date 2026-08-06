import os

from flask import Blueprint, jsonify, request

from backend.app.cleaning.cleaner import DataCleaner
from backend.app.profilers.dataset_profiler import DatasetProfiler
from backend.app.services.dataset_service import DatasetService
from backend.app.services.workspace_service import WorkspaceService

upload_bp = Blueprint("upload_bp", __name__, url_prefix="/api")


@upload_bp.route("/upload", methods=["POST"])
def upload_dataset():
    file_storage = request.files.get("file")
    dataset_service = DatasetService(upload_folder="uploads")
    workspace_service = WorkspaceService()
    profiler = DatasetProfiler()

    try:
        result = dataset_service.upload_dataset(file_storage)
        dataframe = dataset_service._read_dataframe(os.path.join("uploads", result["dataset"]["name"]))
        profile_result = profiler.profile(
            dataframe=dataframe,
            filename=result["dataset"]["name"],
            file_type=result["dataset"]["file_type"],
            file_size=result["metadata"]["file_size"],
        )
        cleaning_result = DataCleaner(dataframe).clean()
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
        result["profile"] = profile_result["profile"]
        result["column_analysis"] = profile_result["column_analysis"]
        result["numeric_analysis"] = profile_result["numeric_analysis"]
        result["categorical_analysis"] = profile_result["categorical_analysis"]
        result["missing_values_details"] = profile_result["missing_values"]
        result["duplicate_analysis"] = profile_result["duplicate_analysis"]
        result["recommendations"] = profile_result["recommendations"]
        result["preview"] = profile_result["preview"]
        result["cleaning"] = {
            "status": cleaning_result["status"],
            "quality_before": cleaning_result["quality_before"],
            "quality_after": cleaning_result["quality_after"],
            "rows_removed": cleaning_result["rows_removed"],
            "missing_values_fixed": cleaning_result["missing_values_fixed"],
            "datatype_conversions": cleaning_result["datatype_conversions"],
            "outliers_detected": cleaning_result["outliers_detected"],
            "cleaning_report": cleaning_result["cleaning_report"],
        }
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Upload failed: {exc}"}), 500

    return jsonify(result)
