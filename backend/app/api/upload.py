from flask import Blueprint, jsonify, request

from backend.app.services.file_service import FileService

upload_bp = Blueprint("upload_bp", __name__, url_prefix="/api")


@upload_bp.route("/upload", methods=["POST"])
def upload_dataset():
    file_storage = request.files.get("file")
    service = FileService(upload_folder="uploads")

    try:
        result = service.upload_and_profile(file_storage)
    except ValueError as exc:
        return jsonify({"type": "error", "message": str(exc)}), 400
    except Exception as exc:
        return jsonify({"type": "error", "message": f"Upload failed: {exc}"}), 500

    return jsonify(result)
