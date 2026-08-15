"""Generate downloadable reports from cached analysis-session results."""
from __future__ import annotations
from io import BytesIO
from typing import Any
from flask import Blueprint, jsonify, request, send_file
from backend.app.api.insights import _analyses
from backend.app.reports import ReportEngine

reports_bp = Blueprint("reports_bp", __name__, url_prefix="/api/reports")
_reports: dict[str, dict[str, Any]] = {}

@reports_bp.route("/generate", methods=["POST"])
def generate_report() -> Any:
    payload = request.get_json(silent=True) or {}; dataset_id = payload.get("dataset_id"); output_format = str(payload.get("format", "pdf")).casefold()
    if not dataset_id: return jsonify({"status": "error", "message": "'dataset_id' is required."}), 400
    analysis = _analyses.get(dataset_id)
    if analysis is None: return jsonify({"status": "not_found", "message": f"Dataset '{dataset_id}' was not found in this server session."}), 404
    try: report, content, content_type = ReportEngine().generate(dataset_id, analysis, output_format)
    except (ValueError, RuntimeError) as exc: return jsonify({"status": "error", "message": str(exc)}), 422
    _reports[report.report_id] = {"content": content, "content_type": content_type, "format": output_format, "title": report.title}
    return jsonify({"status": "completed", "report_id": report.report_id, "download_url": f"/api/reports/{report.report_id}", "report": report.to_dict()})

@reports_bp.route("/<report_id>", methods=["GET"])
def download_report(report_id: str) -> Any:
    stored = _reports.get(report_id)
    if stored is None: return jsonify({"status": "not_found", "message": "Report not found."}), 404
    extension = {"pdf": "pdf", "excel": "xlsx", "powerpoint": "pptx"}[stored["format"]]
    return send_file(BytesIO(stored["content"]), mimetype=stored["content_type"], as_attachment=True, download_name=f"{report_id}.{extension}")
