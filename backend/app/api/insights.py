"""Insight endpoints backed by the in-process uploaded-dataset registry."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from flask import Blueprint, jsonify, request

from backend.app.insights import InsightEngine

insights_bp = Blueprint("insights_bp", __name__, url_prefix="/api/insights")
_analyses: Dict[str, Dict[str, Any]] = {}


def register_analysis(dataset_id: str, dataframe: pd.DataFrame, eda: Dict[str, Any], **context: Any) -> None:
    """Store source evidence for endpoints during the current server session."""
    _analyses[dataset_id] = {"dataframe": dataframe.copy(), "eda": eda, "insights": [], **context}


@insights_bp.route("/generate", methods=["POST"])
def generate_insights():
    payload = request.get_json(silent=True) or {}
    dataset_id = payload.get("dataset_id")
    if not dataset_id:
        return jsonify({"status": "error", "message": "'dataset_id' is required."}), 400
    analysis = _analyses.get(dataset_id)
    if analysis is None:
        return jsonify({"status": "error", "message": f"Dataset '{dataset_id}' was not found in this server session."}), 404
    result = InsightEngine().generate(analysis["dataframe"], analysis["eda"])
    analysis["insights"] = result["insights"]
    return jsonify({"status": "success", "insights": result["insights"]})
