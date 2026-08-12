"""Forecast endpoint backed by datasets uploaded during this server session."""
from __future__ import annotations
from typing import Any
from flask import Blueprint, jsonify, request
from backend.app.api.insights import _analyses
from backend.app.forecasting import Forecaster
from backend.app.forecasting.schemas import ForecastRequest

forecasting_bp = Blueprint("forecasting_bp", __name__, url_prefix="/api/forecast")

@forecasting_bp.route("", methods=["POST"])
def create_forecast() -> Any:
    try:
        forecast_request = ForecastRequest.from_payload(request.get_json(silent=True) or {})
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "'horizon' must be a whole number."}), 400
    if not forecast_request.dataset_id:
        return jsonify({"status": "error", "message": "'dataset_id' is required."}), 400
    analysis = _analyses.get(forecast_request.dataset_id)
    if analysis is None:
        return jsonify({"status": "not_found", "message": f"Dataset '{forecast_request.dataset_id}' was not found in this server session."}), 404
    try:
        result = Forecaster().forecast(analysis["dataframe"], forecast_request.horizon, forecast_request.target, forecast_request.date_column)
    except ValueError as exc:
        return jsonify({"status": "unsupported", "message": str(exc)}), 422
    return jsonify({"status": "success", **result.to_dict()})
