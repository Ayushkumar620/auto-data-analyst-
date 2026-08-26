from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from backend.app.visualization.serializers import figure_to_json
from .confidence import ConfidenceIntervals
from .detector import TimeSeriesDetector
from .evaluator import ModelEvaluator
from .models import CandidateModels
from .preprocessing import TimeSeriesPreprocessor
from .schemas import ForecastResult
from .validator import ForecastValidator

class Forecaster:
    def __init__(self) -> None:
        self.detector, self.preprocessor, self.validator = TimeSeriesDetector(), TimeSeriesPreprocessor(), ForecastValidator()
        self.models, self.evaluator, self.confidence = CandidateModels(), ModelEvaluator(), ConfidenceIntervals()

    def forecast(self, dataframe: pd.DataFrame, horizon: int = 3, target: str | None = None, date_column: str | None = None) -> ForecastResult:
        horizon = max(1, min(int(horizon), 24)); detected = self.detector.detect(dataframe, date_column, target)
        if not detected["date_column"]: raise ValueError("I can't reliably forecast this dataset because no usable date or timestamp column was found.")
        if not detected["target"]: raise ValueError("I can't reliably forecast this dataset because no numeric target column was found.")
        date, metric = str(detected["date_column"]), str(detected["target"])
        prepared, frequency, offset = self.preprocessor.prepare(dataframe, date, metric)
        problem = self.validator.validate(prepared, date, metric)
        if problem: raise ValueError(problem)
        values = prepared[metric].to_numpy(dtype=float); split = max(1, int(len(values) * .2)); train, test = values[:-split], values[-split:]
        selected, all_metrics = self.evaluator.evaluate(test, self.models.candidates(train, len(test)))
        predicted_test = self.models.candidates(train, len(test))[selected]; future = self.models.candidates(values, horizon)[selected]
        intervals = self.confidence.build(future, test - predicted_test)
        future_dates = pd.date_range(prepared[date].iloc[-1] + pd.tseries.frequencies.to_offset(offset), periods=horizon, freq=offset)
        points = [{"date": timestamp.strftime("%Y-%m-%d"), "prediction": round(float(value), 6), "lower": round(lower, 6), "upper": round(upper, 6)} for timestamp, value, (lower, upper) in zip(future_dates, future, intervals)]
        result = ForecastResult(metric, date, frequency, horizon, selected, all_metrics[selected], points, {"start": prepared[date].iloc[0].strftime("%Y-%m-%d"), "end": prepared[date].iloc[-1].strftime("%Y-%m-%d")}, ["Forecasts are estimates, not guarantees.", "Unexpected business or market changes are not represented in the historical data."])
        result.visualization = self._chart(prepared, date, metric, points)
        return result

    @staticmethod
    def _chart(history: pd.DataFrame, date: str, target: str, points: list[dict[str, Any]]) -> dict[str, Any]:
        future_dates = [p["date"] for p in points]; forecast = [p["prediction"] for p in points]
        figure = go.Figure([go.Scatter(x=history[date], y=history[target], name="Historical", mode="lines"), go.Scatter(x=future_dates, y=forecast, name="Forecast", mode="lines+markers"), go.Scatter(x=future_dates + future_dates[::-1], y=[p["upper"] for p in points] + [p["lower"] for p in points][::-1], fill="toself", fillcolor="rgba(31, 119, 180, .16)", line={"color": "rgba(0,0,0,0)"}, name="Prediction interval")])
        figure.update_layout(title=f"{target} forecast", template="plotly_white")
        return figure_to_json(figure)
