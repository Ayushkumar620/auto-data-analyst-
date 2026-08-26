from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from backend.app.visualization.serializers import figure_to_json
from agent.autonomous_forecast_engine import AutonomousForecastEngine
from agent.forecasting_schemas import ForecastRequest as CanonicalForecastRequest
from agent.timeseries_detector import TimeSeriesDetector
from .schemas import ForecastResult


class Forecaster:
    def __init__(self) -> None:
        self.detector = TimeSeriesDetector()
        self.engine = AutonomousForecastEngine()

    def forecast(
        self,
        dataframe: pd.DataFrame,
        horizon: int = 3,
        target: str | None = None,
        date_column: str | None = None,
    ) -> ForecastResult:
        horizon = max(1, min(int(horizon), 24))

        # 1. Detect time and target column using canonical detector
        time_col = date_column if (date_column and date_column in dataframe.columns) else self.detector.detect_time_column(dataframe)
        if not time_col:
            raise ValueError("I can't reliably forecast this dataset because no usable date or timestamp column was found.")

        target_col = target if (target and target in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[target])) else (
            self.detector.detect_target_column(dataframe, time_col=time_col)
        )
        if not target_col:
            raise ValueError("I can't reliably forecast this dataset because no numeric target column was found.")

        # 2. Run canonical autonomous forecasting engine
        fc_req = CanonicalForecastRequest(
            dataset=dataframe,
            target_column=target_col,
            time_column=time_col,
            forecast_horizon=horizon,
        )
        res = self.engine.run_forecast(fc_req)

        if res.status != "SUCCESS":
            reason = res.warnings[0] if res.warnings else "Forecasting validation failed for this dataset."
            raise ValueError(reason)

        # 3. Format points and historical period
        points = [
            {
                "date": p.timestamp,
                "prediction": round(float(p.prediction), 6),
                "lower": round(float(p.lower_bound), 6),
                "upper": round(float(p.upper_bound), 6),
            }
            for p in res.predictions
        ]

        hist_clean = dataframe[[time_col, target_col]].dropna().sort_values(time_col)
        start_str = str(hist_clean[time_col].iloc[0])[:10]
        end_str = str(hist_clean[time_col].iloc[-1])[:10]

        result = ForecastResult(
            target_col,
            time_col,
            res.frequency,
            horizon,
            res.model_name,
            res.validation_metrics,
            points,
            {"start": start_str, "end": end_str},
            res.limitations or ["Forecasts are estimates, not guarantees."],
        )
        result.visualization = self._chart(hist_clean, time_col, target_col, points)
        return result

    @staticmethod
    def _chart(history: pd.DataFrame, date: str, target: str, points: list[dict[str, Any]]) -> dict[str, Any]:
        future_dates = [p["date"] for p in points]
        forecast = [p["prediction"] for p in points]
        figure = go.Figure([
            go.Scatter(x=history[date], y=history[target], name="Historical", mode="lines"),
            go.Scatter(x=future_dates, y=forecast, name="Forecast", mode="lines+markers"),
            go.Scatter(
                x=future_dates + future_dates[::-1],
                y=[p["upper"] for p in points] + [p["lower"] for p in points][::-1],
                fill="toself",
                fillcolor="rgba(31, 119, 180, .16)",
                line={"color": "rgba(0,0,0,0)"},
                name="Prediction interval",
            ),
        ])
        figure.update_layout(title=f"{target} forecast", template="plotly_white")
        return figure_to_json(figure)
