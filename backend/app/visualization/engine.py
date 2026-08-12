"""Public coordination API for recommendation execution and chart responses."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import pandas as pd

from .chart_selector import ChartSelector
from .charts import ChartFactory
from .serializers import figure_to_json


class VisualizationEngine:
    def __init__(self) -> None:
        self.selector = ChartSelector()
        self.factory = ChartFactory()

    def recommend(self, dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
        return self.selector.recommend(dataframe)

    def generate(self, dataframe: pd.DataFrame, recommendation: Dict[str, Any],
                 chart_id: str = "chart_001") -> Dict[str, Any]:
        chart_type = recommendation.get("chart_type") or recommendation.get("type")
        if not chart_type:
            raise ValueError("A chart recommendation must include 'chart_type'.")
        x, y = recommendation.get("x"), recommendation.get("y")
        title = recommendation.get("title") or str(chart_type).title()
        figure = self.factory.create(dataframe, chart_type, x=x, y=y, title=title,
                                     columns=recommendation.get("columns"))
        return {"id": chart_id, "type": chart_type, "title": title,
                "x_column": x, "y_column": y, "figure": figure_to_json(figure)}

    def generate_dashboard(self, dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
        return [self.generate(dataframe, recommendation, f"chart_{index:03d}")
                for index, recommendation in enumerate(self.recommend(dataframe), start=1)]
