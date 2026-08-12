from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from .anomalies import AnomalyDetector
from .categorical import CategoricalAnalyzer
from .correlation import CorrelationAnalyzer
from .distribution import DistributionAnalyzer
from .statistics import StatisticsAnalyzer
from .summary import SummaryAnalyzer
from .time_series import TimeSeriesAnalyzer
from backend.app.visualization.chart_selector import ChartSelector


class EDAOrchestrator:
    def __init__(self) -> None:
        self.statistics_analyzer = StatisticsAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.distribution_analyzer = DistributionAnalyzer()
        self.categorical_analyzer = CategoricalAnalyzer()
        self.time_series_analyzer = TimeSeriesAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.summary_analyzer = SummaryAnalyzer()
        self.chart_selector = ChartSelector()

    def analyze(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        return {
            "summary": self.summary_analyzer.analyze(dataframe),
            "statistics": self.statistics_analyzer.analyze(dataframe),
            "correlations": self.correlation_analyzer.analyze(dataframe),
            "distributions": self.distribution_analyzer.analyze(dataframe),
            "categorical": self.categorical_analyzer.analyze(dataframe),
            "time_series": self.time_series_analyzer.analyze(dataframe),
            "anomalies": self.anomaly_detector.analyze(dataframe),
            "recommended_charts": self._recommended_charts(dataframe),
        }

    def _recommended_charts(self, dataframe: pd.DataFrame) -> list[Dict[str, Any]]:
        return self.chart_selector.recommend(dataframe)
