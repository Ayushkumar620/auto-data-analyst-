"""Explicit registry of operations available to the chat agent."""
from __future__ import annotations
from typing import Any, Callable
from .executor import DataExecutor

class ToolRegistry:
    def __init__(self, executor: DataExecutor | None = None) -> None:
        self.executor = executor or DataExecutor()
        self._tools: dict[str, Callable[..., Any]] = {name: getattr(self.executor, name) for name in ("get_dataset_schema", "get_column_statistics", "filter_data", "aggregate_data", "group_by", "calculate_growth", "detect_anomalies", "calculate_correlation", "create_bar_chart", "create_line_chart", "create_scatter_chart", "create_histogram", "run_eda", "generate_insights", "forecast")}
    def execute(self, name: str, dataframe: Any, **parameters: Any) -> Any:
        if name not in self._tools: raise ValueError(f"Tool '{name}' is not registered.")
        return self._tools[name](dataframe, **parameters)
    @property
    def names(self) -> tuple[str, ...]: return tuple(self._tools)
