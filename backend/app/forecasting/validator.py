from __future__ import annotations
import pandas as pd

class ForecastValidator:
    MIN_OBSERVATIONS = 8
    def validate(self, series: pd.DataFrame, date_column: str, target: str) -> str | None:
        if date_column not in series or target not in series: return "The selected date or target column does not exist."
        if not pd.api.types.is_numeric_dtype(series[target]): return f"'{target}' must be numeric to forecast it."
        if len(series) < self.MIN_OBSERVATIONS: return f"I can't reliably forecast this dataset because there are only {len(series)} historical observations; at least {self.MIN_OBSERVATIONS} are required."
        if series[date_column].isna().any() or series[target].isna().all(): return "I can't reliably forecast this dataset because dates or target values are incomplete."
        if not series[date_column].is_monotonic_increasing: return "The time series could not be ordered reliably."
        return None
