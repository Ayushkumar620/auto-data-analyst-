from __future__ import annotations
import pandas as pd

class TimeSeriesPreprocessor:
    def prepare(self, dataframe: pd.DataFrame, date_column: str, target: str) -> tuple[pd.DataFrame, str, str]:
        series_date = dataframe[date_column]
        if pd.api.types.is_numeric_dtype(series_date) and series_date.dropna().between(1800, 2150).all():
            converted_date = pd.to_datetime(series_date.astype(str) + "-01-01", errors="coerce")
        else:
            converted_date = pd.to_datetime(series_date, errors="coerce")
        data = pd.DataFrame({date_column: converted_date, target: pd.to_numeric(dataframe[target], errors="coerce")}).dropna()
        data = data.groupby(date_column, as_index=False)[target].sum().sort_values(date_column)
        inferred = pd.infer_freq(data[date_column]) if len(data) >= 3 else None
        delta = data[date_column].diff().dropna().median()
        if inferred and inferred.upper().startswith(("M", "ME", "MS")): label, offset = "monthly", "MS"
        elif delta is not pd.NaT and delta >= pd.Timedelta(days=28): label, offset = "monthly", "MS"
        elif delta is not pd.NaT and delta >= pd.Timedelta(days=6): label, offset = "weekly", "W"
        else: label, offset = "daily", "D"
        indexed = data.set_index(date_column).resample(offset)[target].sum().reset_index()
        return indexed, label, offset
