from __future__ import annotations
import pandas as pd

class TimeSeriesDetector:
    DATE_WORDS = ("date", "time", "timestamp", "month", "year")
    def detect(self, dataframe: pd.DataFrame, date_column: str | None = None, target: str | None = None) -> dict[str, str | None]:
        if date_column and date_column not in dataframe.columns: return {"date_column": None, "target": None}
        dates = [date_column] if date_column else [c for c in dataframe.columns if any(w in c.casefold() for w in self.DATE_WORDS)]
        parsed = next((c for c in dates if c and pd.to_datetime(dataframe[c], errors="coerce").notna().mean() >= .8), None)
        numeric = list(dataframe.select_dtypes(include="number").columns)
        metric = target if target in numeric else (numeric[0] if numeric else None)
        return {"date_column": parsed, "target": metric}
