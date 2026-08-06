"""
Data Cleaner - Automatically detects and fixes data quality issues.
Handles missing values, duplicate rows, type coercion, and outlier detection.
"""
import pandas as pd
import numpy as np


class DataCleaner:
    """Automatically cleans a DataFrame and reports what was changed."""

    def __init__(self, data):
        # data may be a single DataFrame or a dict of DataFrames
        self.data = data

    def _get_frames(self):
        if isinstance(self.data, dict):
            return list(self.data.items())
        return [("data", self.data)]

    def clean(self):
        """Clean all frames and return a report of changes made."""
        reports = []
        for name, df in self._get_frames():
            if not isinstance(df, pd.DataFrame):
                continue
            report = self._clean_frame(name, df.copy())
            reports.append(report)
        return reports

    def _clean_frame(self, name, df):
        original_rows = int(df.shape[0])
        original_cols = int(df.shape[1])
        changes = []
        actions = []

        # 1. Trim whitespace in string columns
        str_cols = df.select_dtypes(include=["object"]).columns
        trimmed = 0
        for col in str_cols:
            cleaned = df[col].astype(str).str.strip()
            if df[col].notna().any():
                # count cells where trimming actually changed something
                mask = df[col].notna() & (df[col].astype(str) != cleaned)
                trimmed += int(mask.sum())
        if trimmed > 0:
            for col in str_cols:
                df[col] = df[col].astype(str).str.strip().replace("nan", np.nan)
            changes.append(trimmed)
            actions.append(f"Trimmed whitespace in {trimmed} string cells")

        # 2. Coerce obvious numeric columns (strings that look like numbers)
        coerced_cols = []
        for col in df.columns:
            if df[col].dtype == object:
                sample = df[col].dropna().astype(str).head(20)
                if len(sample) > 0:
                    # Remove currency symbols and commas for checking
                    cleaned_sample = sample.str.replace(r"[₹$€,]", "", regex=True).str.strip()
                    numeric_count = cleaned_sample.str.match(r"^-?\d+(\.\d+)?$").sum()
                    if numeric_count >= len(sample) * 0.8 and numeric_count > 0:
                        df[col] = pd.to_numeric(
                            cleaned_sample, errors="coerce"
                        )
                        coerced_cols.append(col)
        if coerced_cols:
            actions.append(f"Converted {len(coerced_cols)} column(s) to numeric: {', '.join(coerced_cols)}")

        # 3. Parse date columns (heuristic: column name hints or string dates)
        date_cols = []
        for col in df.columns:
            if col.lower() in ("date", "datetime", "time", "transaction_date", "timestamp", "month", "year"):
                try:
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    if parsed.notna().sum() > 0:
                        df[col] = parsed
                        date_cols.append(col)
                except Exception:
                    pass
        if date_cols:
            actions.append(f"Parsed {len(date_cols)} column(s) as date/time: {', '.join(date_cols)}")

        # 4. Drop duplicate rows
        before_dup = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        dup_removed = before_dup - len(df)
        if dup_removed > 0:
            actions.append(f"Removed {dup_removed} duplicate row(s)")

        # 5. Handle missing values
        null_before = int(df.isnull().sum().sum())
        null_filled = 0
        filled_cols = []
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in (np.number, float, int):
                    fill_val = df[col].median()
                    if pd.isna(fill_val):
                        fill_val = df[col].mean()
                    if pd.isna(fill_val):
                        fill_val = 0
                    df[col] = df[col].fillna(fill_val)
                    null_filled += int(df[col].isnull().sum())
                    filled_cols.append(col)
                else:
                    fill_val = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
                    null_filled += int(df[col].isnull().sum())
                    df[col] = df[col].fillna(fill_val)
                    filled_cols.append(col)
        if filled_cols:
            actions.append(f"Filled missing values in {len(filled_cols)} column(s): {', '.join(filled_cols)}")

        # 6. Outlier detection (z-score) on numeric columns
        outlier_cols = []
        outlier_count = 0
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] >= 1:
            for col in numeric.columns:
                vals = df[col].dropna()
                if len(vals) < 4:
                    continue
                mean = vals.mean()
                std = vals.std()
                if std == 0 or pd.isna(std):
                    continue
                z = np.abs((vals - mean) / std)
                n_outliers = int((z > 3).sum())
                if n_outliers > 0:
                    outlier_cols.append(col)
                    outlier_count += n_outliers
        if outlier_count > 0:
            actions.append(f"Detected {outlier_count} potential outlier(s) in: {', '.join(outlier_cols)} (flagged, not removed)")

        return {
            "name": name,
            "original_shape": {"rows": original_rows, "columns": original_cols},
            "cleaned_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "changes": changes,
            "actions": actions,
            "cleaned_data": df.to_dict(orient="records"),
            "columns": list(df.columns),
            "dtypes": {str(c): str(dt) for c, dt in df.dtypes.items()},
        }
