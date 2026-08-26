"""
Canonical Data Layer & Universal Row Validator.

Provides centralized, dataset-agnostic data validation, type normalization,
and granular row accounting across prediction, forecasting, profiling, and analysis.

Separates row metrics into:
- original_rows: total raw rows in user input
- parsed_rows: rows successfully ingested into DataFrame
- valid_rows: rows with usable analysis features
- target_valid_rows: rows with valid target values
- time_series_valid_rows: rows with valid (time, target) pairs
- analysis_rows: rows retained for the final model execution
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class DatasetRowAudit:
    """Granular audit metrics tracking row retention across data lifecycle."""
    original_rows: int
    parsed_rows: int
    valid_rows: int
    target_column: Optional[str]
    time_column: Optional[str]
    target_valid_rows: int
    time_series_valid_rows: int
    analysis_rows: int
    rows_removed: int
    removal_reasons: List[str] = field(default_factory=list)
    minimum_required_rows: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CanonicalDataLayer:
    """Authoritative validation and preprocessing pipeline for analytical engines."""

    @staticmethod
    def coerce_numeric_series(series: pd.Series) -> pd.Series:
        """
        Dataset-agnostic numeric coercion.
        Handles currency symbols ($), commas (,), percentage signs (%),
        parentheses for negative numbers, unit suffixes (k, M, B), and mixed strings.
        """
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce")

        def clean_val(val: Any) -> Any:
            if pd.isna(val):
                return np.nan
            if isinstance(val, (int, float, np.number)):
                return float(val)
            s = str(val).strip()
            if not s:
                return np.nan

            # Handle parentheses notation for negative numbers: "(1,234.50)" -> "-1234.50"
            if s.startswith("(") and s.endswith(")"):
                s = "-" + s[1:-1]

            # Remove currency symbols, commas, and percentage signs using unicode escapes
            s_clean = re.sub(r"[\$,\u00a3\u20ac\u00a5%\s]", "", s)

            # Handle common financial/metric multiplier suffixes
            multiplier = 1.0
            if s_clean.endswith(("k", "K")):
                multiplier = 1e3
                s_clean = s_clean[:-1]
            elif s_clean.endswith(("m", "M")):
                multiplier = 1e6
                s_clean = s_clean[:-1]
            elif s_clean.endswith(("b", "B")):
                multiplier = 1e9
                s_clean = s_clean[:-1]

            try:
                return float(s_clean) * multiplier
            except ValueError:
                return np.nan

        return series.map(clean_val)

    @staticmethod
    def coerce_datetime_series(series: pd.Series) -> pd.Series:
        """
        Dataset-agnostic datetime coercion.
        Handles ISO timestamps, common date strings, integer year sequences (1800-2150),
        quarter strings (2024Q1), and epoch timestamps.
        """
        if pd.api.types.is_datetime64_any_dtype(series):
            return series

        # If integer series in year range 1800..2150
        if pd.api.types.is_integer_dtype(series):
            non_null = series.dropna()
            if not non_null.empty and non_null.between(1800, 2150).all():
                return pd.to_datetime(series.astype(str) + "-01-01", errors="coerce")

        # Standard pandas to_datetime with fallback
        try:
            return pd.to_datetime(series, errors="coerce")
        except Exception:
            return pd.Series(pd.NaT, index=series.index)

    @classmethod
    def audit_dataset_for_target(
        cls,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        time_column: Optional[str] = None,
        minimum_required_rows: int = 5,
    ) -> Tuple[DatasetRowAudit, pd.Series, Optional[pd.Series]]:
        """
        Audit row validity without dropping unrelated data rows.
        Returns: (DatasetRowAudit, clean_target_series, clean_time_series)
        """
        orig_count = len(df)
        removal_reasons = []

        if target_column and target_column in df.columns:
            target_raw = df[target_column]
            target_clean = cls.coerce_numeric_series(target_raw)
            target_null_count = int(target_clean.isna().sum())
            if target_null_count > 0:
                removal_reasons.append(
                    f"Target column '{target_column}' contained {target_null_count} missing or non-numeric values."
                )
            target_valid_count = int(target_clean.notna().sum())
        else:
            target_clean = pd.Series(np.nan, index=df.index)
            target_valid_count = 0

        if time_column and time_column in df.columns:
            time_raw = df[time_column]
            time_clean = cls.coerce_datetime_series(time_raw)
            time_null_count = int(time_clean.isna().sum())
            if time_null_count > 0:
                removal_reasons.append(
                    f"Time column '{time_column}' contained {time_null_count} invalid or unparseable timestamps."
                )
        else:
            time_clean = None

        if time_clean is not None and target_column:
            ts_valid_mask = time_clean.notna() & target_clean.notna()
            time_series_valid_count = int(ts_valid_mask.sum())
        else:
            time_series_valid_count = target_valid_count

        valid_rows = time_series_valid_count if time_clean is not None else target_valid_count
        analysis_rows = valid_rows
        rows_removed = orig_count - analysis_rows

        audit = DatasetRowAudit(
            original_rows=orig_count,
            parsed_rows=orig_count,
            valid_rows=valid_rows,
            target_column=target_column,
            time_column=time_column,
            target_valid_rows=target_valid_count,
            time_series_valid_rows=time_series_valid_count,
            analysis_rows=analysis_rows,
            rows_removed=rows_removed,
            removal_reasons=removal_reasons,
            minimum_required_rows=minimum_required_rows,
        )

        return audit, target_clean, time_clean

    @classmethod
    def prepare_tabular_prediction_data(
        cls,
        df: pd.DataFrame,
        target_column: str,
        minimum_required_rows: int = 10,
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], DatasetRowAudit]:
        """
        Prepare features X and target y for tabular prediction.
        Non-destructive: isolates target validation and imputes/filters features so
        unrelated missing values never discard valid observations.
        """
        audit, target_clean, _ = cls.audit_dataset_for_target(
            df,
            target_column=target_column,
            minimum_required_rows=minimum_required_rows,
        )

        # 1. Check if target is valid
        valid_target_mask = target_clean.notna()
        if valid_target_mask.sum() < minimum_required_rows:
            return None, None, audit

        df_valid = df.loc[valid_target_mask].copy()
        y = target_clean.loc[valid_target_mask].copy()

        # 2. Select Candidate Features (exclude target)
        feature_cols = [c for c in df_valid.columns if c != target_column]
        if not feature_cols:
            # If no features exist, generate time-step positional feature
            X = pd.DataFrame({"__step": np.arange(len(y))}, index=y.index)
            return X, y, audit

        # 3. Clean Feature Columns Without Dropping Valid Target Rows
        X_df = pd.DataFrame(index=y.index)
        for col in feature_cols:
            series = df_valid[col]

            # Exclude feature if > 60% missing values
            if series.isna().mean() > 0.60:
                audit.removal_reasons.append(f"Excluded sparse feature column '{col}' (>60% missing values).")
                continue

            # Exclude feature if constant (0 variance)
            if series.nunique(dropna=True) <= 1:
                continue

            # Datetime conversion
            if pd.api.types.is_datetime64_any_dtype(series):
                X_df[col] = series.astype("int64") // 10**9
                continue

            # Try numeric coercion
            num_s = cls.coerce_numeric_series(series)
            if num_s.notna().mean() >= 0.70:
                # Impute missing numeric with median
                median_val = num_s.median() if not np.isnan(num_s.median()) else 0.0
                X_df[col] = num_s.fillna(median_val)
            else:
                # Treat as categorical, fill NA with 'Unknown' and label encode
                from sklearn.preprocessing import LabelEncoder
                cat_s = series.fillna("Unknown").astype(str)
                try:
                    X_df[col] = LabelEncoder().fit_transform(cat_s)
                except Exception:
                    pass

        if X_df.empty or X_df.shape[1] == 0:
            X_df = pd.DataFrame({"__step": np.arange(len(y))}, index=y.index)

        # Final audit synchronization
        audit.analysis_rows = len(y)
        audit.valid_rows = len(y)
        audit.rows_removed = audit.original_rows - len(y)

        return X_df, y, audit
