"""
Canonical Data Layer & Universal Semantic Profiler.

Establishes a single, authoritative, dataset-agnostic representation for all
analytical agents, models, and pipelines.

Preserves:
- original dataframe
- original row count
- original dtypes
- semantic profile (numeric, categorical, datetime, identifier, text, constant)
- missing and duplicate statistics
- statistical target and feature candidates
- task suitability (regression, classification, forecasting, clustering, descriptive)
- granular row accounting and non-destructive transformations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
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


@dataclass
class SemanticProfile:
    """Deep statistical profiling of dataset characteristics without hardcoded assumptions."""
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    datetime_candidates: List[str] = field(default_factory=list)
    boolean_columns: List[str] = field(default_factory=list)
    identifier_columns: List[str] = field(default_factory=list)
    text_columns: List[str] = field(default_factory=list)
    constant_columns: List[str] = field(default_factory=list)
    high_cardinality_columns: List[str] = field(default_factory=list)
    missing_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    duplicate_stats: Dict[str, Any] = field(default_factory=dict)
    target_candidates: List[Dict[str, Any]] = field(default_factory=list)
    feature_candidates: List[str] = field(default_factory=list)
    suggested_task: str = "descriptive"
    is_time_series_ready: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CanonicalDataset:
    """Authoritative, non-mutated dataset object passed to analytical agents."""
    original_df: pd.DataFrame
    original_rows: int
    columns: List[str]
    original_dtypes: Dict[str, str]
    profile: SemanticProfile
    transformation_log: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_rows": self.original_rows,
            "columns": self.columns,
            "original_dtypes": self.original_dtypes,
            "profile": self.profile.to_dict(),
            "transformation_log": self.transformation_log,
        }


class CanonicalDataLayer:
    """Centralized, dataset-agnostic data validation, profiling, and preparation layer."""

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

        if pd.api.types.is_integer_dtype(series):
            non_null = series.dropna()
            if not non_null.empty and non_null.between(1800, 2150).all():
                return pd.to_datetime(series.astype(str) + "-01-01", errors="coerce")

        try:
            return pd.to_datetime(series, errors="coerce")
        except Exception:
            return pd.Series(pd.NaT, index=series.index)

    @classmethod
    def profile_dataset(cls, df: pd.DataFrame) -> SemanticProfile:
        """
        Perform comprehensive statistical semantic profiling on arbitrary tabular datasets.
        Identifies numeric, categorical, boolean, datetime, identifier, constant, and target candidates.
        """
        n_rows = len(df)
        if n_rows == 0:
            return SemanticProfile()

        numeric_cols: List[str] = []
        categorical_cols: List[str] = []
        datetime_candidates: List[str] = []
        boolean_cols: List[str] = []
        identifier_cols: List[str] = []
        text_cols: List[str] = []
        constant_cols: List[str] = []
        high_cardinality_cols: List[str] = []
        missing_stats: Dict[str, Dict[str, Any]] = {}

        for col in df.columns:
            s = df[col]
            null_count = int(s.isna().sum())
            null_pct = round((null_count / n_rows) * 100.0, 2)
            missing_stats[str(col)] = {"null_count": null_count, "null_pct": null_pct}

            non_null = s.dropna()
            n_valid = len(non_null)
            if n_valid == 0:
                constant_cols.append(str(col))
                continue

            unique_count = int(non_null.nunique())

            # 1. Constant check
            if unique_count <= 1:
                constant_cols.append(str(col))
                continue

            # 2. Boolean check
            if pd.api.types.is_bool_dtype(s) or (unique_count == 2 and set(non_null.unique()).issubset({0, 1, "0", "1", "true", "false", "True", "False", True, False})):
                boolean_cols.append(str(col))
                continue

            # 3. Datetime check
            if pd.api.types.is_datetime64_any_dtype(s):
                datetime_candidates.append(str(col))
                continue

            # Check if object/string column parses as datetime
            if s.dtype == object or str(s.dtype).startswith("str"):
                sample_valid = non_null.head(15)
                try:
                    parsed_sample = pd.to_datetime(sample_valid, errors="coerce")
                    if parsed_sample.notna().mean() >= 0.80 and parsed_sample.nunique() > 1:
                        datetime_candidates.append(str(col))
                        continue
                except Exception:
                    pass

            # 4. Numeric check (direct or coerced)
            coerced_num = cls.coerce_numeric_series(s)
            num_valid_ratio = coerced_num.notna().mean()

            if num_valid_ratio >= 0.70:
                # Check if it is an integer sequential identifier
                if pd.api.types.is_integer_dtype(s) and unique_count / n_valid > 0.95 and coerced_num.min() >= 0:
                    diffs = coerced_num.diff().dropna()
                    if not diffs.empty and (diffs == 1).mean() > 0.8:
                        identifier_cols.append(str(col))
                        continue

                numeric_cols.append(str(col))
                continue

            # 5. Categorical / Text check
            if unique_count <= 50 or (unique_count / n_valid) <= 0.20:
                categorical_cols.append(str(col))
            elif unique_count / n_valid > 0.80 and any(isinstance(x, str) and len(str(x)) > 40 for x in non_null.head(10)):
                text_cols.append(str(col))
            else:
                high_cardinality_cols.append(str(col))

        # Duplicate statistics
        dup_count = int(df.duplicated().sum())
        duplicate_stats = {
            "duplicate_rows": dup_count,
            "duplicate_pct": round((dup_count / n_rows) * 100.0, 2),
        }

        # Target Candidates Ranking
        target_candidates: List[Dict[str, Any]] = []
        for col in numeric_cols:
            if col in identifier_cols or col in constant_cols:
                continue
            s_num = cls.coerce_numeric_series(df[col]).dropna()
            if len(s_num) < 3:
                continue
            variance = float(s_num.var()) if not math.isnan(s_num.var()) else 0.0
            if variance <= 0:
                continue
            score = 60.0 + (len(s_num) / n_rows) * 20.0 + min(20.0, s_num.nunique() / 2.0)
            target_candidates.append({
                "column": col,
                "score": round(score, 2),
                "type": "regression",
                "variance": round(variance, 4),
                "unique_values": int(s_num.nunique()),
            })

        for col in categorical_cols:
            if col in identifier_cols or col in constant_cols:
                continue
            s_cat = df[col].dropna()
            u_count = s_cat.nunique()
            if 2 <= u_count <= 20:
                score = 50.0 + (len(s_cat) / n_rows) * 20.0
                target_candidates.append({
                    "column": col,
                    "score": round(score, 2),
                    "type": "classification",
                    "classes": u_count,
                })

        target_candidates.sort(key=lambda x: x["score"], reverse=True)

        # Feature Candidates
        feature_candidates = [
            c for c in df.columns
            if c not in identifier_cols and c not in constant_cols and c not in high_cardinality_cols
        ]

        # Suggested Task Type
        is_ts = len(datetime_candidates) > 0 and len(numeric_cols) > 0 and n_rows >= 5
        if is_ts:
            suggested_task = "time_series_forecast"
        elif target_candidates and target_candidates[0]["type"] == "classification":
            suggested_task = "classification"
        elif target_candidates and target_candidates[0]["type"] == "regression":
            suggested_task = "regression"
        elif len(numeric_cols) >= 2:
            suggested_task = "clustering"
        else:
            suggested_task = "descriptive"

        return SemanticProfile(
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            datetime_candidates=datetime_candidates,
            boolean_columns=boolean_cols,
            identifier_columns=identifier_cols,
            text_columns=text_cols,
            constant_columns=constant_cols,
            high_cardinality_columns=high_cardinality_cols,
            missing_stats=missing_stats,
            duplicate_stats=duplicate_stats,
            target_candidates=target_candidates,
            feature_candidates=feature_candidates,
            suggested_task=suggested_task,
            is_time_series_ready=is_ts,
        )

    @classmethod
    def ingest(cls, data: Union[pd.DataFrame, Dict[str, Any]]) -> CanonicalDataset:
        """Standardize any input data into a CanonicalDataset object with semantic profile."""
        if isinstance(data, dict):
            for df in data.values():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    df_target = df
                    break
            else:
                df_target = pd.DataFrame()
        elif isinstance(data, pd.DataFrame):
            df_target = data
        else:
            df_target = pd.DataFrame()

        profile = cls.profile_dataset(df_target)
        dtypes_map = {str(c): str(dt) for c, dt in df_target.dtypes.items()}

        return CanonicalDataset(
            original_df=df_target,
            original_rows=len(df_target),
            columns=list(df_target.columns),
            original_dtypes=dtypes_map,
            profile=profile,
            transformation_log=[],
        )

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
                median_val = num_s.median() if not np.isnan(num_s.median()) else 0.0
                X_df[col] = num_s.fillna(median_val)
            else:
                from sklearn.preprocessing import LabelEncoder
                cat_s = series.fillna("Unknown").astype(str)
                try:
                    X_df[col] = LabelEncoder().fit_transform(cat_s)
                except Exception:
                    pass

        if X_df.empty or X_df.shape[1] == 0:
            X_df = pd.DataFrame({"__step": np.arange(len(y))}, index=y.index)

        audit.analysis_rows = len(y)
        audit.valid_rows = len(y)
        audit.rows_removed = audit.original_rows - len(y)

        return X_df, y, audit
