"""
Insights Engine - Generates smart insights and analysis based on parsed command intents.
Handles transaction analysis, text analysis, and tabular data aggregation.
"""
import re
import pandas as pd
import numpy as np
from datetime import datetime

from backend.app.core.evidence_insights import (
    EvidenceBasedInsightsEngine,
    InsightsCatalog,
    StructuredInsight,
)


class InsightsEngine:
    """Generates insights from data based on a parsed command intent."""

    def __init__(self, data):
        self.data = data
        self.evidence_engine = EvidenceBasedInsightsEngine(data)

    def generate_structured_insights(self, model_result=None) -> Dict[str, Any]:
        """Generate structured, evidence-attributed insights with epistemic classifications."""
        catalog = self.evidence_engine.build_catalog(data_input=self.data, model_result=model_result)
        return catalog.to_dict()

    def _get_frames(self):
        if isinstance(self.data, dict):
            return list(self.data.items())
        return [("data", self.data)]

    def _get_main_df(self):
        for name, df in self._get_frames():
            if isinstance(df, pd.DataFrame) and not df.empty:
                return name, df
        return None, None

    def text_analysis(self):
        """Analyze text content: word count, sentences, characters, frequency."""
        name, df = self._get_main_df()
        if df is None:
            return {"error": "No data available for text analysis."}

        # Gather all text from string columns
        text_parts = []
        for col in df.columns:
            if df[col].dtype == object:
                text_parts.extend([str(x) for x in df[col].dropna()])

        if not text_parts:
            return {"error": "No text columns found in the data."}

        full_text = " ".join(text_parts)
        words = re.findall(r"\b\w+\b", full_text)
        sentences = re.split(r"[.!?]+", full_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        chars = len(full_text)

        # Word frequency
        from collections import Counter
        stopwords = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
                     "is", "are", "was", "were", "be", "been", "with", "at", "by",
                     "this", "that", "it", "as", "from", "we", "you", "i", "they"}
        word_freq = Counter(w.lower() for w in words)
        common = {w: c for w, c in word_freq.most_common(15) if w not in stopwords}

        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "character_count": chars,
            "unique_words": len(word_freq),
            "common_words": common,
            "sample_text": full_text[:200],
        }

    def transaction_analysis(self, intent):
        """Analyze financial/transaction data based on command intent."""
        name, df = self._get_main_df()
        if df is None:
            return {"error": "No transaction data available."}

        # Find amount and date columns
        amount_col = None
        date_col = None
        for col in df.columns:
            if col.lower() in ("amount", "value", "total", "price", "credit", "debit"):
                amount_col = col
            if col.lower() in ("date", "datetime", "transaction_date", "time"):
                date_col = col

        if amount_col is None:
            # Try to find a numeric column that looks like money
            numeric = df.select_dtypes(include=[np.number])
            if not numeric.empty:
                amount_col = numeric.columns[0]

        if amount_col is None:
            return {"error": "No amount column found for transaction analysis."}

        result = {"data_name": name, "amount_col": amount_col}

        # Convert amount
        df = df.copy()
        df["_amount"] = pd.to_numeric(df[amount_col], errors="coerce")

        # Filter by type (paid/received)
        if intent.amount_type:
            if intent.amount_type == "received":
                filtered = df[df["_amount"] > 0]
            else:
                filtered = df[df["_amount"] < 0]
            result["type_filter"] = intent.amount_type
        else:
            # Separate into received/paid
            filtered = df
            result["received_total"] = float(df.loc[df["_amount"] > 0, "_amount"].sum())
            result["paid_total"] = float(df.loc[df["_amount"] < 0, "_amount"].sum())
            result["received_count"] = int((df["_amount"] > 0).sum())
            result["paid_count"] = int((df["_amount"] < 0).sum())

        # Apply time filter
        if date_col and intent.time_filter:
            try:
                df["_date"] = pd.to_datetime(df[date_col], errors="coerce")
                tf = intent.time_filter
                if "start" in tf and "end" in tf:
                    filtered = filtered[(df["_date"] >= tf["start"]) & (df["_date"] <= tf["end"])]
                elif tf.get("period") == "month":
                    filtered = filtered[(df["_date"].dt.month == tf["month"]) &
                                        (df["_date"].dt.year == tf["year"] if tf.get("year") else True)]
                result["time_period"] = tf.get("period", "unknown")
            except Exception:
                pass

        # Compute metrics
        if intent.metric == "total":
            result["value"] = float(filtered["_amount"].sum())
        elif intent.metric == "count":
            result["value"] = int(len(filtered))
        elif intent.metric == "average":
            result["value"] = float(filtered["_amount"].mean())
        elif intent.metric == "maximum":
            result["value"] = float(filtered["_amount"].max())
        elif intent.metric == "minimum":
            result["value"] = float(filtered["_amount"].min())
        else:
            result["value"] = float(filtered["_amount"].sum())
            result["count"] = int(len(filtered))

        # Group by category if requested
        if intent.group_by:
            cat_col = None
            for col in df.columns:
                if col.lower() in ("category", "type", "tag", "group", "description", "name"):
                    cat_col = col
                    break
            if cat_col:
                grouped = filtered.groupby(df[cat_col])["_amount"].agg(["sum", "count"]).reset_index()
                grouped.columns = [cat_col, "total", "count"]
                grouped = grouped.sort_values("total", ascending=False)
                result["group_by"] = cat_col
                result["groups"] = grouped.head(15).to_dict(orient="records")

# Clean up
        result.pop("_amount", None)
        if "_date" in result:
            result.pop("_date")
        return result

    def detect_anomalies(self, columns=None):
        """Detect statistical anomalies (outliers) using z-score and IQR methods.

        Returns the flagged unusual values per numeric column.
        """
        name, df = self._get_main_df()
        if df is None:
            return {"error": "No data available for anomaly detection."}

        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            return {"error": "No numeric columns found for anomaly detection."}

        if columns:
            cols = [c for c in numeric.columns if c in columns] or list(numeric.columns)
        else:
            cols = list(numeric.columns)

        anomalies = {}
        for col in cols:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) < 4:
                continue
            mean = vals.mean()
            std = vals.std()
            if std == 0 or pd.isna(std):
                continue

            # Z-score method
            z = np.abs((vals - mean) / std)
            z_outliers = vals[z > 3]

            # IQR method
            q1 = vals.quantile(0.25)
            q3 = vals.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            iqr_outliers = vals[(vals < lower) | (vals > upper)]

            # Combine (union)
            outlier_idx = z_outliers.index.union(iqr_outliers.index)
            if len(outlier_idx) == 0:
                continue

            anomalies[col] = {
                "count": int(len(outlier_idx)),
                "total": int(len(vals)),
                "percentage": round(len(outlier_idx) / len(vals) * 100, 2),
                "mean": round(float(mean), 4),
                "std": round(float(std), 4),
                "lower_bound": round(float(lower), 4),
                "upper_bound": round(float(upper), 4),
                "values": [round(float(v), 4) for v in vals.loc[outlier_idx].head(20)],
                "indices": [int(i) for i in outlier_idx.tolist()[:20]],
            }

        if not anomalies:
            return {
                "data_name": name,
                "message": "No significant anomalies detected in the numeric columns.",
                "columns_checked": cols,
                "anomalies": {},
            }

        return {
            "data_name": name,
            "columns_checked": cols,
            "anomalies": anomalies,
        }

    def detect_anomalies_in_transactions(self):
        """Detect unusual transactions (large/recurring patterns) in bank data."""
        name, df = self._get_main_df()
        if df is None:
            return {"error": "No data available for anomaly detection."}

        # Find amount column
        amount_col = None
        for col in df.columns:
            if col.lower() in ("amount", "value", "total", "credit", "debit"):
                amount_col = col
                break
        if amount_col is None:
            return self.detect_anomalies()

        vals = pd.to_numeric(df[amount_col], errors="coerce").dropna()
        if len(vals) < 4:
            return {"error": "Not enough transaction records for anomaly detection."}

        mean = vals.mean()
        std = vals.std()
        if std == 0 or pd.isna(std):
            return {"error": "No variance in transaction amounts."}

        z = np.abs((vals - mean) / std)
        suspicious = z > 2.5
        suspicious_idx = vals[suspicious].index

        records = []
        for idx in suspicious_idx:
            row = df.loc[idx]
            rec = {"index": int(idx)}
            for col in df.columns:
                val = row[col]
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    rec[col] = round(float(val), 4)
                else:
                    rec[col] = str(val)
            rec["z_score"] = round(float(z.loc[idx]), 4)
            records.append(rec)

        return {
            "data_name": name,
            "amount_col": amount_col,
            "total_transactions": int(len(vals)),
            "suspicious_count": int(len(records)),
            "suspicious_percentage": round(len(records) / len(vals) * 100, 2),
            "threshold": 2.5,
            "records": records,
        }

    def generate_smart_insights(self):
        """Generate a rich set of natural-language insights that mirror a human analyst.

        Combines statistical findings, patterns, drivers, anomalies and recommendations
        into a structured, interpretable result.
        """
        name, df = self._get_main_df()
        if df is None:
            return {"error": "No data available for insight generation."}

        numeric = df.select_dtypes(include=[np.number])
        cat = df.select_dtypes(include=["object", "string", "category"])
        findings = []
        drivers = []
        anomalies = []
        recommendations = []

        # --- Dataset overview ---
        findings.append(f"The dataset '{name}' contains {len(df)} rows and {df.shape[1]} columns.")

        # --- Numeric analysis ---
        if not numeric.empty:
            top_col = numeric.sum().sort_values(ascending=False)
            if len(top_col) > 0:
                strongest = top_col.index[0]
                findings.append(
                    f"The column '{strongest}' has the highest total value ({numeric[strongest].sum():,.0f}), "
                    f"making it the most significant driver of overall totals."
                )
                drivers.append(strongest)

            # Growth / trend detection on first numeric column
            trend_col = numeric.columns[0]
            vals = pd.to_numeric(df[trend_col], errors="coerce").dropna()
            if len(vals) >= 2:
                first = vals.iloc[0]
                last = vals.iloc[-1]
                if first != 0:
                    change = (last - first) / first * 100
                    direction = "upward" if change > 0 else "downward"
                    findings.append(
                        f"'{trend_col}' shows a {direction} trend of {abs(change):.1f}% "
                        f"from the first value ({first:,.0f}) to the last ({last:,.0f})."
                    )
                    if change > 20:
                        recommendations.append(
                            f"Investigate the strong {direction} trend in '{trend_col}' to understand the "
                            f"underlying drivers and plan accordingly."
                        )

            # Correlation insight
            if numeric.shape[1] >= 2:
                corr = numeric.corr().abs()
                pairs = []
                for i, ca in enumerate(corr.columns):
                    for cb in corr.columns[i + 1:]:
                        pairs.append((ca, cb, corr.loc[ca, cb]))
                if pairs:
                    pairs.sort(key=lambda x: x[2], reverse=True)
                    ca, cb, r = pairs[0]
                    if r >= 0.5:
                        findings.append(
                            f"Strong correlation detected between '{ca}' and '{cb}' (r = {r:.2f}), "
                            f"suggesting these move together and may share underlying drivers."
                        )
                        drivers.append(f"{ca} & {cb}")
                        recommendations.append(
                            f"Consider modeling '{ca}' and '{cb}' together since they are strongly correlated "
                            f"(r = {r:.2f})."
                        )

            # Outlier insight
            z_outliers = {}
            for col in numeric.columns:
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(vals) < 4:
                    continue
                mean = vals.mean()
                std = vals.std()
                if std == 0 or pd.isna(std):
                    continue
                z = np.abs((vals - mean) / std)
                n = int((z > 3).sum())
                if n > 0:
                    z_outliers[col] = n
            if z_outliers:
                desc = ", ".join(f"{k} ({v})" for k, v in list(z_outliers.items())[:3])
                findings.append(
                    f"Detected {sum(z_outliers.values())} potential outliers across numeric columns "
                    f"({desc}). These may represent errors or exceptional events worth investigating."
                )
                for col, n in z_outliers.items():
                    anomalies.append({"column": col, "count": n})

        # --- Categorical analysis ---
        if not cat.empty:
            top_col = cat.nunique().idxmax()
            top_values = df[top_col].value_counts()
            if len(top_values) > 0:
                top_val = top_values.index[0]
                pct = top_values.iloc[0] / len(df) * 100
                findings.append(
                    f"'{top_col}' is the most diverse categorical column with {df[top_col].nunique()} "
                    f"unique values. '{top_val}' is the most frequent ({pct:.1f}% of rows)."
                )

        # --- Missing value analysis ---
        na_counts = df.isnull().sum()
        na_total = int(na_counts.sum())
        if na_total > 0:
            na_cols = [str(c) for c in na_counts[na_counts > 0].index]
            findings.append(
                f"Data quality: {na_total} missing values found across {len(na_cols)} column(s) "
                f"({', '.join(na_cols[:3])}). Consider cleaning the data before further analysis."
            )
            recommendations.append(
                "Run the 'clean' command to automatically fill or drop missing values and improve data quality."
            )
        else:
            findings.append("Data quality: No missing values detected — the dataset is complete.")

        # --- Recommendations ---
        if not numeric.empty and len(numeric.columns) >= 2:
            recommendations.append(
                "Use the 'predict' command to build a machine learning model and forecast future outcomes."
            )
        recommendations.append(
            "Use the 'chart' command to visualize the key relationships and trends identified above."
        )

        return {
            "data_name": name,
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "findings": findings,
            "key_drivers": drivers,
            "anomalies": anomalies,
            "recommendations": recommendations,
        }

    def generate_report(self):
        """Generate an executive-ready report from the data.

        Combines the statistical summary, smart insights, and recommendations
        into a structured narrative report.
        """
        name, df = self._get_main_df()
        if df is None:
            return {"error": "No data available for report generation."}

        # Base stats
        shape = {"rows": int(len(df)), "columns": int(df.shape[1])}
        numeric = df.select_dtypes(include=[np.number])
        cat = df.select_dtypes(include=["object", "category"])

        # Overview
        overview = {
            "dataset": name,
            "rows": shape["rows"],
            "columns": shape["columns"],
            "numeric_columns": list(numeric.columns),
            "categorical_columns": list(cat.columns),
            "missing_values": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
        }

        # Key metrics
        metrics = {}
        if not numeric.empty:
            for col in numeric.columns[:5]:
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(vals) == 0:
                    continue
                metrics[col] = {
                    "total": round(float(vals.sum()), 2),
                    "average": round(float(vals.mean()), 2),
                    "min": round(float(vals.min()), 2),
                    "max": round(float(vals.max()), 2),
                }

        # Insights
        smart = self.generate_smart_insights()

        return {
            "title": f"Executive Data Analysis Report — {name}",
            "dataset": name,
            "overview": overview,
            "metrics": metrics,
            "insights": smart,
            "generated_from": "Auto Data Analyst Agent",
        }

    def summary_insights(self):
        """Generate a natural-language summary of key insights from the data."""
        name, df = self._get_main_df()
        if df is None:
            return {"error": "No data available for insights."}

        numeric = df.select_dtypes(include=[np.number])
        insights = {"data_name": name, "rows": int(len(df)), "columns": int(df.shape[1])}

        if not numeric.empty:
            insights["numeric_columns"] = list(numeric.columns)
            # Highest value column
            sums = numeric.sum().sort_values(ascending=False)
            insights["highest_total_column"] = str(sums.index[0])
            insights["highest_total"] = float(sums.iloc[0])
            # Most correlated pair
            if numeric.shape[1] >= 2:
                corr = numeric.corr().abs()
                pairs = []
                for i, col_a in enumerate(corr.columns):
                    for col_b in corr.columns[i + 1:]:
                        pairs.append((col_a, col_b, corr.loc[col_a, col_b]))
                if pairs:
                    pairs.sort(key=lambda x: x[2], reverse=True)
                    insights["strongest_correlation"] = {
                        "col_a": str(pairs[0][0]),
                        "col_b": str(pairs[0][1]),
                        "r": round(float(pairs[0][2]), 3),
                    }

        # Missing values
        na_counts = df.isnull().sum()
        insights["total_null_cells"] = int(na_counts.sum())
        if na_counts.sum() > 0:
            insights["columns_with_nulls"] = [
                str(c) for c in na_counts[na_counts > 0].index
            ]

        # Categorical columns
        cat = df.select_dtypes(include=["object", "category"])
        if not cat.empty:
            insights["categorical_columns"] = list(cat.columns)
            top_col = cat.nunique().idxmax()
            insights["most_diverse_column"] = str(top_col)
            insights["unique_values_in_top"] = int(cat[top_col].nunique())

        return insights

    def aggregate(self, intent):
        """Generic aggregation on tabular data."""
        name, df = self._get_main_df()
        if df is None:
            return {"error": "No data available."}

        df = df.copy()
        result = {"data_name": name}

        # Find the target column for aggregation
        target_col = None
        if intent.column:
            for col in df.columns:
                if col.lower() == intent.column:
                    target_col = col
                    break
            if not target_col:
                # try substring match
                for col in df.columns:
                    if intent.column in col.lower():
                        target_col = col
                        break

        # Determine numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not target_col and numeric_cols:
            target_col = numeric_cols[0]

        if not target_col:
            return {"error": "No numeric column found for aggregation."}

        # Apply time filter if a date column exists
        date_col = None
        for col in df.columns:
            if col.lower() in ("date", "datetime", "month", "year", "time"):
                date_col = col
                break

        if date_col and intent.time_filter:
            try:
                df["_date"] = pd.to_datetime(df[date_col], errors="coerce")
                tf = intent.time_filter
                if "start" in tf and "end" in tf:
                    df = df[(df["_date"] >= tf["start"]) & (df["_date"] <= tf["end"])]
                elif tf.get("period") == "month":
                    df = df[(df["_date"].dt.month == tf["month"])]
                result["time_period"] = tf.get("period", "unknown")
            except Exception:
                pass

        col_type = "numeric" if target_col in numeric_cols else "categorical"
        result["target"] = target_col
        result["column_type"] = col_type

        if col_type == "numeric":
            values = pd.to_numeric(df[target_col], errors="coerce").dropna()
            if intent.metric == "total":
                result["value"] = float(values.sum())
            elif intent.metric == "average":
                result["value"] = float(values.mean())
            elif intent.metric == "count":
                result["value"] = int(len(values))
            elif intent.metric == "maximum":
                result["value"] = float(values.max())
            elif intent.metric == "minimum":
                result["value"] = float(values.min())
            else:
                result["value"] = float(values.sum())
                result.update({
                    "average": float(values.mean()),
                    "max": float(values.max()),
                    "min": float(values.min()),
                    "count": int(len(values)),
                })
        else:
            counts = df[target_col].value_counts()
            result["value"] = int(len(df))
            result["unique_values"] = int(df[target_col].nunique())
            result["top_values"] = counts.head(10).to_dict()

        # Group by
        if intent.group_by:
            gb_col = None
            for col in df.columns:
                if col.lower() == intent.group_by:
                    gb_col = col
                    break
            if gb_col and gb_col != target_col:
                if target_col in numeric_cols:
                    grouped = df.groupby(gb_col)[target_col].sum().sort_values(ascending=False).head(15)
                    result["groups"] = {str(k): float(v) for k, v in grouped.items()}
                else:
                    grouped = df.groupby(gb_col).size().sort_values(ascending=False).head(15)
                    result["groups"] = {str(k): int(v) for k, v in grouped.items()}

        return result
