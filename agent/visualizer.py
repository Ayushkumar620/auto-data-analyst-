"""
Data Visualizer - Generates charts, graphs, and evidence-grounded chart summaries.
Supported Types: Bar, Line, Scatter, Box, Pie, Histogram, Heatmap, Area.
"""
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class DataVisualizer:
    """Creates visualizations and automated summaries from DataFrames."""

    SUPPORTED_CHARTS = {
        "bar": "Bar Chart (Categorical comparisons & totals)",
        "line": "Line Chart (Trends over time & sequence tracking)",
        "scatter": "Scatter Plot (Correlation & 2-variable relationships)",
        "box": "Box Plot (Distribution, quartiles & outlier detection)",
        "pie": "Pie / Donut Chart (Proportions & market share)",
        "histogram": "Histogram (Frequency distribution & skewness)",
        "heatmap": "Correlation Heatmap (Feature correlation matrix)",
        "area": "Area Chart (Cumulative progression over time)",
    }

    def __init__(self, data):
        self.data = data

    def _get_frames(self):
        if isinstance(self.data, dict):
            return list(self.data.items())
        return [("data", self.data)]

    def _to_base64(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def chart(self, chart_type="auto", x=None, y=None):
        """Generate a chart and statistical summary from available DataFrame(s)."""
        results = []
        for name, df in self._get_frames():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue

            actual_type = (
                self._infer_chart(df)
                if chart_type in ("auto", "chart", "plot", "graph", "visualize")
                else chart_type.lower()
            )

            fig, ax = self._make_chart(df, actual_type, x, y)
            summary_text = self._generate_chart_summary(df, actual_type, x, y)

            results.append(
                {
                    "name": name,
                    "image": self._to_base64(fig),
                    "chart_type": actual_type,
                    "summary": summary_text,
                    "available_types": list(self.SUPPORTED_CHARTS.keys()),
                }
            )
        return results

    def _infer_chart(self, df):
        numeric = df.select_dtypes(include=[np.number])
        if len(numeric.columns) >= 2:
            return "scatter"
        cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns
        if len(cat_cols) >= 1:
            return "bar"
        return "line"

    def _make_chart(self, df, chart_type, x, y):
        numeric = df.select_dtypes(include=[np.number])
        cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#1e1e2e")
        ax.set_facecolor("#1e1e2e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#45475a")

        try:
            if chart_type in ("histogram", "hist"):
                col = x if x and x in numeric.columns else (numeric.columns[0] if len(numeric.columns) else df.columns[0])
                data_clean = pd.to_numeric(df[col], errors="coerce").dropna()
                ax.hist(data_clean, bins=20, color="#89b4fa", edgecolor="#1e1e2e", alpha=0.85)
                ax.set_title(f"Histogram of {col}", fontsize=14, pad=12)
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")

            elif chart_type == "scatter":
                colx = x if x and x in df.columns else (numeric.columns[0] if len(numeric.columns) >= 1 else df.columns[0])
                coly = y if y and y in df.columns else (numeric.columns[1] if len(numeric.columns) >= 2 else numeric.columns[0])
                ax.scatter(df[colx], df[coly], alpha=0.65, color="#f38ba8", edgecolors="none", s=50)
                ax.set_title(f"Scatter Plot: {colx} vs {coly}", fontsize=14, pad=12)
                ax.set_xlabel(colx)
                ax.set_ylabel(coly)

            elif chart_type == "bar":
                col = x if x and x in df.columns else (cat_cols[0] if len(cat_cols) else df.columns[0])
                if y and y in numeric.columns:
                    grouped = df.groupby(col)[y].sum().sort_values(ascending=False).head(12)
                    ax.bar(grouped.index.astype(str), grouped.values, color="#a6e3a1", edgecolor="#1e1e2e")
                    ax.set_ylabel(f"Sum of {y}")
                else:
                    counts = df[col].value_counts().head(12)
                    ax.bar(counts.index.astype(str), counts.values, color="#a6e3a1", edgecolor="#1e1e2e")
                    ax.set_ylabel("Count")
                ax.set_title(f"Bar Chart: {col}", fontsize=14, pad=12)
                ax.set_xlabel(col)
                plt.xticks(rotation=45, ha="right")

            elif chart_type == "line":
                colx = x if x and x in df.columns else df.columns[0]
                coly = y if y and y in df.columns else (numeric.columns[0] if len(numeric.columns) else df.columns[1])
                ax.plot(df[colx], df[coly], color="#89b4fa", linewidth=2.5, marker="o", markersize=4)
                ax.set_title(f"Line Chart: {colx} vs {coly}", fontsize=14, pad=12)
                ax.set_xlabel(colx)
                ax.set_ylabel(coly)
                plt.xticks(rotation=30, ha="right")

            elif chart_type == "box":
                num_df = df.select_dtypes(include=[np.number])
                if not num_df.empty:
                    num_df.boxplot(ax=ax, patch_artist=True, boxprops=dict(facecolor="#cba6f7", color="white"),
                                  medianprops=dict(color="#f38ba8", linewidth=2),
                                  whiskerprops=dict(color="white"),
                                  capprops=dict(color="white"))
                    ax.set_title("Box Plot of Numeric Features", fontsize=14, pad=12)
                    plt.xticks(rotation=45, ha="right")

            elif chart_type == "pie":
                col = x if x and x in df.columns else (cat_cols[0] if len(cat_cols) else df.columns[0])
                counts = df[col].value_counts().head(7)
                colors = ["#89b4fa", "#a6e3a1", "#f9e2af", "#f38ba8", "#cba6f7", "#94e2d5", "#fab387"]
                wedges, texts, autotexts = ax.pie(
                    counts.values,
                    labels=counts.index.astype(str),
                    autopct="%1.1f%%",
                    colors=colors[:len(counts)],
                    textprops=dict(color="white"),
                )
                for autotext in autotexts:
                    autotext.set_color("#1e1e2e")
                    autotext.set_weight("bold")
                ax.set_title(f"Pie Chart Breakdown: {col}", fontsize=14, pad=12)

            elif chart_type == "heatmap":
                corr = df.select_dtypes(include=[np.number]).corr()
                if not corr.empty:
                    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
                    ax.set_xticks(range(len(corr.columns)))
                    ax.set_yticks(range(len(corr.columns)))
                    ax.set_xticklabels(corr.columns, rotation=45, ha="right", color="white")
                    ax.set_yticklabels(corr.columns, color="white")
                    cbar = fig.colorbar(im, ax=ax)
                    cbar.ax.yaxis.set_tick_params(color="white")
                    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
                    ax.set_title("Correlation Heatmap Matrix", fontsize=14, pad=12)

            elif chart_type == "area":
                colx = x if x and x in df.columns else df.columns[0]
                coly = y if y and y in df.columns else (numeric.columns[0] if len(numeric.columns) else df.columns[1])
                ax.fill_between(range(len(df[colx])), df[coly], color="#89dceb", alpha=0.4)
                ax.plot(range(len(df[colx])), df[coly], color="#89dceb", linewidth=2)
                ax.set_title(f"Area Chart: {coly} Progression", fontsize=14, pad=12)
                ax.set_ylabel(coly)

            else:
                raise ValueError(f"Unknown chart type: {chart_type}")

        except Exception as e:
            ax.clear()
            ax.text(0.5, 0.5, f"Visualization Error: {e}", ha="center", va="center", color="white", fontsize=12)
            ax.set_title("Chart Could Not Be Plotted", color="white")

        fig.tight_layout()
        return fig, ax

    def _generate_chart_summary(self, df: pd.DataFrame, chart_type: str, x: str = None, y: str = None) -> str:
        """Generate concrete, evidence-backed narrative summary of the plotted chart."""
        numeric = df.select_dtypes(include=[np.number])
        cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns

        try:
            if chart_type in ("histogram", "hist"):
                col = x if x and x in numeric.columns else (numeric.columns[0] if len(numeric.columns) else df.columns[0])
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                mean_val = s.mean()
                median_val = s.median()
                std_val = s.std()
                skew_val = s.skew()
                skew_desc = "right-skewed (positive tail)" if skew_val > 0.5 else ("left-skewed (negative tail)" if skew_val < -0.5 else "approximately symmetric")
                return (
                    f"Histogram for '{col}': The data spans from {s.min():,.2f} to {s.max():,.2f} with a mean of {mean_val:,.2f} "
                    f"and median of {median_val:,.2f} (std: {std_val:,.2f}). The frequency distribution is {skew_desc}."
                )

            elif chart_type == "bar":
                col = x if x and x in df.columns else (cat_cols[0] if len(cat_cols) else df.columns[0])
                if y and y in numeric.columns:
                    grouped = df.groupby(col)[y].sum().sort_values(ascending=False)
                    top_k, top_v = grouped.index[0], grouped.iloc[0]
                    bot_k, bot_v = grouped.index[-1], grouped.iloc[-1]
                    return (
                        f"Bar Chart for '{col}' by '{y}': Highest cumulative volume belongs to '{top_k}' with {top_v:,.2f}, "
                        f"while '{bot_k}' is lowest with {bot_v:,.2f}. Plotted {len(grouped)} distinct categories."
                    )
                else:
                    counts = df[col].value_counts()
                    top_k, top_v = counts.index[0], counts.iloc[0]
                    pct = (top_v / len(df)) * 100
                    return (
                        f"Bar Chart for '{col}': '{top_k}' is the dominant category with {top_v:,} occurrences ({pct:.1f}% of total). "
                        f"Total categories analyzed: {len(counts)}."
                    )

            elif chart_type == "scatter":
                colx = x if x and x in df.columns else (numeric.columns[0] if len(numeric.columns) >= 1 else df.columns[0])
                coly = y if y and y in df.columns else (numeric.columns[1] if len(numeric.columns) >= 2 else numeric.columns[0])
                valid = df[[colx, coly]].dropna()
                corr = valid[colx].corr(valid[coly])
                strength = "strong positive" if corr > 0.7 else ("moderate positive" if corr > 0.3 else ("strong negative" if corr < -0.7 else ("moderate negative" if corr < -0.3 else "weak/neutral")))
                return (
                    f"Scatter Plot of '{colx}' vs '{coly}': Statistically demonstrates a {strength} correlation (r = {corr:.2f}) "
                    f"across {len(valid):,} data points."
                )

            elif chart_type == "line":
                colx = x if x and x in df.columns else df.columns[0]
                coly = y if y and y in df.columns else (numeric.columns[0] if len(numeric.columns) else df.columns[1])
                s = pd.to_numeric(df[coly], errors="coerce").dropna()
                if len(s) >= 2:
                    delta = ((s.iloc[-1] - s.iloc[0]) / max(abs(s.iloc[0]), 1e-9)) * 100
                    trend = "upward trajectory (+{:.1f}%)".format(delta) if delta > 0 else "downward trajectory ({:.1f}%)".format(delta)
                    return (
                        f"Line Chart of '{coly}' along '{colx}': Follows an overall {trend} from start ({s.iloc[0]:,.2f}) to end ({s.iloc[-1]:,.2f}). "
                        f"Peak value of {s.max():,.2f} recorded."
                    )

            elif chart_type == "pie":
                col = x if x and x in df.columns else (cat_cols[0] if len(cat_cols) else df.columns[0])
                counts = df[col].value_counts()
                top_k, top_v = counts.index[0], counts.iloc[0]
                pct = (top_v / max(len(df), 1)) * 100
                return (
                    f"Pie Chart Breakdown of '{col}': '{top_k}' holds the largest share at {pct:.1f}% ({top_v:,} records), "
                    f"followed by '{counts.index[1]}' at {(counts.iloc[1]/len(df))*100:.1f}%."
                )

            elif chart_type == "heatmap":
                corr = df.select_dtypes(include=[np.number]).corr()
                if not corr.empty and len(corr) >= 2:
                    # Find max off-diagonal correlation
                    np.fill_diagonal(corr.values, 0)
                    max_c = corr.max().max()
                    min_c = corr.min().min()
                    return (
                        f"Correlation Matrix: Evaluated {len(corr.columns)} numeric variables. Maximum positive feature coupling "
                        f"reaches r = {max_c:.2f}, while lowest negative correlation is r = {min_c:.2f}."
                    )

            elif chart_type == "box":
                num = df.select_dtypes(include=[np.number])
                if not num.empty:
                    col = num.columns[0]
                    q1 = num[col].quantile(0.25)
                    q3 = num[col].quantile(0.75)
                    iqr = q3 - q1
                    outliers = ((num[col] < (q1 - 1.5 * iqr)) | (num[col] > (q3 + 1.5 * iqr))).sum()
                    return (
                        f"Box Plot Distribution: Primary feature '{col}' has median = {num[col].median():,.2f}, "
                        f"IQR = {iqr:,.2f} (Q1: {q1:,.2f}, Q3: {q3:,.2f}). Detected {outliers} statistical outliers."
                    )

            elif chart_type == "area":
                coly = y if y and y in numeric.columns else (numeric.columns[0] if len(numeric.columns) else df.columns[0])
                s = pd.to_numeric(df[coly], errors="coerce").dropna()
                return (
                    f"Area Chart of '{coly}': Total cumulative sum across the sequence is {s.sum():,.2f} "
                    f"(average volume per step: {s.mean():,.2f})."
                )

        except Exception:
            pass

        return f"Visualization generated for chart type '{chart_type}'."
