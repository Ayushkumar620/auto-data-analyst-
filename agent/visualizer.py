"""
Data Visualizer - Generates charts and graphs using matplotlib.
"""
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class DataVisualizer:
    """Creates visualizations from DataFrames."""

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
        """Generate a chart from the first available DataFrame."""
        results = []
        for name, df in self._get_frames():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            fig, ax = self._make_chart(df, chart_type, x, y)
            actual_type = chart_type if chart_type not in ("auto", "chart", "plot", "graph", "visualize") else self._infer_chart(df)
            results.append(
                {
                    "name": name,
                    "image": self._to_base64(fig),
                    "chart_type": actual_type,
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

        if chart_type in ("auto", "chart", "plot", "graph", "visualize"):
            chart_type = self._infer_chart(df)

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#1e1e2e")
        ax.set_facecolor("#1e1e2e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("white")

        try:
            if chart_type == "histogram" or chart_type == "hist":
                col = numeric.columns[0]
                ax.hist(df[col].dropna(), bins=20, color="#89b4fa", edgecolor="#1e1e2e")
                ax.set_title(f"Histogram of {col}")
                ax.set_xlabel(col)
            elif chart_type == "scatter":
                colx = x if x and x in df.columns else numeric.columns[0]
                coly = y if y and y in df.columns else numeric.columns[1]
                ax.scatter(df[colx], df[coly], alpha=0.6, color="#f38ba8")
                ax.set_title(f"Scatter: {colx} vs {coly}")
                ax.set_xlabel(colx)
                ax.set_ylabel(coly)
            elif chart_type == "bar":
                col = x if x and x in df.columns else (cat_cols[0] if len(cat_cols) else df.columns[0])
                counts = df[col].value_counts().head(15)
                ax.bar(counts.index.astype(str), counts.values, color="#a6e3a1")
                ax.set_title(f"Bar Chart: {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
                plt.xticks(rotation=45, ha="right")
            elif chart_type == "line":
                colx = x if x and x in df.columns else df.columns[0]
                coly = y if y and y in df.columns else numeric.columns[0]
                ax.plot(df[colx], df[coly], color="#89b4fa", linewidth=2)
                ax.set_title(f"Line Chart: {colx} vs {coly}")
                ax.set_xlabel(colx)
                ax.set_ylabel(coly)
            elif chart_type == "box":
                df.select_dtypes(include=[np.number]).boxplot(ax=ax)
                ax.set_title("Box Plot of Numeric Columns")
                plt.xticks(rotation=45, ha="right")
            elif chart_type == "pie":
                col = x if x and x in df.columns else (cat_cols[0] if len(cat_cols) else df.columns[0])
                counts = df[col].value_counts().head(8)
                ax.pie(counts.values, labels=counts.index.astype(str), autopct="%1.1f%%")
                ax.set_title(f"Pie Chart: {col}")
            else:
                raise ValueError(f"Unknown chart type: {chart_type}")
        except Exception as e:
            ax.clear()
            ax.text(0.5, 0.5, f"Chart error: {e}", ha="center", va="center", color="white")
            ax.set_title("Chart could not be generated", color="white")

        fig.tight_layout()
        return fig, ax
