"""Deterministic rules that convert measured facts into conservative insights."""

from __future__ import annotations

from typing import Any, Dict, List

from .schemas import Insight


class InsightRules:
    def evaluate(self, facts: Dict[str, Any]) -> List[Insight]:
        insights: List[Insight] = []
        missing = facts["missing_percentage"]
        if missing > 20:
            insights.append(Insight("risk", "Data Quality Warning",
                f"{missing}% of dataset cells are missing, which may limit the reliability of this analysis.",
                "warning", "high", {"metric": "missing_percentage", "value": missing},
                "Address missing values before making high-impact decisions."))
        for growth in facts["growth"]:
            value, column = growth["growth_percentage"], growth["column"]
            if value > 10:
                insights.append(Insight("trend", f"{self._label(column)} Growth",
                    f"{self._label(column)} increased by {value}% during the analyzed period.", "info", "high",
                    {"metric": f"{column}_growth", "value": value, **growth}))
            elif value < -10:
                insights.append(Insight("risk", f"{self._label(column)} Decline",
                    f"{self._label(column)} decreased by {abs(value)}% during the analyzed period.", "warning", "high",
                    {"metric": f"{column}_growth", "value": value, **growth},
                    f"Investigate the drivers of the {self._label(column).lower()} decline."))
        for correlation in facts["correlations"]:
            value = correlation["correlation"]
            if abs(value) < 0.7:
                continue
            left, right = self._label(correlation["left"]), self._label(correlation["right"])
            direction = "positive" if value > 0 else "negative"
            insights.append(Insight("key_finding", f"Strong {left}-{right} Relationship",
                f"{left} and {right} have a strong {direction} association (correlation {value}).",
                "info", "high", {"metric": "correlation", "value": value, **correlation}))
        for anomaly in facts["anomalies"]:
            count, column = anomaly["anomaly_count"], anomaly["column"]
            insights.append(Insight("anomaly", f"Unusual {self._label(column)} Values",
                f"{count} unusual value{'s' if count != 1 else ''} were detected in {self._label(column)}.",
                "warning", "high", {"metric": "anomaly_count", "value": count, **anomaly},
                "Review these records to determine whether they are valid exceptions or data issues."))
        for share in facts["category_shares"]:
            if share["share"] >= 40:
                column, category, value = share["column"], share["category"], share["share"]
                metric_label = "revenue" if share.get("metric") == "revenue" else "observed records"
                insights.append(Insight("opportunity", f"Concentration in {self._label(column)}",
                    f"{category} accounts for {value}% of {metric_label} within {self._label(column).lower()}.",
                    "info", "medium", {"metric": "category_share", "value": value, **share},
                    f"Investigate whether additional investment in {category} could support continued performance."))
        return insights

    @staticmethod
    def _label(value: str) -> str:
        return str(value).replace("_", " ").title()
