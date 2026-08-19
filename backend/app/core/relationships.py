"""Relationship Discovery Engine.

Automatically detects relationships between columns without any
dataset-specific formulas:

  - generic mathematical relationships: A + B ~ C, A - B ~ C,
    A * B ~ C, A / B ~ C
  - correlation (Pearson) with confidence
  - monotonic relationships (Spearman)
  - functional dependencies (X determines Y)
  - duplicated information (near-identical columns)
  - derived metrics (a column equals a combination of others)

Every discovered relationship carries a confidence score.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Relative tolerance used when checking mathematical identities.
RELATIVE_TOLERANCE = 0.001
ABSOLUTE_TOLERANCE = 1e-6


class RelationshipDiscoveryEngine:
    """Generic column-relationship discovery."""

    def __init__(self) -> None:
        self.relative_tolerance = RELATIVE_TOLERANCE
        self.absolute_tolerance = ABSOLUTE_TOLERANCE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def discover(self, dataframe: pd.DataFrame,
                 max_pairs: int = 200) -> dict[str, Any]:
        numeric = list(dataframe.select_dtypes(include="number").columns)
        relationships: list[dict[str, Any]] = []
        pairs = self._numeric_pairs(numeric, max_pairs)

        # Mathematical identities among triples (A op B = C).
        relationships.extend(self._mathematical_identities(dataframe, numeric))

        # Pairwise correlation / monotonic / functional checks.
        for left, right in pairs:
            relationships.extend(self._pair_relationships(dataframe, left, right))

        # Deduplicate repeated statements.
        relationships = self._dedupe(relationships)
        relationships.sort(key=lambda item: item["confidence"], reverse=True)

        return {
            "relationships": relationships,
            "summary": {
                "count": len(relationships),
                "types": self._count_types(relationships),
            },
        }

    # ------------------------------------------------------------------
    # Mathematical identities (A + B ~ C, etc.)
    # ------------------------------------------------------------------
    def _mathematical_identities(self, dataframe: pd.DataFrame,
                                 numeric: list[str]) -> list[dict[str, Any]]:
        if len(numeric) < 2:
            return []
        results: list[dict[str, Any]] = []
        # Limit triples to avoid combinatorial blowup on wide datasets.
        selected = numeric[:8] if len(numeric) > 8 else numeric
        clean = dataframe[selected].replace([np.inf, -np.inf], np.nan)
        for target in selected:
            series = pd.to_numeric(clean[target], errors="coerce")
            if series.nunique(dropna=True) <= 1:
                continue
            for a_index, left in enumerate(selected):
                if left == target:
                    continue
                for right in selected[a_index + 1:]:
                    if right == target:
                        continue
                    for operation in ("+", "-", "*", "/"):
                        relationship = self._check_identity(
                            clean, series, target, left, right, operation)
                        if relationship is not None:
                            results.append(relationship)
        return results

    def _check_identity(self, clean: pd.DataFrame, target_series: pd.Series,
                        target: str, left: str, right: str,
                        operation: str) -> dict[str, Any] | None:
        a = pd.to_numeric(clean[left], errors="coerce")
        b = pd.to_numeric(clean[right], errors="coerce")
        if operation == "+":
            computed = a + b
            formula = f"{left} + {right}"
        elif operation == "-":
            computed = a - b
            formula = f"{left} - {right}"
        elif operation == "*":
            computed = a * b
            formula = f"{left} * {right}"
        else:
            computed = a / b.replace(0, np.nan)
            formula = f"{left} / {right}"

        both = pd.DataFrame({"computed": computed, "target": target_series}).dropna()
        if len(both) < 10:
            return None

        scale = both["target"].abs().replace(0, np.nan)
        relative = ((both["computed"] - both["target"]).abs() /
                    (self.relative_tolerance + scale.abs())).dropna()
        exact = (both["computed"] - both["target"]).abs() <= self.absolute_tolerance
        match_fraction = float(((relative <= self.relative_tolerance) | exact).mean())
        if match_fraction < 0.95:
            return None

        confidence = 0.5 + 0.45 * match_fraction
        return {
            "type": "derived_metric",
            "columns": [target, left, right],
            "formula": formula,
            "target": target,
            "match_fraction": round(match_fraction, 4),
            "confidence": round(min(0.99, confidence), 3),
            "description": f"{target} closely matches {formula}.",
        }

    # ------------------------------------------------------------------
    # Pairwise relationships
    # ------------------------------------------------------------------
    def _pair_relationships(self, dataframe: pd.DataFrame, left: str, right: str) -> list[dict[str, Any]]:
        a = pd.to_numeric(dataframe[left], errors="coerce")
        b = pd.to_numeric(dataframe[right], errors="coerce")
        clean = pd.DataFrame({"a": a, "b": b}).replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) < 5:
            return []
        results: list[dict[str, Any]] = []
        a_vals = clean["a"].to_numpy(dtype=float)
        b_vals = clean["b"].to_numpy(dtype=float)

        # 1) Pearson correlation.
        if a_vals.std() > 0 and b_vals.std() > 0:
            correlation = float(np.corrcoef(a_vals, b_vals)[0, 1])
            if not np.isnan(correlation):
                results.append({
                    "type": "correlation",
                    "columns": [left, right],
                    "correlation": round(correlation, 4),
                    "confidence": round(min(0.98, abs(correlation)), 3),
                    "description": f"{left} and {right} have a Pearson correlation of {correlation:.3f}.",
                })

        # 2) Monotonic relationship (Spearman rank correlation).
        if len(clean) >= 10:
            rank_a = pd.Series(a_vals).rank().to_numpy(dtype=float)
            rank_b = pd.Series(b_vals).rank().to_numpy(dtype=float)
            if rank_a.std() > 0 and rank_b.std() > 0:
                spearman = float(np.corrcoef(rank_a, rank_b)[0, 1])
                if not np.isnan(spearman) and abs(spearman) >= 0.8:
                    results.append({
                        "type": "monotonic",
                        "columns": [left, right],
                        "spearman_correlation": round(spearman, 4),
                        "confidence": round(min(0.98, abs(spearman)), 3),
                        "description": (f"{left} and {right} move "
                                        f"{'together' if spearman > 0 else 'in opposite directions'} "
                                        f"monotonically (Spearman {spearman:.3f})."),
                    })

        # 3) Functional dependency (left determines right, or vice versa).
        dependency = self._functional_dependency(a, b, left, right)
        if dependency is not None:
            results.append(dependency)

        # 4) Duplicated information.
        duplicated = self._duplicated_info(a_vals, b_vals, left, right)
        if duplicated is not None:
            results.append(duplicated)

        return results

    def _functional_dependency(self, a: pd.Series, b: pd.Series,
                               left: str, right: str) -> dict[str, Any] | None:
        clean = pd.DataFrame({"a": a, "b": b}).dropna()
        if len(clean) < 5:
            return None
        consistent = clean.groupby("a")["b"].nunique()
        left_determines_right = float((consistent <= 1).mean())
        consistent_rev = clean.groupby("b")["a"].nunique()
        right_determines_left = float((consistent_rev <= 1).mean())

        if left_determines_right >= 0.98:
            return {"type": "functional_dependency", "columns": [left, right],
                    "determinant": left, "dependent": right,
                    "consistency": round(left_determines_right, 4),
                    "confidence": round(0.5 + 0.45 * left_determines_right, 3),
                    "description": f"{left} functionally determines {right} (consistent for {left_determines_right:.0%} of values)."}
        if right_determines_left >= 0.98:
            return {"type": "functional_dependency", "columns": [left, right],
                    "determinant": right, "dependent": left,
                    "consistency": round(right_determines_left, 4),
                    "confidence": round(0.5 + 0.45 * right_determines_left, 3),
                    "description": f"{right} functionally determines {left} (consistent for {right_determines_left:.0%} of values)."}
        return None

    def _duplicated_info(self, a: np.ndarray, b: np.ndarray,
                         left: str, right: str) -> dict[str, Any] | None:
        scale = np.maximum(np.abs(a).max(), np.abs(b).max()) or 1.0
        diff = np.abs(a - b) / scale
        identical_fraction = float((diff <= self.absolute_tolerance).mean())
        if identical_fraction >= 0.98:
            return {"type": "duplicated_information", "columns": [left, right],
                    "identical_fraction": round(identical_fraction, 4),
                    "confidence": round(0.5 + 0.45 * identical_fraction, 3),
                    "description": f"{left} and {right} carry effectively identical values."}
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _numeric_pairs(numeric: list[str], max_pairs: int) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for i, left in enumerate(numeric):
            for right in numeric[i + 1:]:
                pairs.append((left, right))
                if len(pairs) >= max_pairs:
                    return pairs
        return pairs

    def _dedupe(self, relationships: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[tuple[str, ...]] = set()
        unique: list[dict[str, Any]] = []
        for item in relationships:
            key = (item["type"], tuple(sorted(item.get("columns", []))))
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique

    @staticmethod
    def _count_types(relationships: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in relationships:
            counts[item["type"]] = counts.get(item["type"], 0) + 1
        return counts
