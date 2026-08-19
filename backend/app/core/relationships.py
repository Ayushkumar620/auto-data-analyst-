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
