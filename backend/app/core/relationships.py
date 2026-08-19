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
