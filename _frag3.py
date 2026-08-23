    # ------------------------------------------------------------------
    # Data cross-check
    # ------------------------------------------------------------------
    def _cross_check(self, result: AgentResult, vr: ValidationResult,
                     context: Optional[Dict[str, Any]]) -> None:
        """Verify evidence references against real data when context is given."""
        if not context:
            return
        data = context.get("dataframe") or context.get("data")
        if data is None:
            return

        actual_columns = context.get("columns")
        if actual_columns is None:
            if isinstance(data, pd.DataFrame):
                actual_columns = [str(c) for c in data.columns]
            elif isinstance(data, dict):
                actual_columns = [
                    str(c) for frame in data.values()
                    if isinstance(frame, pd.DataFrame) for c in frame.columns
                ]
            else:
                return
        actual_columns = [str(c) for c in (actual_columns or [])]

        actual_rows = context.get("row_count")
        if actual_rows is None:
            if isinstance(data, pd.DataFrame):
                actual_rows = len(data)
            elif isinstance(data, dict):
                actual_rows = max(
                    (len(f) for f in data.values() if isinstance(f, pd.DataFrame)),
                    default=0,
                )
            else:
                actual_rows = 0

        for index, evidence in enumerate(result.evidence):
            ref = evidence.data_ref or {}
            field = f"evidence[{index}].data_ref"
            self._cross_check_columns(evidence, ref, actual_columns, field, vr)
            self._cross_check_rows(evidence, ref, actual_rows, field, vr)
            if self._references_nulls(evidence):
                recomputed = self._recompute_null(data, ref)
                if recomputed is not None and isinstance(evidence.raw_value, (int, float)):
                    tolerance = max(1.0, abs(recomputed) * 0.01)
                    if abs(float(evidence.raw_value) - float(recomputed)) > tolerance:
                        vr.add_issue(
                            ValidationSeverity.ERROR, "CALCULATION_MISMATCH",
                            f"Null count in evidence ({evidence.raw_value}) does "
                            f"not match recomputation ({recomputed}).",
                            field=field, expected=recomputed,
                            actual=evidence.raw_value,
                            repair_hint="Recompute the value from the dataset.",
                        )

    def _cross_check_columns(self, evidence: Evidence, ref: Dict[str, Any],
                             actual_columns: List[str], field: str,
                             vr: ValidationResult) -> None:
        for key in ("column_names", "columns"):
            names = ref.get(key)
            if not isinstance(names, (list, tuple)):
                continue
            for name in names:
                if isinstance(name, str) and name not in actual_columns:
                    vr.add_issue(
                        ValidationSeverity.CRITICAL, "EVIDENCE_UNKNOWN_COLUMN",
                        f"Evidence references column '{name}' that is not "
                        "present in the dataset.",
                        field=f"{field}.{key}", expected=name,
                        actual=actual_columns,
                        repair_hint="Drop this evidence item or re-run the "
                                    "analysis against the actual columns.",
                    )
        single_col = ref.get("column") or ref.get("col")
        if isinstance(single_col, str) and single_col not in actual_columns:
            vr.add_issue(
                ValidationSeverity.CRITICAL, "EVIDENCE_UNKNOWN_COLUMN",
                f"Evidence references column '{single_col}' that is not "
                "present in the dataset.",
                field=field, expected=single_col, actual=actual_columns,
                repair_hint="Drop this evidence item or re-run the analysis.",
            )

