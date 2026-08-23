    @staticmethod
    def _cross_check_rows(evidence: Evidence, ref: Dict[str, Any],
                          actual_rows: int, field: str,
                          vr: ValidationResult) -> None:
        if actual_rows is None:
            return
        claimed_rows = ref.get("rows") or ref.get("row_count")
        if isinstance(claimed_rows, (int, float)) and claimed_rows > actual_rows:
            vr.add_issue(
                ValidationSeverity.WARNING, "EVIDENCE_ROWS_EXCEEDED",
                f"Evidence claims {claimed_rows} rows but the dataset has "
                f"{actual_rows}.",
                field=field, expected=actual_rows, actual=claimed_rows,
                repair_hint="Clamp the row count to the dataset size.",
            )

    @staticmethod
    def _references_nulls(evidence: Evidence) -> bool:
        method = (evidence.method or "").casefold()
        return "null" in method or "missing" in method

    def _recompute_null(self, data: Any, ref: Dict[str, Any]) -> Optional[int]:
        frame = data
        if isinstance(data, dict):
            frame_name = ref.get("frame")
            if frame_name in data:
                frame = data[frame_name]
            else:
                return None
        if not isinstance(frame, pd.DataFrame):
            return None
        column = ref.get("column") or ref.get("col")
        if column is not None and column in frame.columns:
            return int(frame[column].isnull().sum())
        return int(frame.isnull().sum().sum())

    # ------------------------------------------------------------------
    # Repair
    # ------------------------------------------------------------------
    def _attempt_repair(self, result: AgentResult,
                        context: Optional[Dict[str, Any]],
                        actions: List[str]) -> bool:
        """Apply one round of safe repairs. Returns True if anything changed."""
        changed = False

        # 1) Clamp / fix out-of-range confidence.
        if not isinstance(result.confidence, (int, float)):
            result.confidence = 1.0
            actions.append("Set missing AgentResult.confidence to 1.0.")
            changed = True
        elif not 0.0 <= float(result.confidence) <= 1.0:
            result.confidence = max(0.0, min(1.0, float(result.confidence)))
            actions.append(f"Clamped AgentResult.confidence to {result.confidence}.")
            changed = True

        # 2) Stamp timestamps on completed results.
        if result.status == AgentStatus.COMPLETED and result.finished_at is None:
            result.finished_at = result.started_at
            actions.append("Stamped finished_at from started_at.")
            changed = True

        # 3) Normalize / drop broken evidence items.
        kept: List[Evidence] = []
        for evidence in result.evidence:
            if not isinstance(evidence, Evidence):
                actions.append("Dropped an item from evidence that was not Evidence.")
                changed = True
                continue
            if not isinstance(evidence.confidence, (int, float)) or not (
                    0.0 <= float(evidence.confidence) <= 1.0):
                actions.append("Dropped an evidence item with invalid confidence.")
                changed = True
                continue
            if not isinstance(evidence.claim_type, ClaimType):
                actions.append("Dropped an evidence item with invalid claim_type.")
                changed = True
                continue
            if evidence.data_ref and self._references_unknown_column(evidence, context):
                actions.append("Dropped an evidence item referencing an unknown column.")
                changed = True
                continue
            kept.append(evidence)
        result.evidence = kept
        return changed

    @staticmethod
    def _references_unknown_column(
        evidence: Evidence, context: Optional[Dict[str, Any]]
    ) -> bool:
        if not context:
            return False
        data = context.get("dataframe") or context.get("data")
        if data is None:
            return False
        columns = context.get("columns")
        if columns is None:
            if isinstance(data, pd.DataFrame):
                real = [str(c) for c in data.columns]
            elif isinstance(data, dict):
                real = [str(c) for frame in data.values()
                        if isinstance(frame, pd.DataFrame) for c in frame.columns]
            else:
                real = []
        else:
            real = [str(c) for c in columns]
        ref = evidence.data_ref or {}
        names = ref.get("column_names") or ref.get("columns") or []
        if not isinstance(names, (list, tuple)):
            names = [ref.get("column")] if ref.get("column") else []
        return any(isinstance(n, str) and n not in real for n in names)


def validate_agent_result(
    result: AgentResult, context: Optional[Dict[str, Any]] = None
) -> ValidationResult:
    """Validate a result and attach the outcome to ``result.validation``."""
    return ResultValidator().validate(result, context)
