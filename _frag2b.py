    def _evidence_check(self, result: AgentResult, vr: ValidationResult) -> None:
        for index, evidence in enumerate(result.evidence):
            field = f"evidence[{index}]"
            if not isinstance(evidence, Evidence):
                vr.add_issue(ValidationSeverity.ERROR, "INVALID_EVIDENCE_TYPE",
                             "Every evidence item must be an Evidence instance.",
                             field=field)
                continue
            if not isinstance(evidence.method, str) or not evidence.method.strip():
                vr.add_issue(ValidationSeverity.ERROR, "EVIDENCE_NO_METHOD",
                             "Evidence must name the method used.",
                             field=field)
            if not isinstance(evidence.source, str) or not evidence.source.strip():
                vr.add_issue(ValidationSeverity.ERROR, "EVIDENCE_NO_SOURCE",
                             "Evidence must name its source agent.",
                             field=field)
            if not isinstance(evidence.data_ref, dict):
                vr.add_issue(ValidationSeverity.WARNING, "EVIDENCE_NO_DATA_REF",
                             "Evidence.data_ref should describe the data used.",
                             field=field)
            if not isinstance(evidence.confidence, (int, float)):
                vr.add_issue(ValidationSeverity.ERROR, "EVIDENCE_BAD_CONFIDENCE",
                             "Evidence confidence must be numeric.",
                             field=field)
            elif not 0.0 <= float(evidence.confidence) <= 1.0:
                vr.add_issue(ValidationSeverity.ERROR, "EVIDENCE_CONFIDENCE_RANGE",
                             "Evidence confidence must be within [0, 1].",
                             field=field, actual=evidence.confidence,
                             repair_hint="Clamp to [0, 1].")
            if not isinstance(evidence.claim_type, ClaimType):
                vr.add_issue(ValidationSeverity.ERROR, "INVALID_CLAIM_TYPE",
                             "Evidence.claim_type must be a ClaimType.",
                             field=field, actual=evidence.claim_type)

    def _claim_integrity_check(self, result: AgentResult, vr: ValidationResult) -> None:
        for index, evidence in enumerate(result.evidence):
            if evidence.claim_type != ClaimType.CORRELATION:
                continue
            meta = evidence.metadata or {}
            probe = [meta.get("description"), meta.get("interpretation"),
                     meta.get("claim")]
            if _looks_causal(*probe):
                vr.add_issue(
                    ValidationSeverity.ERROR, "CORRELATION_AS_CAUSATION",
                    "A CORRELATION evidence item must not imply causation.",
                    field=f"evidence[{index}]",
                    actual=" ".join(_text(p) for p in probe),
                    repair_hint="Rewrite as an INFERENCE with explicit causal "
                                "framing and lower confidence.",
                )

