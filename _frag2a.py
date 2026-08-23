    def _schema_check(self, result: AgentResult, vr: ValidationResult) -> None:
        if not isinstance(result.agent, str) or not result.agent.strip():
            vr.add_issue(ValidationSeverity.ERROR, "MISSING_AGENT_NAME",
                         "AgentResult.agent must be a non-empty string.")
        if not isinstance(result.role, str) or not result.role.strip():
            vr.add_issue(ValidationSeverity.ERROR, "MISSING_ROLE",
                         "AgentResult.role must be a non-empty string.")
        if not isinstance(result.agent_id, str) or not result.agent_id.strip():
            vr.add_issue(ValidationSeverity.ERROR, "MISSING_AGENT_ID",
                         "AgentResult.agent_id must be a non-empty string.")
        if not isinstance(result.status, AgentStatus):
            vr.add_issue(ValidationSeverity.ERROR, "INVALID_STATUS",
                         "AgentResult.status must be an AgentStatus.")
        if not isinstance(result.output, dict):
            vr.add_issue(ValidationSeverity.ERROR, "INVALID_OUTPUT",
                         "AgentResult.output must be a dict.",
                         field="output", actual=type(result.output).__name__)
        if not isinstance(result.confidence, (int, float)):
            vr.add_issue(ValidationSeverity.ERROR, "INVALID_CONFIDENCE_TYPE",
                         "AgentResult.confidence must be numeric.",
                         field="confidence")
        elif not 0.0 <= float(result.confidence) <= 1.0:
            vr.add_issue(ValidationSeverity.ERROR, "CONFIDENCE_OUT_OF_RANGE",
                         "AgentResult.confidence must be within [0, 1].",
                         field="confidence", actual=result.confidence)

    def _consistency_check(self, result: AgentResult, vr: ValidationResult) -> None:
        if result.status == AgentStatus.COMPLETED:
            if result.finished_at is None:
                vr.add_issue(ValidationSeverity.ERROR, "MISSING_FINISHED_AT",
                             "A completed result must have finished_at set.")
            if result.duration_ms < 0:
                vr.add_issue(ValidationSeverity.ERROR, "NEGATIVE_DURATION",
                             "A completed result cannot have a negative duration_ms.",
                             field="duration_ms", actual=result.duration_ms)
            if not result.output:
                vr.add_issue(ValidationSeverity.WARNING, "EMPTY_COMPLETED",
                             "A completed result carries no output.")
        elif result.status == AgentStatus.ERROR:
            if not result.errors:
                vr.add_issue(ValidationSeverity.WARNING, "ERROR_WITHOUT_ERRORS",
                             "An error result should carry at least one AgentError.",
                             field="errors")

