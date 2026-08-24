"""Data & Model Validation Agent - Performs comprehensive statistical and modeling safety audits."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd

from agent.base import BaseAgent
from agent.schemas import AgentResult, ClaimType, ErrorCategory, Evidence
from backend.app.ml.validation_engine import DataModelValidator, IssueSeverity, ValidationAuditReport


class DataValidationAgent(BaseAgent):
    """
    Autonomous Modeling Safety & Integrity Validation Agent.
    Audits datasets and models for data leakage, class imbalance, overfitting,
    lookahead bias in time series, severe outliers, and multicollinearity.
    """
    name = "Data Validation Agent"
    role = "qa_validation_engineer"
    description = "Performs comprehensive safety and integrity audits on datasets and models."

    def __init__(self, data=None):
        super().__init__(data=data)
        self.validator = DataModelValidator()

    def run(self, task: Dict[str, Any]) -> AgentResult:
        """
        Execute comprehensive validation audit.
        Task parameters:
            - data: pd.DataFrame or dict (required)
            - target: str (target column name, optional)
            - train_score: float (optional training performance score)
            - test_score: float (optional test performance score)
            - train_dates: list or pd.Series (optional timestamps)
            - test_dates: list or pd.Series (optional timestamps)
            - auto_repair: bool (if True, drops critically leaking features and returns clean df)
        """
        self._start()
        data = task.get("data") if task.get("data") is not None else self.data

        if isinstance(data, dict):
            for df in data.values():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    df_target = df
                    break
            else:
                return self._error("No valid pandas DataFrame found in task data.", category=ErrorCategory.INPUT_VALIDATION)
        elif isinstance(data, pd.DataFrame):
            df_target = data
        else:
            return self._error("Missing required DataFrame input for validation.", category=ErrorCategory.INPUT_VALIDATION)

        target = task.get("target")
        if not target or target not in df_target.columns:
            num_cols = df_target.select_dtypes(include=["number"]).columns
            target = num_cols[-1] if len(num_cols) > 0 else df_target.columns[-1]

        train_score = task.get("train_score")
        test_score = task.get("test_score")
        train_dates = task.get("train_dates")
        test_dates = task.get("test_dates")
        auto_repair = bool(task.get("auto_repair", False))

        try:
            report: ValidationAuditReport = self.validator.audit_pipeline(
                df=df_target,
                target_column=target,
                train_score=train_score,
                test_score=test_score,
                train_dates=train_dates,
                test_dates=test_dates,
            )
        except Exception as exc:
            return self._error(f"Validation audit failed: {str(exc)}", category=ErrorCategory.COMPUTATION)

        # Build traceable Evidence
        evidence_list: List[Evidence] = []

        # 1. Overall Audit Evidence
        evidence_list.append(
            self.make_evidence(
                method="data_model_safety_audit",
                data_ref={
                    "target": target,
                    "overall_status": report.overall_status,
                    "critical_issues": report.critical_issues_count,
                    "warnings": report.warnings_count,
                },
                confidence=1.0,
                claim_type=ClaimType.FACT,
            )
        )

        # 2. Key issues evidence
        for issue in report.issues[:3]:
            evidence_list.append(
                self.make_evidence(
                    method="integrity_diagnostic",
                    data_ref={
                        "issue_type": issue.check_type.value,
                        "severity": issue.severity.value,
                        "title": issue.title,
                        "affected_columns": issue.affected_columns,
                    },
                    confidence=0.95,
                    claim_type=ClaimType.OBSERVATION,
                )
            )

        output: Dict[str, Any] = {
            "target": target,
            "overall_status": report.overall_status,
            "critical_issues_count": report.critical_issues_count,
            "warnings_count": report.warnings_count,
            "issues": [i.to_dict() for i in report.issues],
            "diagnostics": report.diagnostics,
        }

        # Auto-repair: if critical leakage was found and auto_repair is True, drop leaking columns
        if auto_repair and report.diagnostics.get("leakage", {}).get("leaking_features"):
            leaking_cols = report.diagnostics["leakage"]["leaking_features"]
            clean_df = df_target.drop(columns=[c for c in leaking_cols if c in df_target.columns])
            output["repaired_data"] = clean_df
            output["remediation_applied"] = f"Dropped {len(leaking_cols)} leaking feature(s): {leaking_cols}"

        return self._finish(
            result=output,
            evidence=evidence_list,
            confidence=1.0,
            metadata={"status": report.overall_status, "target": target},
        )
