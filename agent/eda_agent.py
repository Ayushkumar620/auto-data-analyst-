"""
Universal EDA, Data Profiling & Data Quality Intelligence Agent.

Orchestrates CanonicalDataLayer, PreExecutionValidator, EDAEngine, ConfidenceCalculator,
and ResultValidator into the canonical AgentResult lifecycle.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import pandas as pd

from agent.agent_result import (
    AgentError,
    AgentResult,
    AgentStatus,
    ClaimType,
    ErrorCategory,
    Evidence,
)
from agent.base import BaseAgent


class EDAAgent(BaseAgent):
    """
    Autonomous Exploratory Data Analysis & Data Quality Profiling Agent.
    Profiles arbitrary tabular data without hardcoded column names or destructive drops.
    """

    name = "EDA Agent"
    description = "Profiles dataset schema, statistical distributions, missing values, duplicates, and data quality."
    role = "eda"

    def run(self, task: Dict[str, Any]) -> AgentResult:
        self._start()
        try:
            from agent.pre_execution_validator import PreExecutionValidator
            from agent.confidence_calculator import ConfidenceCalculator
            from agent.result_validator import ResultValidator
            from agent.eda_engine import EDAEngine

            data = task.get("data")
            selected_columns = task.get("columns") or task.get("selected_columns") or task.get("features")
            max_categories = task.get("max_categories", 10)

            # 1. Pre-execution validation
            pre_audit = PreExecutionValidator.validate(
                data,
                task_type="eda",
                feature_columns=selected_columns,
                agent_name=self.name,
            )
            if not pre_audit.is_valid:
                err = pre_audit.error
                return self._error(
                    message=err.user_message if err else "EDA pre-execution validation failed.",
                    code=err.code if err else "VALIDATION_FAILURE",
                    category=err.category if err else ErrorCategory.DATA_INVALID,
                    details=err.technical_details if err else {},
                )

            # 2. Run canonical EDAEngine
            engine = EDAEngine()
            result = engine.profile(
                data=data,
                selected_columns=selected_columns,
                max_categories=max_categories,
            )

            if "error" in result:
                return self._error(
                    message=result["error"],
                    code="EDA_PROFILING_FAILED",
                    category=result.get("category", ErrorCategory.MODEL_FAILURE),
                    details=result,
                    output=result,
                )

            summary = result.get("summary", {})
            n_rows = summary.get("row_count", 0)
            n_cols = summary.get("column_count", 0)
            missing_analysis = result.get("missing_analysis", {})
            data_quality = result.get("data_quality", {})
            columns_prof = result.get("columns", {})

            # 3. Canonical Evidence Generation (ClaimType.OBSERVATION)
            evidence_list: List[Evidence] = []

            # A. Dataset structural evidence
            evidence_list.append(
                self.make_evidence(
                    method="eda.structure.summary",
                    data_ref={
                        "row_count": n_rows,
                        "column_count": n_cols,
                        "numeric_columns": summary.get("numeric_columns", []),
                        "categorical_columns": summary.get("categorical_columns", []),
                        "datetime_columns": summary.get("datetime_columns", []),
                    },
                    confidence=0.95,
                    claim_type=ClaimType.OBSERVATION,
                    raw_value={
                        "rows": n_rows,
                        "columns": n_cols,
                        "duplicate_rows": summary.get("duplicate_rows", 0),
                    },
                )
            )

            # B. Quality score evidence
            if "quality_score" in data_quality:
                evidence_list.append(
                    self.make_evidence(
                        method="eda.quality.assessment",
                        data_ref={
                            "quality_score": data_quality["quality_score"],
                            "quality_rating": data_quality.get("quality_rating"),
                            "components": data_quality.get("components"),
                        },
                        confidence=0.90,
                        claim_type=ClaimType.OBSERVATION,
                        raw_value=data_quality,
                    )
                )

            # C. Missing data evidence
            if missing_analysis:
                evidence_list.append(
                    self.make_evidence(
                        method="eda.missing.analysis",
                        data_ref={
                            "total_missing_cells": missing_analysis.get("total_missing_cells", 0),
                            "missing_percentage": missing_analysis.get("overall_missing_percentage", 0.0),
                            "complete_columns": missing_analysis.get("complete_columns_count", 0),
                        },
                        confidence=0.95,
                        claim_type=ClaimType.OBSERVATION,
                        raw_value=int(missing_analysis.get("total_missing_cells", 0)),
                    )
                )

            # 4. Confidence Calculation
            missing_rate = float(missing_analysis.get("overall_missing_percentage", 0.0)) / 100.0
            unusable_cols_cnt = len(summary.get("empty_columns", [])) + len(summary.get("constant_columns", []))
            unusable_ratio = unusable_cols_cnt / max(1, n_cols)
            parse_rate = data_quality.get("components", {}).get("validity", 1.0)

            conf_rep = ConfidenceCalculator.calculate_eda_confidence(
                n_rows=n_rows,
                n_cols=n_cols,
                missing_rate=missing_rate,
                unusable_ratio=unusable_ratio,
                parse_success_rate=parse_rate,
            )

            raw_res = self._finish(
                result,
                evidence=evidence_list,
                confidence=conf_rep.confidence,
                model_used="EDAEngine",
            )

            # 5. Result Validation & Invariant Repair
            repaired_res, _ = ResultValidator().repair(raw_res, context={"data": data})
            return repaired_res
        except Exception as e:
            return self._error(str(e), category=ErrorCategory.COMPUTATION)