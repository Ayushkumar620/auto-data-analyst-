"""
Universal Statistical Relationship & Dependency Analysis Agent.

Executes autonomous bivariate and multivariate relationship discovery,
hypothesis testing, effect size measurement, and FDR correction across arbitrary tabular data.
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


class StatisticalAnalysisAgent(BaseAgent):
    """
    Autonomous Statistical Relationship & Dependency Analysis Agent.
    Identifies, tests, and explains associations across numeric, categorical, and temporal variables.
    """

    name = "Statistical Analysis Agent"
    description = "Measures, tests, and ranks statistical relationships, correlations, and dependencies without causal overclaims."
    role = "statistical_analysis"

    def run(self, task: Dict[str, Any]) -> AgentResult:
        self._start()
        try:
            from agent.pre_execution_validator import PreExecutionValidator
            from agent.confidence_calculator import ConfidenceCalculator
            from agent.result_validator import ResultValidator
            from agent.statistical_analysis_engine import StatisticalAnalysisEngine

            data = task.get("data")
            features = task.get("features") or task.get("feature_columns")
            target = task.get("target") or task.get("target_column")
            alpha = task.get("alpha", 0.05)
            max_pairs = task.get("max_pairs", 250)

            # 1. Pre-execution validation
            pre_audit = PreExecutionValidator.validate(
                data,
                task_type="statistical_analysis",
                feature_columns=features,
                agent_name=self.name,
            )
            if not pre_audit.is_valid:
                err = pre_audit.error
                return self._error(
                    message=err.user_message if err else "Statistical relationship pre-validation failed.",
                    code=err.code if err else "VALIDATION_FAILURE",
                    category=err.category if err else ErrorCategory.DATA_INVALID,
                    details=err.technical_details if err else {},
                )

            # 2. Run canonical StatisticalAnalysisEngine
            engine = StatisticalAnalysisEngine(alpha=alpha, max_pairs=max_pairs)
            result = engine.analyze(
                data=data,
                features=features,
                target=target,
                alpha=alpha,
                max_pairs=max_pairs,
            )

            if "error" in result:
                return self._error(
                    message=result["error"],
                    code="STATISTICAL_ANALYSIS_FAILED",
                    category=result.get("category", ErrorCategory.MODEL_FAILURE),
                    details=result,
                    output=result,
                )

            n_rows = result.get("rows_analyzed", 0)
            relationships = result.get("relationships", [])
            top_relationships = result.get("top_relationships", [])

            # Determine top effect size and min adjusted p-value
            top_eff = top_relationships[0].get("effect_size", 0.5) if top_relationships else 0.0
            min_adj_p = min((r.get("adjusted_p_value", 1.0) for r in relationships), default=1.0)
            has_outlier_sens = any(r.get("outlier_sensitivity", False) for r in relationships)

            # 3. Canonical Evidence generation
            evidence_list: List[Evidence] = []
            for rel in top_relationships[:5]:
                fx = rel.get("feature_x")
                fy = rel.get("feature_y")
                stat = rel.get("statistic")
                p_val = rel.get("p_value")
                adj_p = rel.get("adjusted_p_value")
                method = rel.get("primary_method", "correlation")

                evidence_list.append(
                    self.make_evidence(
                        method=f"stats.{rel.get('pair_type', 'bivariate')}.{method}",
                        data_ref={
                            "feature_x": fx,
                            "feature_y": fy,
                            "method": method,
                            "statistic": stat,
                            "p_value": p_val,
                            "adjusted_p_value": adj_p,
                            "effect_size": rel.get("effect_size"),
                            "valid_rows": rel.get("valid_rows"),
                        },
                        confidence=0.85,
                        claim_type=ClaimType.CORRELATION,
                        raw_value={
                            "statistic": stat,
                            "adjusted_p_value": adj_p,
                            "interpretation": rel.get("interpretation"),
                        },
                    )
                )

            # 4. Confidence calculation
            conf_rep = ConfidenceCalculator.calculate_statistical_relationship_confidence(
                n_samples=n_rows,
                n_pairs=len(relationships),
                top_effect_size=top_eff,
                min_adjusted_p=min_adj_p,
                outlier_sensitivity=has_outlier_sens,
            )

            raw_res = self._finish(
                result,
                evidence=evidence_list,
                confidence=conf_rep.confidence,
                model_used="StatisticalAnalysisEngine",
            )

            # 5. Result validation and repair
            repaired_res, _ = ResultValidator().repair(raw_res, context={"data": data})
            return repaired_res
        except Exception as e:
            return self._error(str(e), category=ErrorCategory.COMPUTATION)