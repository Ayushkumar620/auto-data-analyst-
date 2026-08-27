"""
Universal Hypothesis Testing & Statistical Significance Agent.

Orchestrates PreExecutionValidator, CanonicalDataLayer, HypothesisTestingEngine,
ConfidenceCalculator, and ResultValidator into the canonical AgentResult lifecycle.
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


class HypothesisTestingAgent(BaseAgent):
    """
    Autonomous Hypothesis Testing & Statistical Significance Agent.
    Evaluates group differences and associations with data-driven test selection,
    robust assumption diagnostics, effect sizes, and non-causal reporting.
    """

    name = "Hypothesis Testing Agent"
    description = "Performs data-driven statistical hypothesis tests (t-tests, ANOVA, Mann-Whitney, Kruskal-Wallis, Chi-Square, Fisher's Exact) with effect sizes and FDR corrections."
    role = "hypothesis_testing"

    def run(self, task: Dict[str, Any]) -> AgentResult:
        self._start()
        try:
            from agent.pre_execution_validator import PreExecutionValidator
            from agent.confidence_calculator import ConfidenceCalculator
            from agent.result_validator import ResultValidator
            from agent.hypothesis_testing_engine import HypothesisTestingEngine

            data = task.get("data")
            feature = task.get("feature") or task.get("variable_x")
            group = task.get("group") or task.get("target") or task.get("variable_group")
            feature_2 = task.get("feature_2") or task.get("variable_y")
            features = task.get("features") or task.get("selected_columns")
            alpha = task.get("alpha")
            paired = task.get("paired")
            preferred_test = task.get("preferred_test")

            # 1. Pre-execution validation
            pre_audit = PreExecutionValidator.validate(
                data,
                task_type="hypothesis_testing",
                target=group,
                feature_columns=features or ([feature] if feature else None),
                agent_name=self.name,
            )
            if not pre_audit.is_valid:
                err = pre_audit.error
                return self._error(
                    message=err.user_message if err else "Hypothesis testing pre-execution validation failed.",
                    code=err.code if err else "VALIDATION_FAILURE",
                    category=err.category if err else ErrorCategory.DATA_INVALID,
                    details=err.technical_details if err else {},
                )

            # 2. Run canonical HypothesisTestingEngine
            engine = HypothesisTestingEngine(alpha=alpha if alpha is not None and 0.0 < alpha < 1.0 else 0.05)
            result = engine.test(
                data=data,
                feature=feature,
                group=group,
                feature_2=feature_2,
                features=features,
                target=group,
                alpha=alpha,
                paired=paired,
                preferred_test=preferred_test,
            )

            if "error" in result:
                return self._error(
                    message=result["error"],
                    code="HYPOTHESIS_TESTING_FAILED",
                    category=result.get("category", ErrorCategory.COMPUTATION),
                    details=result,
                    output=result,
                )

            hypotheses = result.get("hypotheses", [])
            summary = result.get("summary", {})
            n_hyp = len(hypotheses)
            orig_rows = summary.get("original_rows", len(data) if isinstance(data, pd.DataFrame) else 0)

            # 3. Canonical Evidence Generation (ClaimType.OBSERVATION)
            evidence_list: List[Evidence] = []
            for i, hyp in enumerate(hypotheses[:10]):
                ev_cols = [hyp["variable_x"]]
                if hyp.get("variable_group"):
                    ev_cols.append(hyp["variable_group"])
                elif hyp.get("variable_y") and hyp["variable_y"] != hyp["variable_x"]:
                    ev_cols.append(hyp["variable_y"])

                evidence_list.append(
                    self.make_evidence(
                        method=f"hypothesis_testing.{hyp['test_method']}",
                        data_ref={
                            "variable_x": hyp["variable_x"],
                            "variable_group": hyp.get("variable_group") or hyp.get("variable_y"),
                            "test_name": hyp["test_name"],
                            "test_statistic": hyp["test_statistic"],
                            "p_value": hyp["p_value"],
                            "adjusted_p_value": hyp["adjusted_p_value"],
                            "effect_size": hyp["effect_size"],
                            "effect_size_type": hyp["effect_size_type"],
                            "statistical_significance": hyp["statistical_significance"],
                            "practical_significance": hyp["practical_significance"],
                            "valid_rows": hyp.get("row_accounting", {}).get("valid_rows", orig_rows),
                        },
                        confidence=0.95,
                        claim_type=ClaimType.OBSERVATION,
                        raw_value=hyp["p_value"],
                    )
                )

            # 4. Confidence Calculation
            n_valid_obs = orig_rows
            assump_pass_ratio = 1.0
            if hypotheses:
                n_valid_obs = max(h.get("row_accounting", {}).get("valid_rows", 0) for h in hypotheses)
                all_assump = [a["status"] == "passed" for h in hypotheses for a in h.get("assumptions", [])]
                if all_assump:
                    assump_pass_ratio = sum(all_assump) / len(all_assump)

            conf_rep = ConfidenceCalculator.calculate_hypothesis_testing_confidence(
                n_observations=n_valid_obs,
                n_tests=n_hyp,
                missing_rate=0.0,
                assumptions_passed_ratio=assump_pass_ratio,
                test_suitability=0.92,
            )

            raw_res = self._finish(
                result,
                evidence=evidence_list,
                confidence=conf_rep.confidence,
                model_used="HypothesisTestingEngine",
            )

            # 5. Result Validation & Invariant Repair
            repaired_res, _ = ResultValidator().repair(raw_res, context={"data": data})
            return repaired_res

        except Exception as e:
            return self._error(str(e), category=ErrorCategory.COMPUTATION)