"""Model Selection Agent - Orchestrates multi-algorithm ML benchmarking, ranking, and explanation."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd

from agent.base import BaseAgent
from agent.schemas import AgentResult, ClaimType, ErrorCategory, Evidence
from backend.app.ml.model_selection import MLModelComparisonEngine, ModelComparisonReport, ProblemType


class ModelSelectionAgent(BaseAgent):
    """
    Autonomous Machine Learning Model Selection Agent.
    Evaluates dataset structure, benchmarks candidate algorithms across model families,
    ranks them on standardized cross-validation & holdout metrics, and explains the winner.
    """
    name = "Model Selection Agent"
    role = "ml_engineer"
    description = "Benchmarks candidate ML algorithms, selects the optimal model, and provides rationale."

    def __init__(self, data=None):
        super().__init__(data=data)
        self.engine = MLModelComparisonEngine()

    def run(self, task: Dict[str, Any]) -> AgentResult:
        """
        Execute ML model selection on the target dataset.
        Task parameters:
            - data: pd.DataFrame or dict (optional, defaults to self.data)
            - target: str (target column name, optional - auto-detected if omitted)
            - features: list[str] (optional list of feature columns)
            - cv_folds: int (number of CV folds, default: 5)
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
            return self._error("Missing required DataFrame input.", category=ErrorCategory.INPUT_VALIDATION)

        if len(df_target) < 10:
            return self._error(f"Dataset requires at least 10 samples for ML benchmarking. Found {len(df_target)}.", category=ErrorCategory.DATA_QUALITY)

        # Detect target column if not provided
        target = task.get("target")
        if not target or target not in df_target.columns:
            # Fall back to last numeric column or first categorical
            num_cols = df_target.select_dtypes(include=["number"]).columns
            if len(num_cols) > 0:
                target = num_cols[-1]
            else:
                target = df_target.columns[-1]

        features = task.get("features")
        cv_folds = int(task.get("cv_folds", 5))

        try:
            report: ModelComparisonReport = self.engine.benchmark_models(
                dataframe=df_target,
                target_column=target,
                feature_columns=features,
                cv_folds=cv_folds,
            )
        except Exception as exc:
            return self._error(f"Model benchmarking failed: {str(exc)}", category=ErrorCategory.COMPUTATION)

        # Build traceable Evidence
        evidence_list: List[Evidence] = []
        best = report.best_model

        # 1. Best model claim
        evidence_list.append(
            self.make_evidence(
                method="cross_validation_benchmark",
                data_ref={
                    "target": target,
                    "model_name": best.model_name,
                    "problem_type": report.problem_type.value,
                    "primary_metric": best.primary_metric_name,
                    "metric_value": best.primary_metric_value,
                    "cv_mean": best.cv_mean,
                    "cv_std": best.cv_std,
                },
                confidence=0.95,
                claim_type=ClaimType.FACT,
            )
        )

        # 2. Top feature importances evidence
        if best.feature_importances:
            sorted_feats = sorted(best.feature_importances.items(), key=lambda x: x[1], reverse=True)[:5]
            evidence_list.append(
                self.make_evidence(
                    method="feature_importance_attribution",
                    data_ref={
                        "target": target,
                        "model_name": best.model_name,
                        "top_features": {k: round(v, 4) for k, v in sorted_feats},
                    },
                    confidence=0.90,
                    claim_type=ClaimType.OBSERVATION,
                )
            )

        output = {
            "problem_type": report.problem_type.value,
            "target": target,
            "best_model": best.to_dict(),
            "selection_rationale": report.selection_rationale,
            "leaderboard": report.leaderboard,
            "candidate_evaluations": [e.to_dict() for e in report.candidate_evaluations],
            "dataset_characteristics": report.dataset_characteristics,
            "models_evaluated_count": len(report.candidate_evaluations),
        }

        return self._finish(
            result=output,
            evidence=evidence_list,
            confidence=0.95,
            metadata={"problem_type": report.problem_type.value, "target": target},
        )

