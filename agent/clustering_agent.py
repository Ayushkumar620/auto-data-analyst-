"""
Universal Clustering & Segmentation Agent.

Executes autonomous unsupervised clustering on arbitrary datasets using
the canonical ClusteringEngine and returns a standardized AgentResult.
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


class ClusteringAgent(BaseAgent):
    """
    Autonomous Clustering & Segmentation Agent.
    Identifies optimal natural data groupings and explains segment characteristics.
    """

    name = "Clustering Agent"
    description = "Discovers natural clusters and customer segments using benchmarked unsupervised algorithms."
    role = "clustering"

    def run(self, task: Dict[str, Any]) -> AgentResult:
        self._start()
        try:
            from agent.pre_execution_validator import PreExecutionValidator
            from agent.confidence_calculator import ConfidenceCalculator
            from agent.result_validator import ResultValidator
            from agent.clustering_engine import ClusteringEngine

            data = task.get("data")
            features = task.get("features") or task.get("feature_columns")
            n_clusters = task.get("n_clusters") or task.get("k", "auto")
            method = task.get("method")
            random_state = task.get("random_state", 42)

            pre_audit = PreExecutionValidator.validate(
                data,
                task_type="clustering",
                feature_columns=features,
                agent_name=self.name,
            )
            if not pre_audit.is_valid:
                err = pre_audit.error
                return self._error(
                    message=err.user_message if err else "Clustering pre-validation failed.",
                    code=err.code if err else "VALIDATION_FAILURE",
                    category=err.category if err else ErrorCategory.DATA_INVALID,
                    details=err.technical_details if err else {},
                )

            engine = ClusteringEngine(random_state=random_state)
            result = engine.cluster(
                data=data,
                features=features,
                n_clusters=n_clusters,
                method=method,
                random_state=random_state,
            )

            if "error" in result:
                return self._error(
                    message=result["error"],
                    code="CLUSTERING_FAILED",
                    category=result.get("category", ErrorCategory.MODEL_FAILURE),
                    details=result,
                    output=result,
                )

            winning_model = result.get("selected_model", "kmeans")
            winning_family = result.get("model_family", "centroid")
            n_rows = result.get("rows_analyzed", 0)
            k_clusters = result.get("cluster_count", 2)
            metrics = result.get("validation_metrics", {})
            sil = metrics.get("silhouette_score", 0.5)
            db = metrics.get("davies_bouldin_score", 1.0)
            noise_ratio = result.get("noise_ratio", 0.0)

            evidence = [
                self.make_evidence(
                    method=f"clustering.{winning_family}.{winning_model}",
                    data_ref={
                        "model": winning_model,
                        "cluster_count": k_clusters,
                        "rows_analyzed": n_rows,
                        "features_used": result.get("features_used", []),
                        "cluster_sizes": result.get("cluster_sizes", {}),
                    },
                    confidence=0.85,
                    claim_type=ClaimType.OBSERVATION,
                    raw_value={
                        "silhouette_score": sil,
                        "davies_bouldin_score": db,
                        "cluster_count": k_clusters,
                    },
                )
            ]

            conf_rep = ConfidenceCalculator.calculate_clustering_confidence(
                silhouette_score=sil,
                davies_bouldin_score=db,
                n_samples=n_rows,
                n_features=len(result.get("features_used", [])) or 2,
                k_clusters=k_clusters,
                noise_ratio=noise_ratio,
            )

            raw_res = self._finish(
                result,
                evidence=evidence,
                confidence=conf_rep.confidence,
                model_used=winning_model,
            )
            repaired_res, _ = ResultValidator().repair(raw_res, context={"data": data})
            return repaired_res
        except Exception as e:
            return self._error(str(e), category=ErrorCategory.COMPUTATION)
