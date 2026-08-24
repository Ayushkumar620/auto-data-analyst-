"""Convolutional Neural Network (CNN) Agent - Orchestrates Spatial, Image, and Signal Learning."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from agent.base import BaseAgent
from agent.schemas import AgentResult, ClaimType, ErrorCategory, Evidence
from backend.app.ml.cnn_engine import CNNEngine, CNNHyperparameters, CNNLayerConfig, CNNTrainingResult


class CNNAgent(BaseAgent):
    """
    Autonomous Convolutional Neural Network (CNN) Agent.
    Specialized in processing 2D images, spatial grid matrices, and 1D sensor/signal series
    transformed into spectrograms. Employs 2D convolutions and spatial pooling, benchmarked
    directly against non-convolutional baselines to measure spatial inductive bias gain.
    """
    name = "CNN Agent"
    role = "computer_vision_engineer"
    description = "Trains Convolutional Neural Networks (CNN) for image, spatial grid, and signal data."

    def __init__(self, data=None):
        super().__init__(data=data)
        self.engine = CNNEngine()

    def run(self, task: Dict[str, Any]) -> AgentResult:
        """
        Execute CNN model training on spatial/image/signal inputs.
        Task parameters:
            - data: pd.DataFrame, np.ndarray (optional, defaults to self.data)
            - target: str or array-like (target column / labels)
            - spatial_shape: tuple[int, int] (e.g. (28, 28), (32, 32))
            - is_signal: bool (if True, converts 1D sensor signals into 2D spectrograms)
            - filters: list[int] (e.g. [16, 32])
            - epochs: int (epochs to train)
            - compare_with_baseline: bool (default: True)
        """
        self._start()
        data = task.get("data") if task.get("data") is not None else self.data

        if data is None:
            return self._error("No data provided for CNN training.", category=ErrorCategory.INPUT_VALIDATION)

        target = task.get("target")
        spatial_shape = task.get("spatial_shape")
        is_signal = bool(task.get("is_signal", False))
        epochs = int(task.get("epochs", 80))
        filters_list = task.get("filters", [16, 32])

        # If signal mode, transform 1D signals to 2D spectrogram grids
        if is_signal and isinstance(data, np.ndarray) and data.ndim == 2:
            try:
                data, spec_shape = self.engine.signal_to_spectrogram(data)
                spatial_shape = spec_shape
            except Exception as exc:
                return self._error(f"Spectrogram transformation failed: {str(exc)}", category=ErrorCategory.COMPUTATION)

        # Build CNN Hyperparameters
        conv_blocks = [
            CNNLayerConfig(filters=f_cnt, kernel_size=3, pool_size=2)
            for f_cnt in filters_list
        ]
        hyperparams = CNNHyperparameters(
            conv_blocks=conv_blocks,
            epochs=epochs,
        )

        try:
            result_obj: CNNTrainingResult = self.engine.train_and_evaluate(
                data=data,
                target=target,
                spatial_shape=spatial_shape,
                hyperparams=hyperparams,
                compare_with_flat_baseline=bool(task.get("compare_with_baseline", True)),
            )
        except Exception as exc:
            return self._error(f"CNN execution failed: {str(exc)}", category=ErrorCategory.COMPUTATION)

        # Build traceable Evidence
        evidence_list: List[Evidence] = []

        # 1. Primary CNN metric evidence
        evidence_list.append(
            self.make_evidence(
                method="cnn_convolutional_training",
                data_ref={
                    "modality": result_obj.data_modality,
                    "spatial_shape": list(result_obj.input_spatial_shape),
                    "architecture": result_obj.architecture_summary,
                    "primary_metric": result_obj.primary_metric_name,
                    "metric_value": result_obj.primary_metric_value,
                },
                confidence=0.95,
                claim_type=ClaimType.FACT,
            )
        )

        # 2. Spatial Inductive Bias Gain evidence
        if result_obj.comparison_with_flat_baseline:
            comp = result_obj.comparison_with_flat_baseline
            evidence_list.append(
                self.make_evidence(
                    method="cnn_vs_flat_baseline_comparison",
                    data_ref={
                        "cnn_score": comp.get("cnn_score"),
                        "flat_baseline_score": comp.get("flat_baseline_score"),
                        "spatial_gain": comp.get("spatial_inductive_bias_gain"),
                    },
                    confidence=0.90,
                    claim_type=ClaimType.OBSERVATION,
                )
            )

        output = {
            "problem_type": result_obj.problem_type.value,
            "data_modality": result_obj.data_modality,
            "spatial_shape": list(result_obj.input_spatial_shape),
            "architecture_summary": result_obj.architecture_summary,
            "hyperparameters": result_obj.hyperparameters,
            "metrics": result_obj.metrics,
            "primary_metric_name": result_obj.primary_metric_name,
            "primary_metric_value": result_obj.primary_metric_value,
            "loss_curve": result_obj.loss_curve,
            "comparison_with_flat_baseline": result_obj.comparison_with_flat_baseline,
            "spatial_inductive_bias_gain": result_obj.spatial_gain,
        }

        return self._finish(
            result=output,
            evidence=evidence_list,
            confidence=0.95,
            metadata={"architecture": result_obj.architecture_summary, "modality": result_obj.data_modality},
        )
