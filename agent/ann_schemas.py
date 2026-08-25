"""
Pydantic Schemas for Artificial Neural Network (ANN) Engine.

Defines:
- ANNConfig: Structural topology, activation, optimization, and training hyperparameter specification
- ANNTrainerResult: Detailed training metrics, loss curve, validation history, and artifact metadata
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field, field_validator


class ANNConfig(BaseModel):
    """Configurable architecture and optimization specification for Tabular ANN/MLP."""
    input_dim: Optional[int] = None
    hidden_layers: Tuple[int, ...] = (128, 64)
    activation: str = "relu"  # relu, tanh, logistic, identity
    output_activation: Optional[str] = None
    dropout: float = Field(default=0.0, ge=0.0, le=0.8)
    learning_rate: float = Field(default=0.001, gt=0.0)
    optimizer: str = "adam"  # adam, sgd, lbfgs
    loss: Optional[str] = None
    batch_size: Union[int, str] = "auto"
    epochs: int = Field(default=200, ge=1, le=2000)
    early_stopping: bool = True
    patience: int = Field(default=10, ge=1)
    random_seed: int = 42
    task_type: str = "regression"  # regression, binary_classification, multiclass_classification
    alpha: float = Field(default=0.0001, ge=0.0)  # L2 regularization

    @field_validator("hidden_layers", mode="before")
    @classmethod
    def validate_layers(cls, v: Any) -> Tuple[int, ...]:
        if isinstance(v, list):
            return tuple(int(x) for x in v)
        if isinstance(v, tuple):
            return tuple(int(x) for x in v)
        return (128, 64)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_layers": list(self.hidden_layers),
            "activation": self.activation,
            "output_activation": self.output_activation,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "loss": self.loss,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
            "random_seed": self.random_seed,
            "task_type": self.task_type,
            "alpha": self.alpha,
        }


def auto_select_ann_architecture(
    n_samples: int,
    n_features: int,
    task_type: str = "regression",
    random_seed: int = 42,
) -> ANNConfig:
    """
    Intelligently select sensible ANN architecture and solver based on dataset size and feature count.
    Avoids oversized models for small datasets while enabling deeper networks for larger data.
    """
    # 1. Small dataset (N < 100 or D <= 5) -> Compact topology with L-BFGS or Adam
    if n_samples < 100 or n_features <= 5:
        layers = (32, 16)
        solver = "lbfgs" if n_samples < 150 else "adam"
        early_stop = False if n_samples < 50 else True
    # 2. Medium dataset (100 <= N < 1000 or 5 < D <= 20) -> Standard dual layer
    elif n_samples < 1000 or n_features <= 20:
        layers = (128, 64)
        solver = "adam"
        early_stop = True
    # 3. Large / complex dataset (N >= 1000 or D > 20) -> Deep 3-layer network
    else:
        layers = (256, 128, 64)
        solver = "adam"
        early_stop = True

    return ANNConfig(
        input_dim=n_features,
        hidden_layers=layers,
        activation="relu",
        optimizer=solver,
        epochs=200 if n_samples >= 100 else 100,
        early_stopping=early_stop,
        patience=10,
        random_seed=random_seed,
        task_type=task_type,
        alpha=0.0001 if n_samples >= 100 else 0.001,
    )
