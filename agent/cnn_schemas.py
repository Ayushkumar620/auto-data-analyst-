"""
Pydantic Schemas for Convolutional Neural Network (CNN) Engine.

Defines:
- CNNLayerConfig: Individual Conv2D + Pooling layer configuration
- CNNConfig: Full convolutional architecture, optimizer, and training specification
- ModalityType: Recognized dataset modalities (tabular, image, signal, spatial_grid, text, unsupported)
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field, field_validator


class ModalityType(str, Enum):
    """Recognized dataset modalities."""
    TABULAR = "tabular"
    IMAGE = "image"
    SIGNAL = "signal"
    SPATIAL_GRID = "spatial_grid"
    TEXT = "text"
    UNSUPPORTED = "unsupported"


class CNNLayerConfig(BaseModel):
    """Configuration for an individual Convolutional + Pooling layer block."""
    filters: int = Field(default=32, ge=1, le=512)
    kernel_size: int = Field(default=3, ge=1, le=11)
    pool_size: int = Field(default=2, ge=1, le=8)
    activation: str = "relu"  # relu, tanh, gelu
    stride: int = Field(default=1, ge=1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "pool_size": self.pool_size,
            "activation": self.activation,
            "stride": self.stride,
        }


class CNNConfig(BaseModel):
    """Configurable architecture and optimization specification for CNN models."""
    input_shape: Optional[Tuple[int, ...]] = None  # (C, H, W) or (H, W)
    num_classes: int = Field(default=2, ge=1)
    conv_blocks: List[CNNLayerConfig] = Field(
        default_factory=lambda: [
            CNNLayerConfig(filters=16, kernel_size=3, pool_size=2),
            CNNLayerConfig(filters=32, kernel_size=3, pool_size=2),
        ]
    )
    dense_units: Tuple[int, ...] = (128, 64)
    activation: str = "relu"
    dropout: float = Field(default=0.25, ge=0.0, le=0.8)
    learning_rate: float = Field(default=0.001, gt=0.0)
    optimizer: str = "adam"  # adam, sgd, rmsprop, lbfgs
    loss: Optional[str] = None
    batch_size: Union[int, str] = 32
    epochs: int = Field(default=100, ge=1, le=1000)
    early_stopping: bool = True
    patience: int = Field(default=10, ge=1)
    random_seed: int = 42
    task_type: str = "image_classification"  # image_classification, spatial_grid, signal_classification
    augmentation: bool = False

    @field_validator("dense_units", mode="before")
    @classmethod
    def validate_dense(cls, v: Any) -> Tuple[int, ...]:
        if isinstance(v, list):
            return tuple(int(x) for x in v)
        if isinstance(v, tuple):
            return tuple(int(x) for x in v)
        return (128, 64)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_shape": list(self.input_shape) if self.input_shape else None,
            "num_classes": self.num_classes,
            "conv_blocks": [b.to_dict() for b in self.conv_blocks],
            "dense_units": list(self.dense_units),
            "activation": self.activation,
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
            "augmentation": self.augmentation,
        }


def auto_select_cnn_architecture(
    spatial_shape: Tuple[int, int],
    num_classes: int = 2,
    task_type: str = "image_classification",
    random_seed: int = 42,
) -> CNNConfig:
    """
    Construct lightweight, resource-conscious CNN configuration suited for local CPU/GPU execution.
    """
    H, W = spatial_shape
    if H <= 16 or W <= 16:
        # Small spatial matrix (e.g. 8x8 or 16x16)
        blocks = [
            CNNLayerConfig(filters=16, kernel_size=3, pool_size=2),
            CNNLayerConfig(filters=32, kernel_size=3, pool_size=2),
        ]
        dense = (64, 32)
    elif H <= 64 or W <= 64:
        # Medium image (e.g. 28x28, 32x32, 64x64)
        blocks = [
            CNNLayerConfig(filters=16, kernel_size=3, pool_size=2),
            CNNLayerConfig(filters=32, kernel_size=3, pool_size=2),
            CNNLayerConfig(filters=64, kernel_size=3, pool_size=2),
        ]
        dense = (128, 64)
    else:
        # Larger image (e.g. 128x128, 224x224)
        blocks = [
            CNNLayerConfig(filters=32, kernel_size=3, pool_size=2),
            CNNLayerConfig(filters=64, kernel_size=3, pool_size=2),
            CNNLayerConfig(filters=128, kernel_size=3, pool_size=2),
        ]
        dense = (256, 128)

    return CNNConfig(
        input_shape=(1, H, W),
        num_classes=num_classes,
        conv_blocks=blocks,
        dense_units=dense,
        epochs=80 if (H <= 32 and W <= 32) else 50,
        random_seed=random_seed,
        task_type=task_type,
    )
