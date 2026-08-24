"""Multi-Modal Computer Vision Engine & Spatial Feature Extractor.

Extracts spatial, structural, texture, and color embeddings from image collections,
video keyframes, and 2D/3D matrix arrays. Converts image datasets into structured tabular
DataFrames for seamless AutoML classification, clustering, and anomaly detection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import io
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Safe optional imports for vision libraries
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@dataclass
class ImageFeatureVector:
    """Extracted feature vector and quality metadata for a single image."""
    image_id: str
    width: int
    height: int
    channels: int
    blur_score: float
    brightness: float
    contrast: float
    color_moments: Dict[str, float]
    spatial_features: List[float]
    label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "blur_score": round(float(self.blur_score), 2),
            "brightness": round(float(self.brightness), 2),
            "contrast": round(float(self.contrast), 2),
            "color_moments": {k: round(float(v), 3) for k, v in self.color_moments.items()},
            "feature_dim": len(self.spatial_features),
            "label": self.label,
        }


@dataclass
class VisionExtractionReport:
    """Summary report of a batch computer vision feature extraction job."""
    total_images: int
    feature_dimension: int
    labels_found: List[str]
    blurry_images_count: int
    feature_dataframe: pd.DataFrame
    duration_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_images": self.total_images,
            "feature_dimension": self.feature_dimension,
            "labels_found": self.labels_found,
            "blurry_images_count": self.blurry_images_count,
            "sample_records": self.feature_dataframe.head(10).to_dict(orient="records"),
            "duration_ms": round(float(self.duration_ms), 3),
        }


class ComputerVisionFeatureEngine:
    """Engine for feature extraction, spatial filtering, and quality auditing on image datasets."""

    # 3x3 Sobel and Laplacian edge convolution kernels
    SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

    def __init__(self, target_size: Tuple[int, int] = (32, 32)):
        self.target_size = target_size

    def extract_features_from_array(
        self,
        image_array: np.ndarray,
        image_id: str = "img_0",
        label: Optional[str] = None,
    ) -> ImageFeatureVector:
        """
        Extract spatial gradients, texture, color moments, and quality scores from a numpy image.
        
        Args:
            image_array: 2D (H, W) or 3D (H, W, C) numpy array with values 0..255 or 0..1.
            image_id: Identifier string for image.
            label: Optional class label.
        """
        arr = np.asarray(image_array, dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0

        if arr.ndim == 2:
            h, w = arr.shape
            c = 1
            gray = arr
            rgb = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3:
            h, w, c = arr.shape
            rgb = arr if c >= 3 else np.pad(arr, ((0, 0), (0, 0), (0, 3 - c)), mode="edge")
            gray = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
        else:
            raise ValueError(f"Expected 2D or 3D image array, got shape {arr.shape}")

        # 1. Image Quality Metrics (Brightness, Contrast, Blur)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        # Laplacian variance for blur detection (lower = more blurry)
        lap_grad = self._convolve2d(gray, self.LAPLACIAN)
        blur_score = float(np.var(lap_grad) * 1000.0)

        # 2. Color Moments (Mean, Std, Skewness for R, G, B)
        color_moments = {
            "r_mean": float(np.mean(rgb[:, :, 0])),
            "r_std": float(np.std(rgb[:, :, 0])),
            "g_mean": float(np.mean(rgb[:, :, 1])),
            "g_std": float(np.std(rgb[:, :, 1])),
            "b_mean": float(np.mean(rgb[:, :, 2])),
            "b_std": float(np.std(rgb[:, :, 2])),
        }

        # 3. Spatial Gradient Features (Sobel X and Y)
        sobel_x_grad = self._convolve2d(gray, self.SOBEL_X)
        sobel_y_grad = self._convolve2d(gray, self.SOBEL_Y)
        grad_mag = np.sqrt(sobel_x_grad**2 + sobel_y_grad**2)

        # 4. Spatial Grid Pooling (4x4 cell pooling into 16 gradient features)
        grid_features = self._spatial_grid_pool(grad_mag, grid_size=(4, 4))

        # 5. Fast Downsampled Intensity Vector
        resized_gray = self._resize_gray(gray, self.target_size)
        intensity_features = resized_gray.flatten().tolist()

        combined_features = list(color_moments.values()) + grid_features + intensity_features

        return ImageFeatureVector(
            image_id=image_id,
            width=w,
            height=h,
            channels=c,
            blur_score=blur_score,
            brightness=brightness,
            contrast=contrast,
            color_moments=color_moments,
            spatial_features=combined_features,
            label=label,
        )

    def extract_batch_to_dataframe(
        self,
        images: Union[List[np.ndarray], Dict[str, np.ndarray]],
        labels: Optional[List[str]] = None,
        blur_threshold: float = 0.5,
    ) -> VisionExtractionReport:
        """
        Process a batch of images into a unified feature DataFrame suitable for ML / AutoML.
        """
        start_t = time.time()
        img_list: List[Tuple[str, np.ndarray, Optional[str]]] = []

        if isinstance(images, dict):
            for i, (k, arr) in enumerate(images.items()):
                lbl = labels[i] if labels and i < len(labels) else None
                img_list.append((k, arr, lbl))
        else:
            for i, arr in enumerate(images):
                lbl = labels[i] if labels and i < len(labels) else None
                img_list.append((f"img_{i:04d}", arr, lbl))

        feature_rows: List[Dict[str, Any]] = []
        blurry_count = 0
        all_labels: Set[str] = set()

        for img_id, arr, lbl in img_list:
            feat_vec = self.extract_features_from_array(arr, image_id=img_id, label=lbl)

            if feat_vec.blur_score < blur_threshold:
                blurry_count += 1

            row = {
                "image_id": feat_vec.image_id,
                "blur_score": feat_vec.blur_score,
                "brightness": feat_vec.brightness,
                "contrast": feat_vec.contrast,
            }
            row.update(feat_vec.color_moments)

            for idx, val in enumerate(feat_vec.spatial_features):
                row[f"vfeat_{idx}"] = val

            if feat_vec.label is not None:
                row["label"] = feat_vec.label
                all_labels.add(str(feat_vec.label))

            feature_rows.append(row)

        df = pd.DataFrame(feature_rows)
        feat_dim = len(feature_rows[0]) - 4 if feature_rows else 0
        duration = (time.time() - start_t) * 1000

        return VisionExtractionReport(
            total_images=len(img_list),
            feature_dimension=feat_dim,
            labels_found=sorted(list(all_labels)),
            blurry_images_count=blurry_count,
            feature_dataframe=df,
            duration_ms=duration,
        )

    # ------------------------------------------------------------------
    # Vectorized Spatial Helpers
    # ------------------------------------------------------------------
    def _convolve2d(self, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """2D convolution with edge padding."""
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")

        # Vectorized 2D window stride convolution
        out_h, out_w = img.shape
        out = np.zeros((out_h, out_w), dtype=np.float32)

        for i in range(kh):
            for j in range(kw):
                out += kernel[i, j] * padded[i : i + out_h, j : j + out_w]

        return out

    def _spatial_grid_pool(self, mag: np.ndarray, grid_size: Tuple[int, int] = (4, 4)) -> List[float]:
        """Divide image into grid cells and pool mean gradient magnitudes."""
        gh, gw = grid_size
        h, w = mag.shape
        step_h, step_w = max(1, h // gh), max(1, w // gw)
        features = []

        for i in range(gh):
            for j in range(gw):
                cell = mag[i * step_h : (i + 1) * step_h, j * step_w : (j + 1) * step_w]
                features.append(float(np.mean(cell)) if cell.size > 0 else 0.0)

        return features

    def _resize_gray(self, gray: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Nearest-neighbor / block downsampling of 2D matrix."""
        th, tw = target_size
        h, w = gray.shape
        idx_h = (np.linspace(0, h - 1, th)).astype(int)
        idx_w = (np.linspace(0, w - 1, tw)).astype(int)
        return gray[np.ix_(idx_h, idx_w)]


# Global singleton vision engine
global_vision_engine = ComputerVisionFeatureEngine()

