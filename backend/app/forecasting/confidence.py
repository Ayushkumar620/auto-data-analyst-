from __future__ import annotations
import numpy as np

class ConfidenceIntervals:
    def build(self, forecast: np.ndarray, residuals: np.ndarray) -> list[tuple[float, float]]:
        spread = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else max(abs(float(forecast[0])) * .1, 1.0)
        return [(max(0.0, float(value) - 1.96 * spread), float(value) + 1.96 * spread) for value in forecast]
