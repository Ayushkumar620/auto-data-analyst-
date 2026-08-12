from __future__ import annotations
import numpy as np

class ModelEvaluator:
    def evaluate(self, actual: np.ndarray, predictions: dict[str, np.ndarray]) -> tuple[str, dict[str, dict[str, float | None]]]:
        results = {}
        for name, predicted in predictions.items():
            errors = actual - predicted
            mape = np.mean(np.abs(errors / actual)) * 100 if np.all(actual != 0) else None
            results[name] = {"mae": round(float(np.mean(np.abs(errors))), 6), "rmse": round(float(np.sqrt(np.mean(errors ** 2))), 6), "mape": None if mape is None else round(float(mape), 6)}
        return min(results, key=lambda name: results[name]["mae"]), results
