from __future__ import annotations
import numpy as np

class CandidateModels:
    @staticmethod
    def naive(train: np.ndarray, horizon: int) -> np.ndarray: return np.repeat(train[-1], horizon)
    @staticmethod
    def moving_average(train: np.ndarray, horizon: int) -> np.ndarray: return np.repeat(np.mean(train[-min(3, len(train)):]), horizon)
    @staticmethod
    def exponential_smoothing(train: np.ndarray, horizon: int, alpha: float = .4) -> np.ndarray:
        level = float(train[0])
        for value in train[1:]: level = alpha * float(value) + (1 - alpha) * level
        return np.repeat(level, horizon)
    def candidates(self, train: np.ndarray, horizon: int) -> dict[str, np.ndarray]:
        return {"naive": self.naive(train, horizon), "moving_average": self.moving_average(train, horizon), "exponential_smoothing": self.exponential_smoothing(train, horizon)}
