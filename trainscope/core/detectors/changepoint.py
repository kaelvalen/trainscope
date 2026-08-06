"""Change-point anomaly detector with optional ``ruptures`` backend."""

import collections

import numpy as np

from trainscope.core.detectors.base import AnomalyDetector

try:
    import ruptures as rpt
except Exception:  # pragma: no cover
    rpt = None


class ChangePointDetector(AnomalyDetector):
    """Detect distributional shifts in the loss stream.

    If ``ruptures`` is installed, a PELT model is used to detect change points
    in the recent loss history. Otherwise a lightweight CUSUM implementation
    is used as a fallback.

    Parameters
    ----------
    threshold:
        CUSUM multiplier or ``ruptures`` penalty scale.
    window:
        Number of recent observations to consider.
    min_observations:
        Number of observations required before reporting.
    """

    def __init__(
        self,
        threshold: float = 3.5,
        window: int = 200,
        min_observations: int = 30,
    ):
        self.threshold = threshold
        self.window = window
        self.min_observations = min_observations
        self._history: collections.deque[float] = collections.deque()
        self._positive_cusum = 0.0
        self._negative_cusum = 0.0

    @property
    def warmup(self) -> bool:
        return len(self._history) < self.min_observations

    def _trim(self) -> None:
        if len(self._history) > self.window:
            self._history.popleft()

    def _history_array(self) -> np.ndarray:
        return np.fromiter(self._history, dtype=float, count=len(self._history))

    def update(self, loss: float) -> float | None:
        if len(self._history) < self.min_observations:
            self._history.append(loss)
            self._trim()
            return None

        history = self._history_array()
        mean = float(np.mean(history))
        std = float(np.std(history)) if len(history) > 1 else 0.0

        if rpt is not None and len(history) >= 2 * self.min_observations:
            signal = history.reshape(-1, 1)
            try:
                algo = rpt.Pelt(model="l2", min_size=max(2, self.min_observations // 5)).fit(signal)
                change_points = algo.predict(pen=self.threshold**2)
                # If the most recent point is flagged as a change, report it.
                if len(change_points) > 1 and change_points[-2] == len(history):
                    self._history.append(loss)
                    self._trim()
                    deviation = (loss - mean) / max(std, 1e-12)
                    return float(deviation)
            except Exception:
                pass

        # Lightweight CUSUM fallback.
        if std > 0.0:
            normalized = (loss - mean) / std
            self._positive_cusum = max(0.0, self._positive_cusum + normalized - self.threshold)
            self._negative_cusum = min(0.0, self._negative_cusum + normalized + self.threshold)

            self._history.append(loss)
            self._trim()

            if self._positive_cusum > self.threshold or abs(self._negative_cusum) > self.threshold:
                score = (
                    self._positive_cusum
                    if self._positive_cusum > abs(self._negative_cusum)
                    else self._negative_cusum
                )
                self._positive_cusum = 0.0
                self._negative_cusum = 0.0
                return float(score)
        else:
            self._history.append(loss)
            self._trim()

        return None
