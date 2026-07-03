"""Percentile-threshold anomaly detector."""

import collections

import numpy as np

from trainscope.core.detectors.base import AnomalyDetector


class PercentileDetector(AnomalyDetector):
    """Flag values outside a percentile band of recent losses.

    The band is defined by ``lower`` and ``upper`` percentiles of the observed
    history. When a value falls outside the band, a normalized score is
    returned.

    Parameters
    ----------
    lower:
        Lower percentile bound (0--100).
    upper:
        Upper percentile bound (0--100).
    window:
        Maximum number of observations retained in the history.
    min_observations:
        Number of observations required before the detector starts reporting.
    """

    def __init__(
        self,
        lower: float = 1.0,
        upper: float = 99.0,
        window: int | None = 200,
        min_observations: int = 30,
    ):
        if not 0.0 <= lower <= 100.0 or not 0.0 <= upper <= 100.0 or lower >= upper:
            raise ValueError("lower and upper must be valid percentiles with lower < upper")
        self.lower = lower
        self.upper = upper
        self.window = window
        self.min_observations = min_observations
        self._history: collections.deque[float] = collections.deque()

    @property
    def warmup(self) -> bool:
        return len(self._history) < self.min_observations

    def _trim(self) -> None:
        if self.window is not None and len(self._history) > self.window:
            self._history.popleft()

    def _history_array(self) -> np.ndarray:
        return np.fromiter(self._history, dtype=float, count=len(self._history))

    def update(self, loss: float) -> float | None:
        if len(self._history) < self.min_observations:
            self._history.append(loss)
            self._trim()
            return None

        history = self._history_array()
        lo = float(np.percentile(history, self.lower))
        hi = float(np.percentile(history, self.upper))
        median = float(np.median(history))

        self._history.append(loss)
        self._trim()

        if loss < lo or loss > hi:
            iqr = max(hi - lo, 1e-12)
            return float((loss - median) / iqr)
        return None

    def get_band(self) -> tuple[float, float] | None:
        """Return the current (lower, upper) percentile band, or None in warmup."""
        if self.warmup:
            return None
        history = self._history_array()
        return float(np.percentile(history, self.lower)), float(np.percentile(history, self.upper))
