"""Z-score based anomaly detector with optional robust median/MAD mode."""

import collections
import math
from statistics import median

from trainscope.core.detectors.base import AnomalyDetector


class ZScoreDetector(AnomalyDetector):
    """Online z-score spike detector using Welford's algorithm.

    Maintains a rolling mean and variance over the most recent ``window`` loss
    values. The current loss is compared against the baseline computed *before*
    the value is incorporated, so a spike cannot inflate its own denominator.

    Parameters
    ----------
    threshold:
        Absolute z-score above which a value is reported as a spike.
    window:
        Maximum number of observations retained in the baseline. Older values
        are forgotten. If ``None`` the window is unbounded.
    min_observations:
        Number of observations required before the detector starts reporting.
    robust:
        If True, use median and MAD instead of mean and standard deviation.
    """

    def __init__(
        self,
        threshold: float = 3.5,
        window: int | None = 200,
        min_observations: int = 30,
        robust: bool = False,
    ):
        self.threshold = threshold
        self.window = window
        self.min_observations = min_observations
        self.robust = robust

        self._history: collections.deque[float] = collections.deque()
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0

    @property
    def warmup(self) -> bool:
        """True when not enough observations have been seen to report spikes."""
        return len(self._history) < self.min_observations

    def update(self, loss: float) -> float | None:
        """Incorporate ``loss`` and return its z-score if it is a spike."""
        n = len(self._history)

        if n < self.min_observations:
            self._add(loss)
            return None

        # Baseline is computed from the existing window, excluding the current
        # value, so a spike cannot contaminate its own denominator.
        if self.robust:
            baseline = list(self._history)
            med = median(baseline)
            abs_deviations = [abs(v - med) for v in baseline]
            mad = median(abs_deviations)
            self._add(loss)
            if mad == 0.0:
                if loss != med:
                    return math.copysign(math.inf, loss - med)
                return None
            z_score = (loss - med) / (1.4826 * mad)
        else:
            mean = self._mean
            variance = self._m2 / (self._count - 1) if self._count > 1 else 0.0
            self._add(loss)
            if variance <= 0.0:
                if loss != mean:
                    return math.copysign(math.inf, loss - mean)
                return None
            z_score = (loss - mean) / math.sqrt(variance)

        return z_score if abs(z_score) > self.threshold else None

    def _add(self, value: float):
        if self.window is not None and len(self._history) >= self.window:
            self._remove(self._history.popleft())
        self._history.append(value)
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        delta2 = value - self._mean
        self._m2 += delta * delta2

    def _remove(self, value: float):
        if self._count <= 1:
            self._count = 0
            self._mean = 0.0
            self._m2 = 0.0
            return
        old_count = self._count
        self._count -= 1
        old_mean = self._mean
        self._mean = (old_count * old_mean - value) / self._count
        self._m2 -= (value - old_mean) * (value - self._mean)


# Backwards-compatible alias used by existing consumers.
SpikeDetector = ZScoreDetector
