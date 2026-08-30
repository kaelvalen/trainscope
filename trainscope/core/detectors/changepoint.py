"""Change-point anomaly detector with optional ``ruptures`` backend."""

import collections

import numpy as np

from trainscope.core.detectors.base import AnomalyDetector

try:
    import ruptures as rpt
except Exception:  # pragma: no cover
    rpt = None


class ChangePointDetector(AnomalyDetector):
    """Detect distributional shifts and cumulative drift in the loss stream.

    Uses Page's CUSUM (Cumulative Sum) test to detect subtle, persistent loss
    drifts before they escalate into full spikes. If ``ruptures`` is installed,
    a PELT (Pruned Exact Linear Time) model is additionally evaluated on recent
    history for exact change point localization.

    Score semantics: CUSUM-triggered spikes return the cumulative sum at
    trigger time, so ``|score| >= threshold`` always holds. PELT-triggered
    spikes return the raw median/MAD-normalized deviation
    ``(loss - median) / (1.4826 * MAD)`` of the triggering observation,
    preserving the true magnitude of the jump: subtle change points can carry
    ``|score|`` below ``threshold``. Since 0.6.0 the PELT score is no longer
    clamped to ``threshold`` (it previously made every PELT spike look
    identical at ``|score| == threshold``).

    Parameters
    ----------
    threshold:
        Decision threshold for cumulative sum ($h$).
    slack:
        Allowance / shift reference parameter ($k$).
    window:
        Number of recent observations to consider for baseline statistics.
    min_observations:
        Number of observations required before reporting.
    """

    def __init__(
        self,
        threshold: float = 6.0,
        slack: float = 1.0,
        window: int = 200,
        min_observations: int = 30,
    ):
        self.threshold = threshold
        self.slack = slack
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

    def update(self, loss: float) -> float | None:
        if len(self._history) < self.min_observations:
            self._history.append(loss)
            return None

        history = np.fromiter(self._history, dtype=float, count=len(self._history))
        mean = float(np.median(history))
        mad = float(np.median(np.abs(history - mean)))
        std = float(mad * 1.4826)

        if std < 1e-12:
            std = 1e-12

        # Optional ruptures PELT evaluation if installed. This runs alongside
        # (not instead of) the CUSUM test below: a PELT-confirmed change
        # point at the current observation resets the CUSUM accumulators (so
        # stale pre-change-point state doesn't leak into the next call) and
        # is reported with its raw deviation magnitude (see the class
        # docstring for score semantics), not a value clamped to ``threshold``.
        if rpt is not None and len(history) >= 2 * self.min_observations:
            try:
                signal = history.reshape(-1, 1)
                algo = rpt.Pelt(model="l2", min_size=max(2, self.min_observations // 5)).fit(signal)
                change_points = algo.predict(pen=self.threshold**2)
                # ruptures' predict() always ends with the series length n
                # (the breakpoints are segment ends); a change "right now"
                # means the final segment holds only the current observation,
                # i.e. the last real breakpoint is n - 1. (The previous code
                # compared against n, which no real ruptures output ever
                # equals, so the PELT path was dead.)
                if len(change_points) > 1 and change_points[-2] == len(history) - 1:
                    self._history.append(loss)
                    self._trim()
                    self._positive_cusum = 0.0
                    self._negative_cusum = 0.0
                    return float((loss - mean) / std)
            except Exception:
                pass

        # Page's CUSUM calculation
        z = (loss - mean) / std
        self._positive_cusum = max(0.0, self._positive_cusum + (z - self.slack))
        self._negative_cusum = min(0.0, self._negative_cusum + (z + self.slack))

        self._history.append(loss)
        self._trim()

        if self._positive_cusum >= self.threshold:
            score = self._positive_cusum
            self._positive_cusum = 0.0
            self._negative_cusum = 0.0
            return float(score)
        elif abs(self._negative_cusum) >= self.threshold:
            score = self._negative_cusum
            self._positive_cusum = 0.0
            self._negative_cusum = 0.0
            return float(score)

        return None
