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
    drifts before they escalate into full spikes. CUSUM is the *trigger*; it
    accumulates evidence across steps, which is what gives it a calibrated
    false-alarm rate. If ``ruptures`` is installed, a PELT (Pruned Exact
    Linear Time) model is evaluated on recent history **only when CUSUM has
    already fired**, to refine the score's magnitude — PELT never triggers on
    its own (see below).

    Score semantics: CUSUM-triggered spikes return the cumulative sum at
    trigger time, so ``|score| >= threshold`` always holds. When PELT
    confirms a change at the current observation, the returned score is the
    raw median/MAD-normalized deviation ``(loss - median) / (1.4826 * MAD)``
    instead of the clamped cumulative sum, preserving the true magnitude of
    the jump. PELT only overrides the score of an already-fired CUSUM spike;
    it is not an independent trigger.

    Why not an independent PELT trigger: PELT refits from scratch on the
    whole window at every call, with no sequential correction, so its
    per-step false-alarm rate is its raw single-test rate (measured ~2% on
    noise) — categorically different from CUSUM's accumulated-evidence
    design (calibrated to <0.05% in tests). Gating PELT behind a fired
    CUSUM keeps the calibrated false-alarm guarantee intact while still
    preserving jump magnitude. (Pre-1.8.1 code ran PELT as a would-be
    independent trigger, but a breakpoint-comparison bug kept that branch
    dead; its "PELT-triggered spikes" description never ran.)

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

        # Page's CUSUM calculation — this is the trigger. CUSUM accumulates
        # evidence across steps, which is what gives it a calibrated
        # false-alarm rate; PELT (below) never fires independently.
        z = (loss - mean) / std
        self._positive_cusum = max(0.0, self._positive_cusum + (z - self.slack))
        self._negative_cusum = min(0.0, self._negative_cusum + (z + self.slack))

        self._history.append(loss)
        self._trim()

        if not (
            self._positive_cusum >= self.threshold or abs(self._negative_cusum) >= self.threshold
        ):
            return None

        # CUSUM has fired. Reset accumulators and compute the canonical score.
        if self._positive_cusum >= self.threshold:
            score = self._positive_cusum
        else:
            score = self._negative_cusum
        self._positive_cusum = 0.0
        self._negative_cusum = 0.0

        # PELT refines the magnitude only — it runs solely because CUSUM
        # already fired, so its ~2% per-step raw single-test false-alarm rate
        # cannot leak into the calibrated CUSUM trigger. A PELT-confirmed
        # change at the current observation replaces the clamped cumulative
        # sum with the raw median/MAD-normalized deviation, preserving the
        # true magnitude of the jump.
        if rpt is not None and len(history) >= 2 * self.min_observations:
            try:
                signal = history.reshape(-1, 1)
                min_size = max(2, self.min_observations // 5)
                algo = rpt.Pelt(model="l2", min_size=min_size).fit(signal)
                change_points = algo.predict(pen=self.threshold**2)
                # ruptures' predict() always ends with the series length n
                # (the breakpoints are segment ends, sorted). A change "right
                # now" means the final segment is as short as the structural
                # minimum allows: its end is at n and its start is no earlier
                # than n - min_size, so the last real breakpoint sits in
                # [n - min_size, n). Comparing for equality against n - 1
                # would be unreachable for any min_size > 1 (min_size is at
                # least 2), silently dead like the pre-fix n comparison.
                if len(change_points) > 1 and change_points[-2] >= len(history) - min_size:
                    return float((loss - mean) / std)
            except Exception:
                pass

        return float(score)
