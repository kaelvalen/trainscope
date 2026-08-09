"""Expert-utilization drift detector for Mixtral-style MoE models.

Detects routing *concentration* (one expert dominating the token routing)
before it becomes a loss divergence. The v1.3.0 experiment
(``scripts/verify_expert_collapse_signal.py``) established the empirical
claim: on a mini MoE (4 experts, top-2 routing) trained on wikitext-2, max
expert share exceeding ``0.85`` for several consecutive steps preceded the
loss explosion by 4-12 steps (mean 7.7) in 3/3 seeds, with zero false
positives in the stable control.

Importantly, the same experiment showed that a *dead* expert (share below a
small threshold) is NOT a signal: with top-2-of-4 routing one expert
naturally lingers near zero even in healthy runs. This detector therefore
keys on concentration, not on abandonment.
"""

import collections

from trainscope.core.detectors.base import AnomalyDetector


class ExpertUtilizationDriftDetector(AnomalyDetector):
    """Report spikes when per-step max expert share durably exceeds a threshold.

    The signal this detector consumes is the *max expert share* of the step
    (fraction of tokens routed to the most-used expert), not the loss. The
    scope feeds it via ``update()`` when this detector is active (see
    ``TrainScope.step``).

    Score semantics: triggered steps return the max share at trigger time
    (``>= threshold``), so the magnitude of the concentration is preserved.
    """

    def __init__(
        self,
        threshold: float = 0.85,
        min_observations: int = 30,
        run_steps: int = 3,
        window: int = 200,
    ):
        self.threshold = threshold
        self.min_observations = min_observations
        self.run_steps = run_steps
        self.window = window
        self._history: collections.deque[float] = collections.deque()
        self._consecutive_over = 0

    @property
    def warmup(self) -> bool:
        return len(self._history) < self.min_observations

    def _trim(self) -> None:
        if len(self._history) > self.window:
            self._history.popleft()

    def update(self, max_share: float) -> float | None:
        if len(self._history) < self.min_observations:
            self._history.append(max_share)
            return None

        if max_share >= self.threshold:
            self._consecutive_over += 1
        else:
            self._consecutive_over = 0

        self._history.append(max_share)
        self._trim()

        if self._consecutive_over >= self.run_steps:
            self._consecutive_over = 0
            return float(max_share)
        return None


__all__ = ["ExpertUtilizationDriftDetector"]
