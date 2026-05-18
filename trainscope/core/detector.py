import collections
import math


class SpikeDetector:
    def __init__(self, threshold: float = 3.5, window: int = 200):
        self.threshold = threshold
        self._min_observations = 30
        self._history: collections.deque = collections.deque(maxlen=window)

    def update(self, loss: float) -> float | None:
        n = len(self._history)

        if n < self._min_observations:
            self._history.append(loss)
            return None

        # Compute baseline from existing window BEFORE including current value
        # so a spike cannot inflate its own z-score denominator.
        mean = sum(self._history) / n
        variance = sum((x - mean) ** 2 for x in self._history) / (n - 1)
        self._history.append(loss)

        if variance <= 0.0:
            # Perfectly flat baseline: any deviation is an infinite z-score.
            if loss != mean:
                return math.copysign(math.inf, loss - mean)
            return None

        z_score = (loss - mean) / math.sqrt(variance)
        return z_score if abs(z_score) > self.threshold else None
