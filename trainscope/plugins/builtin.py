"""Built-in example plugins shipped with trainscope."""

from typing import Any, ClassVar

from trainscope.plugins import MetricPlugin


class GradientNormRatioPlugin(MetricPlugin):
    """Tracks the ratio of the current global gradient norm to its rolling mean.

    A sudden increase in this ratio often precedes a loss spike.
    """

    name: ClassVar[str] = "gradient_norm_ratio"

    def __init__(self, window: int = 50):
        self.window = window
        self._history: list[float] = []

    def compute(self, model: Any, optimizer: Any, step: int) -> dict[str, float]:
        total_sq = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_sq += float(param.grad.detach().float().norm(2).item() ** 2)
        grad_norm = total_sq**0.5

        self._history.append(grad_norm)
        if len(self._history) > self.window:
            self._history.pop(0)

        avg = sum(self._history) / len(self._history) if self._history else 0.0
        ratio = grad_norm / avg if avg > 0.0 else 0.0

        return {
            "gradient_norm": grad_norm,
            "gradient_norm_ratio": ratio,
        }
