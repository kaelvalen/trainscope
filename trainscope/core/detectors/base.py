"""Abstract base class for trainscope anomaly detectors."""

from abc import ABC, abstractmethod


class AnomalyDetector(ABC):
    """Base class for anomaly detectors consumed by :class:`TrainScope`."""

    @abstractmethod
    def update(self, loss: float) -> float | None:
        """Incorporate ``loss`` and return an anomaly score if one is detected.

        A return value of ``None`` means no anomaly. Non-``None`` values are
        treated as anomaly scores and surfaced in the spike payload.
        """

    @property
    @abstractmethod
    def warmup(self) -> bool:
        """True when the detector has not yet seen enough observations."""
