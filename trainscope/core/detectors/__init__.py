"""Anomaly detector implementations and factory."""

from typing import TYPE_CHECKING

from trainscope.core.detectors.base import AnomalyDetector
from trainscope.core.detectors.changepoint import ChangePointDetector
from trainscope.core.detectors.percentile import PercentileDetector
from trainscope.core.detectors.z_score import SpikeDetector, ZScoreDetector

if TYPE_CHECKING:
    from trainscope.core.config import TrainScopeConfig

_REGISTRY: dict[str, type[AnomalyDetector]] = {
    "z_score": ZScoreDetector,
    "percentile": PercentileDetector,
    "changepoint": ChangePointDetector,
}


def register_detector(name: str, cls: type[AnomalyDetector]) -> None:
    """Register a detector class under ``name`` for config lookup."""
    if not issubclass(cls, AnomalyDetector):
        raise TypeError(f"Detector class must subclass AnomalyDetector, got {cls}")
    _REGISTRY[name] = cls


def make_detector(config: "TrainScopeConfig | dict | None" = None) -> AnomalyDetector:
    """Build an :class:`AnomalyDetector` from a config object or dict.

    ``config.detector`` (or the dict) must contain a ``name`` key. Any
    remaining keys are passed to the detector constructor.
    """
    if config is None:
        detector_cfg = {"name": "z_score"}
    elif hasattr(config, "detector"):
        detector_cfg = dict(getattr(config, "detector") or {"name": "z_score"})
        # spike_threshold is scaled for the z_score detector's raw z-score
        # cutoff; other detectors either don't accept a "threshold" kwarg
        # (e.g. percentile) or use it on a different scale (e.g. CUSUM's
        # cumulative-sum decision threshold), so only apply it there.
        if (
            detector_cfg.get("name") == "z_score"
            and "threshold" not in detector_cfg
            and hasattr(config, "spike_threshold")
        ):
            detector_cfg.setdefault("threshold", getattr(config, "spike_threshold"))
    elif isinstance(config, dict):
        detector_cfg = dict(config)
    else:
        raise TypeError(f"Expected TrainScopeConfig or dict, got {type(config)}")

    name = detector_cfg.pop("name", "z_score")
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown detector '{name}'. Available: {', '.join(sorted(_REGISTRY))}")
    return cls(**detector_cfg)


__all__ = [
    "AnomalyDetector",
    "ZScoreDetector",
    "SpikeDetector",
    "PercentileDetector",
    "ChangePointDetector",
    "register_detector",
    "make_detector",
]
