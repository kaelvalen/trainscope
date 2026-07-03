"""Backwards-compatible re-export of the z-score spike detector."""

from trainscope.core.detectors.z_score import SpikeDetector, ZScoreDetector

__all__ = ["SpikeDetector", "ZScoreDetector"]
