"""TensorBoard integration callback."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class TensorBoardCallback:
    """Log TrainScope step and spike metrics to TensorBoard.

    Requires the optional ``tensorboard`` dependency::

        pip install trainscope[integrations]
    """

    def __init__(self, log_dir: str | None = None, **kwargs: Any):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise ImportError(
                "TensorBoardCallback requires tensorboard. "
                "Install it with: pip install trainscope[integrations]"
            ) from exc

        self._writer = SummaryWriter(log_dir=log_dir, **kwargs)

    def on_step(
        self,
        global_snap: dict[str, Any],
        layer_snaps: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        step = global_snap.get("step")
        scalars = {
            "train/loss": global_snap.get("loss"),
            "train/grad_norm": global_snap.get("grad_norm_before_clip"),
            "train/lr": global_snap.get("lr"),
            "train/step_time_ms": global_snap.get("step_time_ms"),
        }
        if global_snap.get("cpu_memory_mb") is not None:
            scalars["system/cpu_memory_mb"] = global_snap.get("cpu_memory_mb")
        if global_snap.get("cuda_memory_mb") is not None:
            scalars["system/cuda_memory_mb"] = global_snap.get("cuda_memory_mb")

        try:
            for tag, value in scalars.items():
                if value is not None:
                    self._writer.add_scalar(tag, value, step)

            if layer_snaps:
                for layer_name, snap in layer_snaps.items():
                    grad_norm = snap.get("grad_l2_norm")
                    if grad_norm is not None:
                        self._writer.add_scalar(
                            f"layers/{layer_name}/grad_l2_norm", grad_norm, step
                        )
                    act_kurtosis = snap.get("act_kurtosis")
                    if act_kurtosis is not None:
                        self._writer.add_scalar(
                            f"layers/{layer_name}/act_kurtosis", act_kurtosis, step
                        )
        except Exception:
            logger.exception("Failed to log step to TensorBoard")

    def on_spike(
        self,
        spike_info: dict[str, Any],
        global_snap: dict[str, Any],
        layer_snaps: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        step = spike_info.get("step")
        try:
            self._writer.add_scalar("spike/z_score", spike_info.get("z_score", 0.0), step)
            self._writer.add_scalar("spike/loss", spike_info.get("loss", 0.0), step)
            self._writer.add_text(
                "spike/event",
                f"Spike at step {step}, z={spike_info.get('z_score')}",
                step,
            )
        except Exception:
            logger.exception("Failed to log spike to TensorBoard")
