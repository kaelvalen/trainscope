"""Weights & Biases integration callback."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class WandbCallback:
    """Log TrainScope step and spike metrics to Weights & Biases.

    Requires the optional ``wandb`` dependency::

        pip install trainscope[integrations]
    """

    def __init__(self, project: str | None = None, entity: str | None = None, **kwargs: Any):
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "WandbCallback requires wandb. Install it with: pip install trainscope[integrations]"
            ) from exc

        active_run = getattr(wandb, "run", None)
        if active_run is not None:
            self._run = active_run
        else:
            self._run = wandb.init(project=project, entity=entity, **kwargs)

    def on_step(
        self,
        global_snap: dict[str, Any],
        layer_snaps: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        step = global_snap.get("step")
        metrics = {
            "train/loss": global_snap.get("loss"),
            "train/grad_norm": global_snap.get("grad_norm_before_clip"),
            "train/lr": global_snap.get("lr"),
            "train/step_time_ms": global_snap.get("step_time_ms"),
        }
        if global_snap.get("cpu_memory_mb") is not None:
            metrics["system/cpu_memory_mb"] = global_snap.get("cpu_memory_mb")
        if global_snap.get("cuda_memory_mb") is not None:
            metrics["system/cuda_memory_mb"] = global_snap.get("cuda_memory_mb")

        if layer_snaps:
            max_grad_norm = max(
                (s.get("grad_l2_norm", 0.0) for s in layer_snaps.values()),
                default=0.0,
            )
            max_act_kurtosis = max(
                (s.get("act_kurtosis", 0.0) for s in layer_snaps.values()),
                default=0.0,
            )
            metrics["layers/max_grad_l2_norm"] = max_grad_norm
            metrics["layers/max_act_kurtosis"] = max_act_kurtosis

        metrics = {k: v for k, v in metrics.items() if v is not None}
        try:
            self._run.log(metrics, step=step)
        except Exception:
            logger.exception("Failed to log step to WandB")

    def on_spike(
        self,
        spike_info: dict[str, Any],
        global_snap: dict[str, Any],
        layer_snaps: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        step = spike_info.get("step")
        metrics = {
            "spike/z_score": spike_info.get("z_score"),
            "spike/loss": spike_info.get("loss"),
        }
        metrics = {k: v for k, v in metrics.items() if v is not None}
        try:
            self._run.log({"spike": True, **metrics}, step=step)
            if hasattr(self._run, "alert"):
                z_score = spike_info.get("z_score", 0.0)
                loss_val = spike_info.get("loss", 0.0)
                self._run.alert(
                    title=f"TrainScope Loss Spike Detected at Step {step}",
                    text=f"Loss spike detected at step {step} (z-score: {z_score:.2f}, loss: {loss_val:.4f}).",
                    level="WARN",
                )
        except Exception:
            logger.exception("Failed to log spike to WandB")
