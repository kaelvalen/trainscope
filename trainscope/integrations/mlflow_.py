"""MLflow integration callback."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MlflowCallback:
    """Log TrainScope step and spike metrics to MLflow.

    Requires the optional ``mlflow`` dependency::

        pip install trainscope[integrations]
    """

    def __init__(self, experiment_name: str | None = None, **kwargs: Any):
        try:
            import mlflow
        except ImportError as exc:
            raise ImportError(
                "MlflowCallback requires mlflow. Install it with: pip install trainscope[integrations]"
            ) from exc

        self._mlflow = mlflow
        if experiment_name is not None:
            mlflow.set_experiment(experiment_name)
        self._run = mlflow.start_run(**kwargs) if not mlflow.active_run() else mlflow.active_run()

    def on_step(
        self,
        global_snap: dict[str, Any],
        layer_snaps: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        step = global_snap.get("step")
        metrics = {
            "train_loss": global_snap.get("loss"),
            "grad_norm": global_snap.get("grad_norm_before_clip"),
            "learning_rate": global_snap.get("lr"),
            "step_time_ms": global_snap.get("step_time_ms"),
        }
        if global_snap.get("cpu_memory_mb") is not None:
            metrics["cpu_memory_mb"] = global_snap.get("cpu_memory_mb")
        if global_snap.get("cuda_memory_mb") is not None:
            metrics["cuda_memory_mb"] = global_snap.get("cuda_memory_mb")

        metrics = {k: v for k, v in metrics.items() if v is not None}
        try:
            self._mlflow.log_metrics(metrics, step=step)

            if layer_snaps:
                top_layer = max(
                    layer_snaps.items(),
                    key=lambda item: item[1].get("grad_l2_norm", 0.0),
                    default=(None, {}),
                )
                if top_layer[0] is not None:
                    self._mlflow.log_metric(
                        "max_grad_l2_norm",
                        top_layer[1].get("grad_l2_norm", 0.0),
                        step=step,
                    )
                    self._mlflow.log_param("top_grad_layer", top_layer[0])
        except Exception:
            logger.exception("Failed to log step to MLflow")

    def on_spike(
        self,
        spike_info: dict[str, Any],
        global_snap: dict[str, Any],
        layer_snaps: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        step = spike_info.get("step")
        metrics = {
            "spike_z_score": spike_info.get("z_score"),
            "spike_loss": spike_info.get("loss"),
        }
        metrics = {k: v for k, v in metrics.items() if v is not None}
        try:
            self._mlflow.log_metrics(metrics, step=step)
            self._mlflow.log_param("spike_step", step)
        except Exception:
            logger.exception("Failed to log spike to MLflow")
