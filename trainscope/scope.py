import logging
import math
import time
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from trainscope.core.buffer import RollingBuffer
from trainscope.core.config import TrainScopeConfig
from trainscope.core.detectors import make_detector
from trainscope.core.metrics import (
    compute_activation_metrics,
    compute_gradient_metrics,
    compute_weight_histogram,
    compute_weight_metrics,
)
from trainscope.io import RemoteWriter
from trainscope.io.writer import DiskWriter
from trainscope.plugins import instantiate_metric_plugins, load_detector_plugins

logger = logging.getLogger("trainscope")


class StopTraining(Exception):
    def __init__(self, step: int, z_score: float):
        super().__init__(f"Spike detected at step {step} (z={z_score:.2f})")
        self.step = step
        self.z_score = z_score


class TrainScope:
    """Context-aware training debugger that records per-step metrics and spikes.

    Recommended usage::

        scope = TrainScope(model, optimizer, config).attach()
        for batch in loader:
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            scope.step(loss.item())   # before optimizer.step()
            optimizer.step()

    ``step()`` may also be called after ``optimizer.step()`` for backward
    compatibility, but recording gradient norms before the step is preferred.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        config: TrainScopeConfig | None = None,
    ):
        self._model = model
        self._optimizer = optimizer
        self._config = config or TrainScopeConfig()

        run_name = self._config.run_name
        assert run_name is not None

        if self._config.storage_uri:
            suffix = f"_rank{self._config.rank}" if self._config.rank is not None else ""
            run_path = f"{self._config.storage_uri.rstrip('/')}/{run_name}{suffix}"
            self._writer: DiskWriter | RemoteWriter = RemoteWriter(run_path, self._config)
        else:
            run_path = Path(self._config.run_dir) / run_name
            if self._config.rank is not None:
                run_path = run_path.parent / f"{run_path.name}_rank{self._config.rank}"
            self._writer = DiskWriter(run_path, self._config)

        self._buffer = RollingBuffer(
            full_resolution_window=self._config.full_resolution_window,
            decimation_factor=self._config.decimation_factor,
        )

        # Load detector plugins (entry points + explicit config paths) before
        # selecting the detector so plugin detector names are available.
        load_detector_plugins(self._config.detector_plugins)
        self._detector = make_detector(self._config)
        self._metric_plugins = instantiate_metric_plugins(self._config.metric_plugins)

        self._act_cache: dict[str, dict] = {}
        self._hooks: list[Any] = []
        self._step_idx = 0
        self._last_step_time: float | None = None

    @property
    def writer(self) -> DiskWriter | RemoteWriter:
        return self._writer

    def _metric_device(self, tensor: torch.Tensor) -> str | torch.device:
        """Return the device on which ``tensor`` should be inspected.

        By default metrics are computed on CPU to avoid GPU synchronization.
        Set ``config.device`` explicitly (e.g. to the tensor's device or another
        device) to override this.
        """
        if self._config.device is not None:
            return self._config.device
        return "cpu"

    def attach(self) -> "TrainScope":
        self._writer.write_meta(
            model_name=self._model.__class__.__name__,
            model_config=self._config.to_dict(),
        )

        for name, module in self._model.named_modules():
            children = list(module.children())
            if children:
                continue

            filt = self._config.activation_layer_filter
            if filt is not None and not any(s in name for s in filt):
                continue

            def make_forward_hook(layer_name: str):
                def hook(module, input, output):
                    n = self._step_idx
                    if n % self._config.trace_every_n_steps != 0:
                        return

                    tensor = None
                    if isinstance(output, torch.Tensor):
                        tensor = output
                    elif isinstance(output, (tuple, list)):
                        for item in output:
                            if isinstance(item, torch.Tensor):
                                tensor = item
                                break

                    if tensor is None:
                        return

                    if n % self._config.activation_metrics_every_n_steps == 0:
                        self._act_cache[layer_name] = compute_activation_metrics(
                            tensor, device=self._metric_device(tensor)
                        )

                return hook

            h_fwd = module.register_forward_hook(make_forward_hook(name))
            self._hooks.append(h_fwd)

        logger.info("TrainScope attached to %s", self._model.__class__.__name__)
        return self

    def _compute_global_grad_norm(self) -> float:
        total_sq = 0.0
        for param in self._model.parameters():
            if param.grad is not None:
                total_sq += float(param.grad.detach().float().norm(2).item() ** 2)
        return math.sqrt(total_sq)

    def _compute_optimizer_v_norm(self) -> float:
        total_sq = 0.0
        state = self._optimizer.state
        opt_type = type(self._optimizer).__name__
        if opt_type not in ("Adam", "AdamW"):
            return 0.0
        for param in self._model.parameters():
            if param in state and "exp_avg_sq" in state[param]:
                v = state[param]["exp_avg_sq"]
                total_sq += float(v.detach().float().norm(2).item() ** 2)
        return math.sqrt(total_sq)

    def _get_lr(self) -> float:
        return float(self._optimizer.param_groups[0]["lr"])

    @staticmethod
    def _memory_mb() -> tuple[float, float]:
        """Return (cpu_memory_mb, cuda_memory_mb)."""
        import resource

        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        cpu_mb = rss_kb / 1024.0
        cuda_mb = 0.0
        if torch.cuda.is_available():
            try:
                cuda_mb = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
            except Exception:
                pass
        return cpu_mb, cuda_mb

    def _run_metric_plugins(self, step: int):
        for plugin in self._metric_plugins:
            try:
                metrics = plugin.compute(self._model, self._optimizer, step)
                if metrics:
                    self._writer.append_plugin_metrics(step, plugin.name, metrics)
            except Exception:
                logger.exception(
                    "Metric plugin %s failed at step %d", plugin.name, step
                )

    def step(
        self,
        loss: float,
        *,
        batch_index: int | None = None,
        clip_grad_norm: float | None = None,
    ) -> dict | None:
        step_idx = self._step_idx
        self._step_idx += 1

        now = time.monotonic()
        if self._last_step_time is not None:
            step_time_ms = (now - self._last_step_time) * 1000.0
        else:
            step_time_ms = 0.0
        self._last_step_time = now

        if step_idx % self._config.trace_every_n_steps != 0:
            return None

        if clip_grad_norm is not None:
            warnings.warn(
                "clip_grad_norm is deprecated in TrainScope.step(); call "
                "torch.nn.utils.clip_grad_norm_() yourself before step() instead. "
                "This parameter is ignored and will be removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )

        loss_f = float(loss)
        if not math.isfinite(loss_f):
            logger.warning("Non-finite loss recorded at step %d: %s", step_idx, loss_f)

        grad_norm_before = self._compute_global_grad_norm()
        grad_norm_after = grad_norm_before

        optimizer_v_norm = self._compute_optimizer_v_norm()
        lr = self._get_lr()

        if math.isnan(loss_f):
            # Do not poison the detector's baseline with NaN.
            score = None
        else:
            score = self._detector.update(loss_f)
        is_spike = score is not None

        should_histogram = step_idx % self._config.histogram_every_n_steps == 0 or is_spike

        global_snap = {
            "step": step_idx,
            "loss": loss_f,
            "grad_norm_before_clip": grad_norm_before,
            "grad_norm_after_clip": grad_norm_after,
            "lr": lr,
            "optimizer_v_norm": optimizer_v_norm,
            "step_time_ms": step_time_ms,
            "batch_index": batch_index if batch_index is not None else -1,
            "is_spike": is_spike,
        }
        if self._config.track_memory:
            cpu_mb, cuda_mb = self._memory_mb()
            global_snap["cpu_memory_mb"] = cpu_mb
            global_snap["cuda_memory_mb"] = cuda_mb

        self._run_metric_plugins(step_idx)

        layer_snaps: dict[str, dict] = {}
        for name, param in self._model.named_parameters():
            module_name = name.rsplit(".", 1)[0] if "." in name else name
            compute_dev = self._metric_device(param.data)
            weight_metrics = compute_weight_metrics(param.data, device=compute_dev)
            if should_histogram:
                hist_counts, hist_edges = compute_weight_histogram(
                    param.data, n_bins=self._config.n_histogram_bins, device=compute_dev
                )
            else:
                hist_counts, hist_edges = [], []

            act_metrics = self._act_cache.get(
                module_name,
                {
                    "act_mean": 0.0,
                    "act_std": 0.0,
                    "act_max_abs": 0.0,
                    "act_kurtosis": 0.0,
                    "act_min": 0.0,
                    "act_max": 0.0,
                    "act_median": 0.0,
                },
            )
            grad_metrics = compute_gradient_metrics(param.grad, device=compute_dev)

            layer_name = name
            layer_snap = {
                "step": step_idx,
                "grad_l2_norm": grad_metrics.get("grad_l2_norm", 0.0),
                "weight_l2_norm": weight_metrics.get("weight_l2_norm", 0.0),
                "act_mean": act_metrics.get("act_mean", 0.0),
                "act_std": act_metrics.get("act_std", 0.0),
                "act_max_abs": act_metrics.get("act_max_abs", 0.0),
                "act_kurtosis": act_metrics.get("act_kurtosis", 0.0),
                "grad_nan_inf_ratio": grad_metrics.get("grad_nan_inf_ratio", 0.0),
                "hist_counts": hist_counts,
                "hist_edges": hist_edges,
                "grad_max_abs": grad_metrics.get("grad_max_abs", 0.0),
                "grad_mean": grad_metrics.get("grad_mean", 0.0),
                "weight_mean": weight_metrics.get("weight_mean", 0.0),
                "weight_std": weight_metrics.get("weight_std", 0.0),
                "weight_max_abs": weight_metrics.get("weight_max_abs", 0.0),
                "weight_min": weight_metrics.get("weight_min", 0.0),
                "act_min": act_metrics.get("act_min", 0.0),
                "act_max": act_metrics.get("act_max", 0.0),
                "act_median": act_metrics.get("act_median", 0.0),
            }
            layer_snaps[layer_name] = layer_snap

        self._buffer.add(global_snap, layer_snaps)
        self._writer.append_global(global_snap)
        for layer_name, layer_snap in layer_snaps.items():
            self._writer.append_layer(layer_name, layer_snap)

        self._act_cache.clear()

        if is_spike:
            window = self._buffer.get_window(
                step_idx,
                before=self._config.spike_window_before,
                after=self._config.spike_window_after,
            )
            layer_windows: dict[str, list[dict]] = {}
            for entry in window:
                for lname, lsnap in entry["layers"].items():
                    layer_windows.setdefault(lname, []).append(lsnap)

            self._writer.write_spike_window(step_idx, window, layer_windows)
            self._writer.save_rng_state(step_idx)

            if self._config.checkpoint_on_spike:
                try:
                    optimizer_state = (
                        self._optimizer.state_dict()
                        if hasattr(self._optimizer, "state_dict")
                        else None
                    )
                    self._writer.save_checkpoint(
                        step_idx,
                        self._model.state_dict(),
                        optimizer_state=optimizer_state,
                    )
                except Exception:
                    logger.exception("Failed to save checkpoint on spike at step %d", step_idx)

            if (
                self._config.rng_every_n_steps > 0
                and step_idx % self._config.rng_every_n_steps != 0
            ):
                # Spike already saved above; only save here if not already saved.
                pass

            self._writer.flush()

            assert score is not None
            spike_info = {
                "step": step_idx,
                "loss": loss_f,
                "z_score": float(score),
            }

            logger.info("Spike detected at step %d (score=%.2f)", step_idx, float(score))

            if self._config.stop_on_spike:
                raise StopTraining(step_idx, score)

            return spike_info

        if self._config.rng_every_n_steps > 0 and step_idx % self._config.rng_every_n_steps == 0:
            self._writer.save_rng_state(step_idx)

        return None

    def detach(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._writer.close()
        logger.info("TrainScope detached")

    def __enter__(self):
        return self.attach()

    def __exit__(self, *args):
        self.detach()
