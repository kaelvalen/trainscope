import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from losswatch.core.buffer import RollingBuffer
from losswatch.core.config import LossWatchConfig
from losswatch.core.detector import SpikeDetector
from losswatch.core.metrics import (
    compute_activation_metrics,
    compute_gradient_metrics,
    compute_weight_histogram,
    compute_weight_metrics,
)
from losswatch.io.writer import DiskWriter


class StopTraining(Exception):
    def __init__(self, step: int, z_score: float):
        super().__init__(f"Spike detected at step {step} (z={z_score:.2f})")
        self.step = step
        self.z_score = z_score


class LossWatch:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        config: LossWatchConfig | None = None,
    ):
        self._model = model
        self._optimizer = optimizer
        self._config = config or LossWatchConfig()

        run_path = Path(self._config.run_dir) / self._config.run_name
        if self._config.rank is not None:
            run_path = run_path.parent / f"{run_path.name}_rank{self._config.rank}"
        self._writer = DiskWriter(run_path, self._config)
        self._buffer = RollingBuffer(
            full_resolution_window=self._config.full_resolution_window,
            decimation_factor=self._config.decimation_factor,
        )
        self._detector = SpikeDetector(threshold=self._config.spike_threshold)

        self._act_cache: dict[str, dict] = {}
        self._hooks: list[Any] = []
        self._step_idx = 0
        self._last_step_time: float | None = None

    @property
    def writer(self) -> DiskWriter:
        return self._writer

    def attach(self) -> "LossWatch":
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
                    if (n % self._config.trace_every_n_steps != 0 or
                            n % self._config.activation_metrics_every_n_steps != 0):
                        return
                    tensor = None
                    if isinstance(output, torch.Tensor):
                        tensor = output
                    elif isinstance(output, (tuple, list)):
                        for item in output:
                            if isinstance(item, torch.Tensor):
                                tensor = item
                                break
                    if tensor is not None:
                        self._act_cache[layer_name] = compute_activation_metrics(tensor)
                return hook

            h_fwd = module.register_forward_hook(make_forward_hook(name))
            self._hooks.append(h_fwd)

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

        grad_norm_before = self._compute_global_grad_norm()

        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), clip_grad_norm)
            grad_norm_after = self._compute_global_grad_norm()
        else:
            grad_norm_after = grad_norm_before

        optimizer_v_norm = self._compute_optimizer_v_norm()
        lr = self._get_lr()

        z_score = self._detector.update(loss)
        is_spike = z_score is not None
        should_histogram = (
            step_idx % self._config.histogram_every_n_steps == 0 or is_spike
        )

        global_snap = {
            "step": step_idx,
            "loss": float(loss),
            "grad_norm_before_clip": grad_norm_before,
            "grad_norm_after_clip": grad_norm_after,
            "lr": lr,
            "optimizer_v_norm": optimizer_v_norm,
            "step_time_ms": step_time_ms,
            "batch_index": batch_index if batch_index is not None else -1,
            "is_spike": is_spike,
        }

        layer_snaps: dict[str, dict] = {}
        for name, param in self._model.named_parameters():
            module_name = name.rsplit(".", 1)[0] if "." in name else name
            weight_metrics = compute_weight_metrics(param.data)
            if should_histogram:
                hist_counts, hist_edges = compute_weight_histogram(
                    param.data, n_bins=self._config.n_histogram_bins
                )
            else:
                hist_counts, hist_edges = [], []

            act_metrics = self._act_cache.get(module_name, {
                "act_mean": 0.0, "act_std": 0.0,
                "act_max_abs": 0.0, "act_kurtosis": 0.0,
            })
            grad_metrics = compute_gradient_metrics(param.grad)

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
            self._writer.flush()

            spike_info = {
                "step": step_idx,
                "loss": float(loss),
                "z_score": float(z_score),
            }

            if self._config.stop_on_spike:
                raise StopTraining(step_idx, z_score)

            return spike_info

        return None

    def detach(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __enter__(self):
        return self.attach()

    def __exit__(self, *args):
        self.detach()
        self._writer.close()
