import math
from typing import TypeGuard

import torch
import torch.distributed as dist

try:
    from torch.distributed.tensor import DTensor, Shard
except ImportError:  # torch < 2.4: no public DTensor, so no FSDP2 parameters either
    DTensor = None  # type: ignore[assignment,misc]


def _is_dtensor(tensor: torch.Tensor) -> TypeGuard[DTensor]:
    return DTensor is not None and isinstance(tensor, DTensor)


def _to_compute_device(tensor: torch.Tensor, device: str | torch.device) -> torch.Tensor:
    """Move a flattened float copy of ``tensor`` to the requested compute device."""
    flat = tensor.detach().float().flatten()
    if str(flat.device) == str(device):
        return flat
    return flat.to(device)


def _is_sharded(tensor: torch.Tensor) -> TypeGuard[DTensor]:
    """A DTensor sharded over a 1-D mesh (FSDP2 ``fully_shard``)."""
    return (
        _is_dtensor(tensor)
        and tensor.device_mesh.ndim == 1
        and isinstance(tensor.placements[0], Shard)
    )


def _unshard(tensor: torch.Tensor) -> torch.Tensor:
    # shortcut: DTensors on multi-dim (HSDP/TP) or replicated meshes are gathered
    # whole; reduce per sharded mesh dim if that memory spike matters.
    return tensor.full_tensor() if _is_dtensor(tensor) else tensor


def _sharded_moments(tensor: DTensor) -> dict[str, float]:
    """Global statistics of a sharded DTensor from per-shard partials.

    DTensor's own reductions fail when a rank's shard is empty (a parameter
    smaller than the world size), so each rank reduces its local shard with
    identity values for the empty case and two all-reduces combine them.
    """
    group = tensor.device_mesh.get_group()
    local = tensor.to_local().detach().double().flatten()
    sums = torch.stack(
        [
            local.new_tensor(float(local.numel())),
            local.sum(),
            (local * local).sum(),
            (~torch.isfinite(local)).sum().double(),
        ]
    )
    if local.numel():
        extremes = torch.stack([local.max(), -local.min(), local.abs().max()])
    else:
        extremes = local.new_full((3,), -math.inf)
    dist.all_reduce(sums, group=group)
    dist.all_reduce(extremes, op=dist.ReduceOp.MAX, group=group)
    n, total, total_sq, nonfinite = sums.tolist()
    max_v, neg_min, max_abs = extremes.tolist()
    mean = total / n
    return {
        "n": n,
        "mean": mean,
        "std": math.sqrt(max(total_sq / n - mean * mean, 0.0)),
        "l2": math.sqrt(total_sq),
        "max": max_v,
        "min": -neg_min,
        "max_abs": max_abs,
        "nonfinite": nonfinite,
    }


def _sharded_histogram(weight: DTensor, n_bins: int) -> tuple[list[float], list[float]]:
    """Histogram of a 1-D-mesh sharded DTensor without gathering it: the global
    range and the bin counts are all-reduced across the shards."""
    group = weight.device_mesh.get_group()
    local = weight.to_local().detach().float().flatten()
    finite = local[torch.isfinite(local)]
    if finite.numel():
        bounds = torch.stack([-finite.min(), finite.max()])
    else:
        bounds = torch.full((2,), -math.inf, device=local.device)
    dist.all_reduce(bounds, op=dist.ReduceOp.MAX, group=group)
    w_min, w_max = -float(bounds[0]), float(bounds[1])
    if not math.isfinite(w_min):
        return [0.0] * n_bins, [0.0] * (n_bins + 1)
    if w_min == w_max:
        # Same layout as the unsharded constant-tensor case.
        count = torch.tensor([float(finite.numel())], device=local.device)
        dist.all_reduce(count, group=group)
        counts = [0.0] * n_bins
        counts[n_bins // 2] = float(count)
        return counts, [w_min + i - n_bins / 2 for i in range(n_bins + 1)]
    counts_t = torch.histc(finite, bins=n_bins, min=w_min, max=w_max)
    dist.all_reduce(counts_t, group=group)
    edges = torch.linspace(w_min, w_max, n_bins + 1)
    return [float(v) for v in counts_t.tolist()], [float(v) for v in edges.tolist()]


def compute_activation_metrics(act: torch.Tensor, device: str | torch.device = "cpu") -> dict:
    if act.numel() == 0:
        return {
            "act_mean": 0.0,
            "act_std": 0.0,
            "act_max_abs": 0.0,
            "act_kurtosis": 0.0,
            "act_min": 0.0,
            "act_max": 0.0,
            "act_median": 0.0,
        }

    act_f = _to_compute_device(act, device)
    mean = act_f.mean()
    std = act_f.std(unbiased=False)
    max_abs = act_f.abs().max()

    if std.item() == 0.0:
        kurtosis = 0.0
    else:
        normalized = (act_f - mean) / std
        kurtosis = float((normalized**4).mean().item()) - 3.0

    return {
        "act_mean": float(mean.item()),
        "act_std": float(std.item()),
        "act_max_abs": float(max_abs.item()),
        "act_kurtosis": kurtosis,
        "act_min": float(act_f.min().item()),
        "act_max": float(act_f.max().item()),
        "act_median": float(act_f.median().item()),
    }


def compute_gradient_metrics(grad: torch.Tensor | None, device: str | torch.device = "cpu") -> dict:
    if grad is None or grad.numel() == 0:
        return {
            "grad_l2_norm": 0.0,
            "grad_nan_inf_ratio": 0.0,
            "grad_max_abs": 0.0,
            "grad_mean": 0.0,
        }

    if _is_sharded(grad):
        m = _sharded_moments(grad)
        return {
            "grad_l2_norm": m["l2"],
            "grad_nan_inf_ratio": m["nonfinite"] / m["n"],
            "grad_max_abs": m["max_abs"],
            "grad_mean": m["mean"],
        }
    grad_f = _to_compute_device(_unshard(grad), device)
    l2_norm = float(grad_f.norm(2).item())
    # A finite norm proves every element is finite (NaN/Inf propagate), so the
    # elementwise count is only needed when the norm is not.
    if math.isfinite(l2_norm):
        nan_inf_ratio = 0.0
    else:
        nan_inf_ratio = float((~torch.isfinite(grad_f)).sum().item()) / grad_f.numel()

    return {
        "grad_l2_norm": l2_norm,
        "grad_nan_inf_ratio": float(nan_inf_ratio),
        "grad_max_abs": float(grad_f.abs().max().item()),
        "grad_mean": float(grad_f.mean().item()),
    }


def compute_weight_metrics(weight: torch.Tensor, device: str | torch.device = "cpu") -> dict:
    if weight.numel() == 0:
        return {
            "weight_l2_norm": 0.0,
            "weight_mean": 0.0,
            "weight_std": 0.0,
            "weight_max_abs": 0.0,
            "weight_min": 0.0,
        }

    if _is_sharded(weight):
        m = _sharded_moments(weight)
        return {
            "weight_l2_norm": m["l2"],
            "weight_mean": m["mean"],
            "weight_std": m["std"],
            "weight_max_abs": m["max_abs"],
            "weight_min": m["min"],
        }
    w_f = _to_compute_device(_unshard(weight), device)
    return {
        "weight_l2_norm": float(w_f.norm(2).item()),
        "weight_mean": float(w_f.mean().item()),
        "weight_std": float(w_f.std(unbiased=False).item()),
        "weight_max_abs": float(w_f.abs().max().item()),
        "weight_min": float(w_f.min().item()),
    }


def compute_weight_histogram(
    weight: torch.Tensor, n_bins: int = 16, device: str | torch.device = "cpu"
) -> tuple[list[float], list[float]]:
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")

    if weight.numel() == 0:
        edges = [0.0] * (n_bins + 1)
        counts = [0.0] * n_bins
        return counts, edges

    if _is_sharded(weight):
        return _sharded_histogram(weight, n_bins)
    w_f = _to_compute_device(_unshard(weight), device)
    finite = w_f[torch.isfinite(w_f)]

    if finite.numel() == 0:
        edges = [0.0] * (n_bins + 1)
        counts = [0.0] * n_bins
        return counts, edges

    w_min = float(finite.min().item())
    w_max = float(finite.max().item())

    if w_min == w_max:
        counts_t = torch.zeros(n_bins)
        counts_t[n_bins // 2] = float(finite.numel())
        step = 1.0
        edges_list = [float(w_min + i * step - step * n_bins / 2) for i in range(n_bins + 1)]
        return [float(v) for v in counts_t.tolist()], edges_list

    # histc, not histogram: same equal-width bins over [min, max], but it also
    # has a CUDA kernel, so config.device="cuda" works.
    counts_t = torch.histc(finite, bins=n_bins, min=w_min, max=w_max)
    edges_t = torch.linspace(w_min, w_max, n_bins + 1)
    return [float(v) for v in counts_t.tolist()], [float(v) for v in edges_t.tolist()]
