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


def _activation_stats(act: torch.Tensor, device: str | torch.device) -> torch.Tensor:
    """``[mean, std, min, max, kurtosis, median]`` of a non-empty activation, left
    on ``device`` so callers can read many of them back in one transfer."""
    act_f = _to_compute_device(act, device)
    std, mean = torch.std_mean(act_f, correction=0)
    lo, hi = torch.aminmax(act_f)
    kurtosis = (((act_f - mean) / std) ** 4).mean() - 3.0
    return torch.stack([mean, std, lo, hi, kurtosis, act_f.median()])


def _activation_metrics_from_stats(vals: list[float]) -> dict:
    mean, std, lo, hi, kurtosis, median = vals
    return {
        "act_mean": mean,
        "act_std": std,
        "act_max_abs": max(-lo, hi),
        "act_kurtosis": 0.0 if std == 0.0 else kurtosis,
        "act_min": lo,
        "act_max": hi,
        "act_median": median,
    }


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
    return _activation_metrics_from_stats(_activation_stats(act, device).tolist())


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
    # One host transfer for all statistics: each .item() is a device sync.
    l2_norm, max_abs, mean = torch.stack(
        [grad_f.norm(2), grad_f.abs().max(), grad_f.mean()]
    ).tolist()
    # A finite norm proves every element is finite (NaN/Inf propagate), so the
    # elementwise count is only needed when the norm is not.
    if math.isfinite(l2_norm):
        nan_inf_ratio = 0.0
    else:
        nan_inf_ratio = float((~torch.isfinite(grad_f)).sum().item()) / grad_f.numel()

    return {
        "grad_l2_norm": l2_norm,
        "grad_nan_inf_ratio": float(nan_inf_ratio),
        "grad_max_abs": max_abs,
        "grad_mean": mean,
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
    l2_norm, mean, std, max_abs, min_v = torch.stack(
        [w_f.norm(2), w_f.mean(), w_f.std(unbiased=False), w_f.abs().max(), w_f.min()]
    ).tolist()
    return {
        "weight_l2_norm": l2_norm,
        "weight_mean": mean,
        "weight_std": std,
        "weight_max_abs": max_abs,
        "weight_min": min_v,
    }


def compute_param_metrics(
    weights: list[torch.Tensor],
    grads: list[torch.Tensor | None],
    devices: list[str | torch.device],
) -> list[tuple[dict, dict]]:
    """``(compute_weight_metrics, compute_gradient_metrics)`` for many tensors.

    On accelerators, tensors sharing a compute device go through
    ``_fused_param_metrics`` (one host transfer instead of a device sync per
    tensor). CPU has no sync to save and its foreach ops are plain loops, so
    CPU, DTensors and empty tensors take the per-tensor path.
    """
    out: list[tuple[dict, dict] | None] = [None] * len(weights)
    groups: dict[str, list[int]] = {}
    for i, (w, g, dev) in enumerate(zip(weights, grads, devices)):
        if (
            torch.device(dev).type == "cpu"
            or _is_dtensor(w)
            or w.numel() == 0
            or (g is not None and (_is_dtensor(g) or g.numel() == 0))
        ):
            out[i] = (compute_weight_metrics(w, dev), compute_gradient_metrics(g, dev))
        else:
            groups.setdefault(str(dev), []).append(i)
    for idx in groups.values():
        fused = _fused_param_metrics(
            [weights[i] for i in idx], [grads[i] for i in idx], devices[idx[0]]
        )
        for i, metrics in zip(idx, fused):
            out[i] = metrics
    return out  # type: ignore[return-value]


def _fused_param_metrics(
    weights: list[torch.Tensor], grads: list[torch.Tensor | None], dev: str | torch.device
) -> list[tuple[dict, dict]]:
    """Two passes per tensor (std_mean, aminmax), read back with one ``.tolist()``.

    L2 and max-abs follow from them: l2^2 = n * (std^2 + mean^2), max_abs =
    max(-min, max). A non-finite statistic (overflow/NaN) falls back to the
    exact per-tensor functions so inf norms and nan/inf ratios stay as before.
    """
    tensors = [_to_compute_device(t, dev) for t in [*weights, *(g for g in grads if g is not None)]]
    scalars = []
    for t in tensors:
        scalars += [*torch.std_mean(t, correction=0), *torch.aminmax(t)]
    vals = torch.stack(scalars).tolist()

    def stats(j: int) -> tuple[float, float, float, float, float] | None:
        std, mean, lo, hi = vals[4 * j : 4 * j + 4]
        if not all(math.isfinite(v) for v in (std, mean, lo, hi)):
            return None
        return math.sqrt(tensors[j].numel() * (std * std + mean * mean)), mean, std, lo, hi

    out = []
    g_pos = len(weights)
    for j, (w, g) in enumerate(zip(weights, grads)):
        ws = stats(j)
        weight_metrics = (
            compute_weight_metrics(w, dev)
            if ws is None
            else {
                "weight_l2_norm": ws[0],
                "weight_mean": ws[1],
                "weight_std": ws[2],
                "weight_max_abs": max(-ws[3], ws[4]),
                "weight_min": ws[3],
            }
        )
        gs = None
        if g is not None:
            gs = stats(g_pos)
            g_pos += 1
        grad_metrics = (
            compute_gradient_metrics(g, dev)
            if gs is None
            else {
                "grad_l2_norm": gs[0],
                "grad_nan_inf_ratio": 0.0,
                "grad_max_abs": max(-gs[3], gs[4]),
                "grad_mean": gs[1],
            }
        )
        out.append((weight_metrics, grad_metrics))
    return out


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
