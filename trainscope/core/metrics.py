import torch


def _to_compute_device(tensor: torch.Tensor, device: str | torch.device) -> torch.Tensor:
    """Move a flattened float copy of ``tensor`` to the requested compute device."""
    flat = tensor.detach().float().flatten()
    if str(flat.device) == str(device):
        return flat
    return flat.to(device)


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

    grad_f = _to_compute_device(grad, device)
    l2_norm = float(grad_f.norm(2).item())
    nan_inf_count = float((~torch.isfinite(grad_f)).sum().item())
    nan_inf_ratio = nan_inf_count / grad_f.numel()

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

    w_f = _to_compute_device(weight, device)
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

    w_f = _to_compute_device(weight, device)
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

    counts_t, edges_t = torch.histogram(finite, bins=n_bins)
    return [float(v) for v in counts_t.tolist()], [float(v) for v in edges_t.tolist()]
