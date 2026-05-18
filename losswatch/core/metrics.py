import torch


def compute_activation_metrics(act: torch.Tensor) -> dict:
    if act.numel() == 0:
        return {
            "act_mean": 0.0,
            "act_std": 0.0,
            "act_max_abs": 0.0,
            "act_kurtosis": 0.0,
        }

    act_f = act.detach().float().flatten()
    mean = act_f.mean()
    std = act_f.std(unbiased=False)
    max_abs = act_f.abs().max()

    if std.item() == 0.0:
        kurtosis = 0.0
    else:
        normalized = (act_f - mean) / std
        kurtosis = float((normalized ** 4).mean().item()) - 3.0

    return {
        "act_mean": float(mean.item()),
        "act_std": float(std.item()),
        "act_max_abs": float(max_abs.item()),
        "act_kurtosis": kurtosis,
    }


def compute_gradient_metrics(grad: torch.Tensor | None) -> dict:
    if grad is None or grad.numel() == 0:
        return {
            "grad_l2_norm": 0.0,
            "grad_nan_inf_ratio": 0.0,
        }

    grad_f = grad.detach().float().flatten()
    l2_norm = float(grad_f.norm(2).item())
    nan_inf_count = float((~torch.isfinite(grad_f)).sum().item())
    nan_inf_ratio = nan_inf_count / grad_f.numel()

    return {
        "grad_l2_norm": l2_norm,
        "grad_nan_inf_ratio": float(nan_inf_ratio),
    }


def compute_weight_metrics(weight: torch.Tensor) -> dict:
    if weight.numel() == 0:
        return {"weight_l2_norm": 0.0}

    w_f = weight.detach().float()
    return {"weight_l2_norm": float(w_f.norm(2).item())}


def compute_weight_histogram(
    weight: torch.Tensor, n_bins: int = 16
) -> tuple[list[float], list[float]]:
    if weight.numel() == 0:
        edges = [0.0] * (n_bins + 1)
        counts = [0.0] * n_bins
        return counts, edges

    w_f = weight.detach().float().flatten()
    w_min = w_f.min().item()
    w_max = w_f.max().item()

    if w_min == w_max:
        counts_t = torch.zeros(n_bins)
        counts_t[n_bins // 2] = float(w_f.numel())
        step = 1.0
        edges_list = [float(w_min + i * step - step * n_bins / 2) for i in range(n_bins + 1)]
        return counts_t.tolist(), edges_list

    counts_t, edges_t = torch.histogram(w_f, bins=n_bins)
    return [float(v) for v in counts_t.tolist()], [float(v) for v in edges_t.tolist()]
