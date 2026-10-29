"""Per-layer metric matrix for the UI's layer heatmap.

Each layer's series is bucketed onto a shared step axis (max per bucket) and
scored as a robust z against the layer's own pre-spike baseline, so "which
layer heated up first" can be read across rows.
"""

import math
import re
from statistics import median
from typing import Any

_Z_MAX = 6.0
_MAD_TO_SIGMA = 1.4826


def natural_key(name: str) -> list:
    """Sort key that orders ``blocks.2`` before ``blocks.10``."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def _robust_z(x: float, center: float, mad: float) -> float:
    if mad == 0.0:
        return 0.0 if x <= center else _Z_MAX
    return min(max((x - center) / (_MAD_TO_SIGMA * mad), 0.0), _Z_MAX)


def build_layer_matrix(
    series: dict[str, list[tuple[int, float | None]]],
    *,
    buckets: int = 600,
    start: int | None = None,
    end: int | None = None,
    first_spike: int | None = None,
) -> dict[str, Any]:
    layers = sorted(series, key=natural_key)
    all_steps = sorted({step for points in series.values() for step, _ in points})
    lo = all_steps[0] if all_steps and start is None else start
    hi = all_steps[-1] if all_steps and end is None else end
    in_range = [s for s in all_steps if lo is not None and hi is not None and lo <= s <= hi]
    if not in_range:
        blank: list[list[float | None]] = [[] for _ in layers]
        return {
            "layers": layers,
            "bucket_starts": [],
            "values": blank,
            "z": blank,
            "baseline": None,
        }
    assert lo is not None and hi is not None

    span = hi - lo + 1
    k = min(buckets, len(in_range))
    bucket_starts = [lo + (i * span) // k for i in range(k)]

    baseline_steps = [s for s in all_steps if first_spike is not None and s < first_spike]
    if not baseline_steps:
        baseline_steps = all_steps[: max(1, len(all_steps) // 2)]
    b_lo, b_hi = baseline_steps[0], baseline_steps[-1]

    values: list[list[float | None]] = []
    zs: list[list[float | None]] = []
    for layer in layers:
        row: list[float | None] = [None] * k
        base: list[float] = []
        for step, value in series[layer]:
            if value is None or not math.isfinite(value):
                continue
            if b_lo <= step <= b_hi:
                base.append(value)
            if lo <= step <= hi:
                i = min(k - 1, ((step - lo) * k) // span)
                current = row[i]
                row[i] = value if current is None else max(current, value)
        values.append(row)
        if base:
            center = median(base)
            mad = median(abs(v - center) for v in base)
            zs.append([None if v is None else _robust_z(v, center, mad) for v in row])
        else:
            zs.append([None] * k)

    return {
        "layers": layers,
        "bucket_starts": bucket_starts,
        "values": values,
        "z": zs,
        "baseline": {"start": b_lo, "end": b_hi},
    }
