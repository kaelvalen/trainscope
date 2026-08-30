"""Signal-signature analysis shared by the UI server and the CLI report.

The early-warning signal signature of a run — which of the four verified
signals fired, which fired first (chronologically), and where the objective
explosion happened — is used both by ``/api/cluster`` (Runs view clustering)
and by ``trainscope report``. Keeping it in one module guarantees the two
consumers never drift apart on the crossing rule or the explosion
definition, both of which mirror the v1.6.0 verification experiments.
"""

import logging
import math
import re
from pathlib import Path
from typing import Any

import anyio

from trainscope.io.writer import read_arrow_rows_sync

logger = logging.getLogger("trainscope")


def read_arrow_sync(path: Path) -> list[dict]:
    """Read an Arrow file into a list of row dicts.

    Handles both on-disk formats (legacy IPC files and the append-only IPC
    streams written since 0.7.0) and gracefully handles in-flight writes when
    a file tail is momentarily incomplete.
    """
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        return read_arrow_rows_sync(path)
    except Exception as exc:
        logger.debug("Transient error reading Arrow file %s: %s", path, exc)
        return []


async def read_arrow(path: Path) -> list[dict]:
    try:
        return await anyio.to_thread.run_sync(read_arrow_sync, path)
    except Exception as exc:
        logger.warning("Transient error reading Arrow file %s: %s", path, exc)
        return []


async def get_spike_steps(run_path: Path) -> set[int]:
    spikes_dir = run_path / "spikes"
    if not await anyio.to_thread.run_sync(spikes_dir.exists):
        return set()
    files = await anyio.to_thread.run_sync(lambda: sorted(spikes_dir.glob("spike_step_*.arrow")))
    steps: set[int] = set()
    for f in files:
        match = re.search(r"spike_step_(\d+)\.arrow", f.name)
        if match:
            steps.add(int(match.group(1)))
    return steps


def first_crossing_step(values: list[float]) -> int | None:
    """Step index where ``values`` durably crosses baseline median + 3*MAD.

    Baseline is the first 40 samples (warmup), the crossing must hold for
    3 consecutive steps — the same robust rule used across all four
    verification experiments. Returns the step index of the first crossing
    (i.e. the step where the 3-step run began), or None when the signal never
    crosses. The step index — not a boolean — is what lets clustering order
    signals chronologically.
    """
    if len(values) < 50:
        return None
    base = values[:40]
    med = sorted(base)[len(base) // 2]
    mad = sorted(abs(v - med) for v in base)[len(base) // 2]
    threshold = med + 3.0 * mad
    run = 0
    for i in range(40, len(values)):
        v = values[i]
        if math.isfinite(v) and v > threshold:
            run += 1
        else:
            run = 0
        if run >= 3:
            return i - run + 1
    return None


async def run_signal_signature(run_dir: Path) -> dict[str, Any]:
    """Which early-warning signals fired in this run, and which fired first.

    Mirrors the v1.6.0 cascade ordering (kurtosis -> grad norm ->
    concentration -> loss CUSUM): each signal is tested with the same robust
    crossing rule, and the run's *first* signal anchors its cluster label.
    The first signal is the one with the earliest crossing step —
    chronological, not code-order.
    """
    fired: dict[str, int] = {}
    global_rows = await read_arrow(run_dir / "global.arrow")
    if global_rows:
        grad_norms = [r.get("grad_norm_before_clip") for r in global_rows]
        if any(v is not None for v in grad_norms):
            step = first_crossing_step([v or 0.0 for v in grad_norms])
            if step is not None:
                fired["grad_norm"] = step

        # Loss signal: a spike recorded by the detector (or a 10x loss jump in
        # the last row, which is what the objective explosion definition uses
        # across the experiments).
        spikes = await get_spike_steps(run_dir)
        if spikes:
            fired["loss"] = min(spikes)
        elif global_rows:
            last_loss = global_rows[-1].get("loss")
            last_step = global_rows[-1].get("step")
            if last_loss is not None and last_step is not None:
                base_mean = sum((r.get("loss") or 0.0) for r in global_rows[:40]) / min(
                    40, len(global_rows)
                )
                if base_mean > 0 and last_loss > 10.0 * base_mean:
                    fired["loss"] = int(last_step)

    # Kurtosis from layer arrows (act_kurtosis, when recorded).
    kurtosis_values: list[float] = []
    layers_dir = run_dir / "layers"
    if await anyio.to_thread.run_sync(layers_dir.exists):
        files = await anyio.to_thread.run_sync(lambda: sorted(layers_dir.glob("*.arrow"))[:5])
        for arrow_file in files:
            rows = await read_arrow(arrow_file)
            values = [r.get("act_kurtosis") for r in rows]
            if any(v is not None for v in values):
                kurtosis_values = [v or 0.0 for v in values]
                break
    if kurtosis_values:
        step = first_crossing_step(kurtosis_values)
        if step is not None:
            fired["kurtosis"] = step

    # Concentration from moe.arrow.
    moe_rows = await read_arrow(run_dir / "moe.arrow")
    if moe_rows:
        by_step: dict[int, float] = {}
        for row in moe_rows:
            shares = row.get("shares") or []
            step = row.get("step")
            if shares and step is not None:
                by_step[step] = max(by_step.get(step, 0.0), max(shares))
        concentration_values = [by_step[s] for s in sorted(by_step)]
        step = first_crossing_step(concentration_values)
        if step is not None:
            fired["concentration"] = step

    # Chronological order: earliest crossing step first.
    fired_names = sorted(fired, key=lambda name: fired[name])

    # Objective explosion step (same definition as the verification
    # experiments): first non-finite loss or a loss > 10x baseline mean.
    explosion_step = None
    if global_rows:
        base_mean = sum((r.get("loss") or 0.0) for r in global_rows[:40]) / min(
            40, len(global_rows)
        )
        for row in global_rows:
            loss = row.get("loss")
            step = row.get("step")
            if step is None:
                continue
            if (
                loss is None
                or not math.isfinite(loss)
                or (base_mean > 0 and loss > 10.0 * base_mean)
            ):
                explosion_step = int(step)
                break

    return {
        "fired": fired_names,
        "first": fired_names[0] if fired_names else None,
        "first_steps": fired,
        "explosion_step": explosion_step,
    }


__all__ = [
    "read_arrow",
    "read_arrow_sync",
    "get_spike_steps",
    "first_crossing_step",
    "run_signal_signature",
]
