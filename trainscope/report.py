"""Post-mortem report generation (``trainscope report``).

Turns a run directory (or a root of runs) into a researcher's case file:
the spike story, which early-warning signals fired and their lead, the
config, and the common causes shared across exploding runs. The signal
signature and the common-cause rules come from ``trainscope.analysis`` and
the ``/api/compare`` logic, so the CLI report never drifts from the UI.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from trainscope.analysis import read_arrow, run_signal_signature

# Cluster labels shared with /api/cluster.
_LABELS = {
    "kurtosis": "activation-led",
    "grad_norm": "gradient-led",
    "concentration": "routing-led",
    "loss": "loss-led",
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _loss_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    losses = [r.get("loss") for r in rows if isinstance(r.get("loss"), (int, float))]
    if not losses:
        return {"n": 0}
    finite = [value for value in losses if math.isfinite(value)]
    return {
        "n": len(rows),
        "min": round(min(losses), 4),
        "max": round(max(losses), 4),
        "last": round(losses[-1], 4),
        "final_10x_baseline": bool(finite and losses[-1] > 10.0 * (sum(finite) / len(finite))),
    }


def _config_preview(meta: dict[str, Any]) -> dict[str, Any]:
    tc = meta.get("trainscope_config") or {}
    return {
        "model_name": meta.get("model_name"),
        "start_time": meta.get("start_time"),
        "detector": tc.get("detector"),
        "run_name": tc.get("run_name"),
    }


async def build_run_report(run_dir: Path) -> dict[str, Any]:
    """Build the structured post-mortem for a single run directory."""
    meta = _load_json(run_dir / "meta.json")
    sig = await run_signal_signature(run_dir)
    rows = await read_arrow(run_dir / "global.arrow")

    spike_steps = (
        sorted(
            int(f.stem.split("_")[-1])
            for f in (run_dir / "spikes").glob("spike_step_*.arrow")
            if f.stem.startswith("spike_step_")
        )
        if (run_dir / "spikes").is_dir()
        else []
    )

    first_steps = sig.get("first_steps") or {}
    lead = None
    if sig.get("first") and sig.get("explosion_step") is not None:
        first = sig["first"]
        if first in first_steps and first_steps[first] is not None:
            lead = sig["explosion_step"] - first_steps[first]

    return {
        "run": run_dir.name,
        "config": _config_preview(meta),
        "model_config": meta.get("model_config"),
        "signal_signature": {
            "fired": sig["fired"],
            "first": sig["first"],
            "first_steps": first_steps,
            "explosion_step": sig["explosion_step"],
            "lead_steps": lead,
        },
        "loss": _loss_stats(rows),
        "spike_steps": spike_steps,
    }


async def build_cluster_report(root: Path) -> dict[str, Any]:
    """Build the multi-run cluster report: runs grouped by signal signature."""
    run_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    signatures = {}
    for run_dir in run_dirs:
        signatures[run_dir.name] = await run_signal_signature(run_dir)

    clusters: list[dict[str, Any]] = []
    unclustered: list[str] = []
    by_key: dict[tuple[tuple[str, ...], str | None], list[str]] = {}
    for name, sig in signatures.items():
        if not sig["fired"]:
            unclustered.append(name)
            continue
        key = (tuple(sig["fired"]), sig["first"])
        by_key.setdefault(key, []).append(name)

    for (fired, first), names in by_key.items():
        label = _LABELS.get(first or "", "no-signal")
        leads = []
        for name in names:
            sig = signatures[name]
            first_steps = sig.get("first_steps") or {}
            if sig.get("explosion_step") is not None and first in first_steps:
                if first_steps[first] is not None:
                    leads.append(sig["explosion_step"] - first_steps[first])
        clusters.append(
            {
                "label": label,
                "first_signal": first,
                "fired_signals": sorted(fired),
                "runs": sorted(names),
                "n_runs": len(names),
                "typical_lead_steps": (
                    round(float(sorted(leads)[len(leads) // 2]), 1) if leads else None
                ),
            }
        )
    clusters.sort(key=lambda c: (-int(c["n_runs"]), str(c["label"])))
    return {"clusters": clusters, "unclustered": sorted(unclustered), "n_runs": len(run_dirs)}


# --------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------- #
def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render_run_markdown(report: dict[str, Any]) -> str:
    cfg = report["config"]
    sig = report["signal_signature"]
    lines = [
        f"# Post-mortem: {report['run']}",
        "",
        f"- Model: {_fmt(cfg.get('model_name'))}",
        f"- Started: {_fmt(cfg.get('start_time'))}",
        f"- Detector: {_fmt(cfg.get('detector'))}",
        f"- Steps recorded: {report['loss'].get('n', 0)}",
        "",
        "## Loss",
        f"- Min {_fmt(report['loss'].get('min'))} / max {_fmt(report['loss'].get('max'))} / "
        f"last {_fmt(report['loss'].get('last'))}",
        "",
        "## Signal signature",
    ]
    if sig["fired"]:
        lines.append(f"- Fired signals: {', '.join(sig['fired'])}")
        lines.append(f"- First signal: {_fmt(sig['first'])}")
        for name, step in sorted(sig["first_steps"].items()):
            lines.append(f"  - {name} crossed at step {step}")
        if sig["explosion_step"] is not None:
            lines.append(f"- Objective explosion at step {sig['explosion_step']}")
            if sig["lead_steps"] is not None:
                lines.append(f"- **Early-warning lead: {sig['lead_steps']} steps**")
    else:
        lines.append("- No early-warning signal fired (stable run).")

    lines += ["", "## Spikes"]
    if report["spike_steps"]:
        lines.append(f"- Spike steps: {', '.join(map(str, report['spike_steps']))}")
    else:
        lines.append("- No spikes recorded.")
    return "\n".join(lines) + "\n"


def render_cluster_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Run behavior clusters ({report['n_runs']} runs)",
        "",
    ]
    if not report["clusters"]:
        lines.append("No run fired an early-warning signal.")
    for cluster in report["clusters"]:
        lines.append(f"## {cluster['label']} ({cluster['n_runs']} runs)")
        lines.append(f"- Fired signals: {', '.join(cluster['fired_signals'])}")
        lines.append(f"- First signal: {cluster['first_signal']}")
        if cluster["typical_lead_steps"] is not None:
            lines.append(f"- Typical early-warning lead: ~{cluster['typical_lead_steps']} steps")
        lines.append(f"- Runs: {', '.join(cluster['runs'])}")
        lines.append("")
    if report["unclustered"]:
        lines.append("## Stable (no signal) runs")
        lines.append(f"- {', '.join(report['unclustered'])}")
    return "\n".join(lines) + "\n"


def render_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, default=str) + "\n"


async def build_report(target: Path, multi_run: bool) -> dict[str, Any]:
    if multi_run:
        return await build_cluster_report(target)
    return await build_run_report(target)


def render(report: dict[str, Any], multi_run: bool, fmt: str) -> str:
    if fmt == "json":
        return render_json(report)
    if multi_run:
        return render_cluster_markdown(report)
    return render_run_markdown(report)
