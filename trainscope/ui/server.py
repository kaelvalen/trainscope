import asyncio
import json
import logging
import math
import re
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, cast
from urllib.parse import quote, unquote

import anyio
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel, Field

from trainscope.io.writer import read_arrow_rows_sync
from trainscope.ui.auth import auth_enabled, auth_middleware_factory, verify_request

logger = logging.getLogger("trainscope")


def _encode_layer_name(name: str) -> str:
    """Return the filesystem-safe, reversible encoding used by DiskWriter."""
    return quote(name, safe="")


def _decode_layer_name(filename: str) -> str:
    return unquote(Path(filename).stem)


def _safe_kl(p: list[float], q: list[float]) -> float:
    eps = 1e-10
    if not p or not q or len(p) != len(q):
        return 0.0
    sum_p = sum(p) + eps * len(p)
    sum_q = sum(q) + eps * len(q)
    kl = 0.0
    for pi, qi in zip(p, q):
        pi_n = (pi + eps) / sum_p
        qi_n = (qi + eps) / sum_q
        kl += pi_n * math.log(pi_n / qi_n)
    return max(0.0, kl)


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return sum((v - m) ** 2 for v in values) / len(values)


def _read_arrow_sync(path: Path) -> list[dict]:
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
        # File is being concurrently replaced or has an incomplete tail.
        logger.debug("Transient error reading Arrow file %s: %s", path, exc)
        return []


async def _read_arrow(path: Path) -> list[dict]:
    try:
        return await anyio.to_thread.run_sync(_read_arrow_sync, path)
    except Exception as exc:
        logger.warning("Transient error reading Arrow file %s: %s", path, exc)
        return []


async def _read_json(path: Path) -> Any:
    try:
        f = await anyio.open_file(path, "r")
    except Exception as exc:
        logger.exception("Failed to open JSON file %s", path)
        raise HTTPException(status_code=500, detail=f"Failed to open {path.name}") from exc
    try:
        content = await f.read()
    finally:
        await f.aclose()
    try:
        return json.loads(content)
    except Exception as exc:
        logger.exception("Failed to parse JSON file %s", path)
        raise HTTPException(status_code=500, detail=f"Invalid JSON in {path.name}") from exc


async def _read_json_optional(path: Path) -> Any:
    """Read JSON without raising HTTP exceptions (used by WebSocket)."""
    if not await anyio.to_thread.run_sync(path.exists):
        return None
    try:
        return await _read_json(path)
    except Exception:
        logger.exception("Failed to read optional JSON file %s", path)
        return None


async def _get_spike_steps(run_path: Path) -> set[int]:
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


async def _get_layers(run_path: Path) -> list[str]:
    layers_dir = run_path / "layers"
    if not await anyio.to_thread.run_sync(layers_dir.exists):
        return []
    files = await anyio.to_thread.run_sync(lambda: sorted(layers_dir.glob("*.arrow")))
    return [_decode_layer_name(f.name) for f in files]


class _TTLCache:
    """Simple bounded cache with per-entry TTL.

    Eviction is LRU (within the TTL window) and the cache is bounded by
    ``maxsize``.  Entries that exceed ``ttl`` seconds are treated as missing.
    """

    def __init__(self, maxsize: int, ttl: float):
        self._maxsize = max(maxsize, 1)
        self._ttl = ttl
        self._data: OrderedDict[str, tuple[Any, float]] = OrderedDict()

    def get(self, key: str) -> Any | None:
        now = time.monotonic()
        if key in self._data:
            value, expiry = self._data[key]
            if expiry > now:
                self._data.move_to_end(key)
                return value
            del self._data[key]
        return None

    def set(self, key: str, value: Any) -> None:
        now = time.monotonic()
        while len(self._data) >= self._maxsize:
            self._data.popitem(last=False)
        self._data[key] = (value, now + self._ttl)
        self._data.move_to_end(key)

    def clear(self) -> None:
        self._data.clear()


class DiffParams(BaseModel):
    step_a: int = Field(..., ge=0, description="First step to compare")
    step_b: int = Field(..., ge=0, description="Second step to compare")


class RankedParams(BaseModel):
    top_n: int = Field(
        8,
        ge=1,
        le=10_000,
        description="Maximum number of ranked layers to return",
    )


class SelectRunParams(BaseModel):
    name: str = Field(..., min_length=1, description="Run directory name to activate")


class CompareParams(BaseModel):
    runs: str = Field(..., description="Comma-separated run directory names to compare")


def _discover_run_dirs(root: Path) -> list[Path]:
    """Return run directories under ``root`` (sorted, deterministic)."""
    if not root.exists() or not root.is_dir():
        return []
    return sorted(
        (
            p
            for p in root.iterdir()
            if p.is_dir() and ((p / "meta.json").exists() or (p / "global.arrow").exists())
        ),
        key=lambda p: p.name.lower(),
    )


def create_app(run_path: str, runs_root: str | None = None) -> FastAPI:
    rp = Path(run_path).resolve()
    rr = Path(runs_root).resolve() if runs_root else None
    multi_run = rr is not None
    if multi_run:
        assert rr is not None
        discovered = _discover_run_dirs(rr)
        if discovered:
            rp = discovered[0]
    static_dir = Path(__file__).parent / "static"
    _cache = _TTLCache(maxsize=128, ttl=60.0)

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        mode = "multi-run" if multi_run else "single-run"
        logger.info("Starting TrainScope UI (%s) for run: %s", mode, rp)
        if multi_run:
            assert rr is not None
            logger.info("Runs root: %s (%d run dirs)", rr, len(_discover_run_dirs(rr)))
        if not rp.exists():
            logger.warning("Run path does not exist: %s", rp)
        elif not rp.is_dir():
            logger.warning("Run path is not a directory: %s", rp)
        yield

    app = FastAPI(title="TrainScope UI", lifespan=_lifespan)
    # Auth (TRAINSCOPE_API_KEY / TRAINSCOPE_BASIC_AUTH) uses header-based
    # credentials, not cookies, so allow_credentials must stay False: the
    # CORS spec forbids combining a wildcard origin with allow_credentials,
    # and enabling it would let any origin make credentialed requests.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(auth_middleware_factory())

    @app.middleware("http")
    async def disable_api_cache(request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, max-age=0"
        return response

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "run_path": str(rp),
            "exists": rp.exists(),
            "static_served": (static_dir / "index.html").exists(),
        }

    async def _read_run_summary(run_dir: Path) -> dict[str, Any]:
        """Build a lightweight summary of a single run from its metadata files."""
        meta = await _read_json_optional(run_dir / "meta.json") or {}
        manifest = await _read_json_optional(run_dir / "manifest.json") or {}
        tc_config = meta.get("trainscope_config") or {}

        steps = await _get_spike_steps(run_dir)

        last_row: dict[str, Any] = {}
        global_rows = await _read_arrow(run_dir / "global.arrow")
        if global_rows:
            last_row = global_rows[-1]

        return {
            "name": run_dir.name,
            "path": str(run_dir),
            "model_name": meta.get("model_name"),
            "model_config": meta.get("model_config"),
            "detector": tc_config.get("detector"),
            "start_time": meta.get("start_time"),
            "last_step": manifest.get("last_step"),
            "n_global_rows": manifest.get("n_global_rows"),
            "spike_count": len(steps),
            "last_loss": last_row.get("loss"),
            "last_grad_norm": last_row.get("grad_norm_before_clip"),
            "updated_at": manifest.get("updated_at"),
            "is_active": run_dir == rp,
        }

    @app.get("/api/runs")
    async def get_runs() -> list[dict[str, Any]]:
        """List every run under the root (or the single run) with meta summary."""
        if multi_run:
            assert rr is not None
            dirs = _discover_run_dirs(rr)
        else:
            dirs = [rp] if rp.is_dir() else []
        summaries = []
        for run_dir in dirs:
            summaries.append(await _read_run_summary(run_dir))
        summaries.sort(key=lambda s: (s["start_time"] or "", s["name"]))
        return summaries

    @app.post("/api/runs/select")
    async def select_run(params: SelectRunParams) -> dict[str, Any]:
        """Switch the active run in multi-run mode (no-op in single-run mode)."""
        nonlocal rp
        if not multi_run:
            if params.name != rp.name:
                raise HTTPException(
                    status_code=400,
                    detail="Single-run mode only exposes the current run",
                )
            return await _read_run_summary(rp)

        assert rr is not None
        candidates = [d for d in _discover_run_dirs(rr) if d.name == params.name]
        if not candidates:
            raise HTTPException(status_code=404, detail=f"Run '{params.name}' not found")
        rp = candidates[0]
        # Per-run caches (ranked layers, diff index) are keyed without a run
        # suffix; drop them so the next request rebuilds against the new run.
        _cache.clear()
        logger.info("Switched active run to: %s", rp)
        return await _read_run_summary(rp)

    def _flatten_config(meta: dict[str, Any]) -> dict[str, Any]:
        """Flatten trainscope_config + model_config into dot-path scalar fields."""
        flat: dict[str, Any] = {}

        def walk(prefix: str, value: Any) -> None:
            if isinstance(value, dict):
                for key, sub in value.items():
                    walk(f"{prefix}.{key}" if prefix else key, sub)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                flat[prefix] = value

        tc = meta.get("trainscope_config") or {}
        mc = meta.get("model_config") or {}
        # run_name/run_dir are per-run identities, not comparison dimensions.
        walk("config", {k: v for k, v in tc.items() if k not in {"run_name", "run_dir"}})
        walk("model", mc)
        return flat

    async def _find_divergence_step(
        series: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any] | None:
        """First step where the selected runs' loss curves durably separate.

        Uses the median pairwise loss gap over a warmup prefix as baseline and
        requires the gap to exceed 3x that baseline for ``min_run`` consecutive
        steps, so a single noisy step cannot trigger a false divergence.
        """
        steps_by_run = [{(row["step"]) for row in rows} for rows in series.values()]
        if not steps_by_run:
            return None
        common_steps = sorted(set.intersection(*steps_by_run))
        if len(common_steps) < 10:
            return None

        warmup_len = min(20, max(1, len(common_steps) // 5))
        warmup = common_steps[:warmup_len]
        step_index = {step: i for i, step in enumerate(common_steps)}

        loss_at: dict[str, list[float | None]] = {}
        for name, rows in series.items():
            by_step = {row["step"]: row.get("loss") for row in rows}
            loss_at[name] = [by_step.get(step) for step in common_steps]

        names = list(series.keys())
        baseline_gaps: list[float] = []
        for step in warmup:
            values = [loss_at[n][step_index[step]] for n in names]
            finite = [v for v in values if v is not None and math.isfinite(v)]
            if len(finite) >= 2:
                baseline_gaps.append(max(finite) - min(finite))
        if not baseline_gaps:
            return None
        baseline = sorted(baseline_gaps)[len(baseline_gaps) // 2]  # median
        threshold = max(3.0 * baseline, 1e-6)

        min_run = 3
        run_count = 0
        for step in common_steps[warmup_len:]:
            values = [loss_at[n][step_index[step]] for n in names]
            finite = [v for v in values if v is not None and math.isfinite(v)]
            if len(finite) >= 2 and (max(finite) - min(finite)) > threshold:
                run_count += 1
            else:
                run_count = 0
            if run_count >= min_run:
                step = common_steps[common_steps.index(step) - min_run + 1]
                return {
                    "step": step,
                    "baseline_gap": baseline,
                    "threshold": threshold,
                    "min_run": min_run,
                }
        return None

    @app.get("/api/compare")
    async def get_compare(
        params: Annotated[CompareParams, Depends()],
    ) -> dict[str, Any]:
        """Compare loss curves, config, and common causes across runs.

        Multi-run mode only (a single-run server has nothing to compare).
        Returns per-run loss series (optionally decimated), the first step at
        which the curves durably diverge, config fields that differ between
        runs, and a text summary of shared traits among runs with spikes.
        """
        if not multi_run:
            raise HTTPException(
                status_code=404,
                detail="Comparison requires multi-run mode (--runs)",
            )
        assert rr is not None
        names = [n.strip() for n in params.runs.split(",") if n.strip()]
        if len(names) < 2:
            raise HTTPException(
                status_code=400,
                detail="Select at least two runs to compare",
            )

        by_name = {d.name: d for d in _discover_run_dirs(rr)}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise HTTPException(status_code=404, detail=f"Run(s) not found: {', '.join(missing)}")

        summaries = await asyncio.gather(*(_read_run_summary(by_name[n]) for n in names))
        meta_by_name = {
            n: (await _read_json_optional(by_name[n] / "meta.json")) or {} for n in names
        }

        # Loss series per run, decimated so large runs stay cheap to plot.
        max_points = 2000
        series: dict[str, list[dict[str, Any]]] = {}
        for n in names:
            rows = await _read_arrow(by_name[n] / "global.arrow")
            if len(rows) > max_points:
                stride = len(rows) / max_points
                rows = [r for i, r in enumerate(rows) if i % max(1, int(stride)) == 0]
            series[n] = [{"step": r["step"], "loss": r.get("loss")} for r in rows]

        divergence = await _find_divergence_step(series)

        # Routing/addressing concentration series per run (MoE & addressor):
        # per step, the max share across all recorded blocks. This is the
        # architecture-aware detector's signal, so multi-run comparison can
        # ask "which runs concentrated" — not just "which runs blew up".
        concentration_series: dict[str, list[dict[str, Any]]] = {}
        for n in names:
            moe_rows = await _read_arrow(by_name[n] / "moe.arrow")
            by_step: dict[int, float] = {}
            for row in moe_rows:
                shares = row.get("shares") or []
                if shares:
                    step = row.get("step")
                    if step is not None:
                        by_step[step] = max(by_step.get(step, 0.0), max(shares))
            steps = sorted(by_step)
            if len(steps) > max_points:
                stride = len(steps) / max_points
                steps = [s for i, s in enumerate(steps) if i % max(1, int(stride)) == 0]
            concentration_series[n] = [{"step": s, "max_share": by_step[s]} for s in steps]

        # Config diff: fields whose value differs across the selected runs.
        flat_by_name = {n: _flatten_config(meta_by_name[n]) for n in names}
        all_keys = sorted({k for f in flat_by_name.values() for k in f})
        differing = [
            {"field": k, "values": {n: flat_by_name[n].get(k) for n in names}}
            for k in all_keys
            if len({flat_by_name[n].get(k) for n in names}) > 1
        ]

        # Common cause: shared trait of spiked runs absent from stable runs.
        common_cause: list[dict[str, Any]] = []
        spiked = [n for n in names if (summaries[names.index(n)].get("spike_count") or 0) > 0]
        stable = [n for n in names if n not in spiked]
        if spiked and stable:
            for entry in differing:
                k = cast(str, entry["field"])
                spiked_values = {flat_by_name[n].get(k) for n in spiked}
                stable_values = {flat_by_name[n].get(k) for n in stable}
                # Only report numeric thresholds and boolean flags; names and
                # free-text fields are noise.
                if not all(
                    isinstance(v, (int, float, bool)) for v in spiked_values | stable_values
                ):
                    continue
                if spiked_values == stable_values:
                    continue
                common_cause.append(
                    {
                        "field": k,
                        "spiked_value": next(iter(spiked_values)),
                        "stable_value": next(iter(stable_values)),
                    }
                )
                if len(common_cause) >= 5:
                    break

            # Runtime signal: routing/addressing concentration. A run whose
            # peak max-share crossed the detector's configured threshold (or
            # the family default for its detector) counts as "concentrated".
            # If every spiked run concentrated and no stable run did, report
            # it as a common cause.
            def _concentration_threshold(meta: dict[str, Any]) -> float | None:
                det = (meta.get("trainscope_config") or {}).get("detector") or {}
                if not isinstance(det, dict):
                    return None
                thr = det.get("threshold")
                if thr is not None:
                    return float(thr)
                name = det.get("name")
                if name == "expert_utilization_drift":
                    return 0.85
                if name == "addressor_concentration_drift":
                    return 0.6
                return None

            conc_peak = {
                n: max((row["max_share"] for row in concentration_series[n]), default=0.0)
                for n in names
            }
            thresholds = {n: _concentration_threshold(meta_by_name[n]) for n in names}
            if any(t is not None for t in thresholds.values()) and any(conc_peak.values()):
                spiked_conc = {n: conc_peak[n] for n in spiked}
                stable_conc = {n: conc_peak[n] for n in stable}
                # Only report when the groups separate cleanly.
                min_spiked = min(spiked_conc.values()) if spiked_conc else 0.0
                max_stable = max(stable_conc.values()) if stable_conc else 0.0
                if min_spiked > max_stable:
                    common_cause.append(
                        {
                            "field": "max routing concentration",
                            "spiked_value": round(min_spiked, 3),
                            "stable_value": round(max_stable, 3),
                        }
                    )

        return {
            "runs": [s["name"] for s in summaries],
            "summaries": summaries,
            "loss_series": series,
            "divergence": divergence,
            "config_diff": differing,
            "common_cause": common_cause,
            "concentration_series": concentration_series,
        }

    # ------------------------------------------------------------------ #
    # Run clustering (signal signatures)
    # ------------------------------------------------------------------ #
    def _signal_crossing(values: list[float]) -> bool:
        """True when ``values`` durably crosses baseline median + 3*MAD.

        Baseline is the first 40 samples (warmup), the crossing must hold for
        3 consecutive steps — the same robust rule used across all four
        verification experiments.
        """
        if len(values) < 50:
            return False
        base = values[:40]
        med = sorted(base)[len(base) // 2]
        mad = sorted(abs(v - med) for v in base)[len(base) // 2]
        threshold = med + 3.0 * mad
        run = 0
        for v in values[40:]:
            if math.isfinite(v) and v > threshold:
                run += 1
            else:
                run = 0
            if run >= 3:
                return True
        return False

    async def _run_signal_signature(run_dir: Path) -> dict[str, Any]:
        """Which early-warning signals fired in this run, and which fired first.

        Mirrors the v1.6.0 cascade ordering (kurtosis -> grad norm ->
        concentration -> loss CUSUM): each signal is tested with the same
        robust crossing rule, and the run's *first* signal anchors its
        cluster label.
        """
        signals: dict[str, bool] = {}
        order: list[str] = []
        global_rows = await _read_arrow(run_dir / "global.arrow")
        if global_rows:
            grad_norms = [r.get("grad_norm_before_clip") for r in global_rows]
            if any(v is not None for v in grad_norms):
                fired = _signal_crossing([v or 0.0 for v in grad_norms])
                signals["grad_norm"] = fired
                order.append("grad_norm")

            # Loss signal: a spike recorded by the detector (or a 10x loss jump
            # in the last row, which is what the objective explosion definition
            # uses across the experiments).
            spikes = await _get_spike_steps(run_dir)
            loss_fired = len(spikes) > 0
            if global_rows:
                last_loss = global_rows[-1].get("loss")
                if last_loss is not None:
                    base_mean = sum((r.get("loss") or 0.0) for r in global_rows[:40]) / min(
                        40, len(global_rows)
                    )
                    if base_mean > 0 and last_loss > 10.0 * base_mean:
                        loss_fired = True
            signals["loss"] = loss_fired
            order.append("loss")

        # Kurtosis from layer arrows (act_kurtosis, when recorded).
        kurtosis_values: list[float] = []
        layers_dir = run_dir / "layers"
        if await anyio.to_thread.run_sync(layers_dir.exists):
            files = await anyio.to_thread.run_sync(lambda: sorted(layers_dir.glob("*.arrow"))[:5])
            for arrow_file in files:
                rows = await _read_arrow(arrow_file)
                values = [r.get("act_kurtosis") for r in rows]
                if any(v is not None for v in values):
                    kurtosis_values = [v or 0.0 for v in values]
                    break
        if kurtosis_values:
            signals["kurtosis"] = _signal_crossing(kurtosis_values)
            order.append("kurtosis")

        # Concentration from moe.arrow.
        moe_rows = await _read_arrow(run_dir / "moe.arrow")
        if moe_rows:
            by_step: dict[int, float] = {}
            for row in moe_rows:
                shares = row.get("shares") or []
                step = row.get("step")
                if shares and step is not None:
                    by_step[step] = max(by_step.get(step, 0.0), max(shares))
            concentration_values = [by_step[s] for s in sorted(by_step)]
            signals["concentration"] = _signal_crossing(concentration_values)
            order.append("concentration")

        fired_names = [name for name in order if signals[name]]
        return {
            "signals": signals,
            "fired": fired_names,
            "first": fired_names[0] if fired_names else None,
        }

    @app.get("/api/cluster")
    async def get_clusters() -> dict[str, Any]:
        """Group runs by their early-warning signal signature.

        Runs with the same set of fired signals (and the same first signal)
        form a cluster, named after the v1.6.0 cascade position of the first
        signal: activation-led (kurtosis), gradient-led, routing-led
        (concentration), loss-led (CUSUM/spike), or no-signal (stable).
        """
        if not multi_run:
            raise HTTPException(status_code=404, detail="Clustering requires multi-run mode")
        assert rr is not None

        groups: dict[tuple[tuple[str, ...], str | None], dict[str, Any]] = {}
        unclustered: list[str] = []
        for run_dir in _discover_run_dirs(rr):
            sig = await _run_signal_signature(run_dir)
            if not sig["signals"]:
                unclustered.append(run_dir.name)
                continue
            key = (tuple(sig["fired"]), sig["first"])
            group = groups.setdefault(
                key, {"runs": [], "fired": sig["fired"], "first": sig["first"]}
            )
            group["runs"].append(run_dir.name)

        clusters: list[dict[str, Any]] = []
        for (fired, first), group in groups.items():
            labels = {
                "kurtosis": "activation-led",
                "grad_norm": "gradient-led",
                "concentration": "routing-led",
                "loss": "loss-led",
            }
            label = labels.get(first or "", "no-signal")
            clusters.append(
                {
                    "label": label,
                    "first_signal": first,
                    "fired_signals": fired,
                    "runs": sorted(group["runs"]),
                    "n_runs": len(group["runs"]),
                }
            )
        clusters.sort(key=lambda c: (-int(c["n_runs"]), str(c["label"])))
        return {"clusters": clusters, "unclustered": sorted(unclustered)}

    @app.get("/api/manifest")
    async def manifest() -> Any:
        path = rp / "manifest.json"
        if not await anyio.to_thread.run_sync(path.exists):
            return {}
        try:
            return await _read_json(path)
        except Exception:
            return {}

    @app.get("/api/meta")
    async def get_meta() -> Any:
        meta_file = rp / "meta.json"
        if not await anyio.to_thread.run_sync(meta_file.exists):
            return {
                "model_name": "Training in progress...",
                "trainscope_config": {"run_name": rp.name},
            }
        try:
            return await _read_json(meta_file)
        except Exception:
            return {
                "model_name": "Training in progress...",
                "trainscope_config": {"run_name": rp.name},
            }

    @app.get("/api/global")
    async def get_global() -> list[dict]:
        return await _read_arrow(rp / "global.arrow")

    @app.get("/api/layers")
    async def get_layers() -> list[str]:
        return await _get_layers(rp)

    @app.get("/api/moe")
    async def get_moe() -> list[dict]:
        """Per-block expert routing shares for the active run (MoE models)."""
        return await _read_arrow(rp / "moe.arrow")

    @app.get("/api/layers/ranked")
    async def get_layers_ranked(
        params: Annotated[RankedParams, Depends()],
    ) -> list[str]:
        cached = _cache.get("ranked_layers")
        if cached is not None:
            return cast(list[str], cached[: params.top_n])

        scored: list[tuple[str, float]] = []
        layers_dir = rp / "layers"
        if await anyio.to_thread.run_sync(layers_dir.exists):
            files = await anyio.to_thread.run_sync(lambda: sorted(layers_dir.glob("*.arrow")))
            for arrow_file in files:
                rows = await _read_arrow(arrow_file)
                layer_name = _decode_layer_name(arrow_file.name)
                grad_norms = [r["grad_l2_norm"] for r in rows if "grad_l2_norm" in r]
                scored.append((layer_name, _variance(grad_norms)))
        scored.sort(key=lambda x: x[1], reverse=True)
        ranked = [name for name, _ in scored]
        _cache.set("ranked_layers", ranked)
        return ranked[: params.top_n]

    @app.get("/api/layers/{layer_name:path}")
    async def get_layer(layer_name: str) -> list[dict]:
        encoded = _encode_layer_name(layer_name)
        path = rp / "layers" / f"{encoded}.arrow"
        if not await anyio.to_thread.run_sync(path.exists):
            raise HTTPException(status_code=404, detail=f"Layer '{layer_name}' not found")
        return await _read_arrow(path)

    @app.get("/api/spikes")
    async def get_spikes() -> list[dict[str, Any]]:
        steps = await _get_spike_steps(rp)
        return [{"step": step, "file": f"spike_step_{step}.arrow"} for step in sorted(steps)]

    @app.get("/api/spikes/{step}")
    async def get_spike(step: int) -> list[dict]:
        path = rp / "spikes" / f"spike_step_{step}.arrow"
        if not await anyio.to_thread.run_sync(path.exists):
            raise HTTPException(status_code=404, detail=f"Spike at step {step} not found")
        return await _read_arrow(path)

    @app.get("/api/spikes/{step}/layers")
    async def get_spike_layers(step: int) -> list[str]:
        layers_dir = rp / "spikes" / f"spike_step_{step}_layers"
        if not await anyio.to_thread.run_sync(layers_dir.exists):
            return []
        files = await anyio.to_thread.run_sync(lambda: sorted(layers_dir.glob("*.arrow")))
        return [_decode_layer_name(f.name) for f in files]

    @app.get("/api/spikes/{step}/layers/{layer_name:path}")
    async def get_spike_layer(step: int, layer_name: str) -> list[dict]:
        encoded = _encode_layer_name(layer_name)
        path = rp / "spikes" / f"spike_step_{step}_layers" / f"{encoded}.arrow"
        if not await anyio.to_thread.run_sync(path.exists):
            raise HTTPException(
                status_code=404,
                detail=f"Layer '{layer_name}' not in spike {step}",
            )
        return await _read_arrow(path)

    async def _get_diff_index() -> dict[str, dict[int, list[float]]]:
        cached = _cache.get("diff_index")
        if cached is not None:
            return cast(dict[str, dict[int, list[float]]], cached)

        index: dict[str, dict[int, list[float]]] = {}
        layers_dir = rp / "layers"
        if await anyio.to_thread.run_sync(layers_dir.exists):
            files = await anyio.to_thread.run_sync(lambda: sorted(layers_dir.glob("*.arrow")))
            for arrow_file in files:
                rows = await _read_arrow(arrow_file)
                layer_name = _decode_layer_name(arrow_file.name)
                index[layer_name] = {r["step"]: r.get("hist_counts") or [] for r in rows}
        _cache.set("diff_index", index)
        return index

    @app.get("/api/diff")
    async def get_diff(
        params: Annotated[DiffParams, Depends()],
    ) -> list[dict[str, Any]]:
        index = await _get_diff_index()
        result: list[dict[str, Any]] = []
        for layer_name, step_map in index.items():
            counts_a = step_map.get(params.step_a)
            counts_b = step_map.get(params.step_b)
            if not counts_a or not counts_b:
                continue
            result.append(
                {
                    "layer": layer_name,
                    "kl_divergence": _safe_kl(counts_a, counts_b),
                }
            )
        result.sort(key=lambda x: x["kl_divergence"], reverse=True)
        return result

    # ------------------------------------------------------------------ #
    # WebSocket streaming
    # ------------------------------------------------------------------ #
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        # AuthMiddleware is a BaseHTTPMiddleware and never runs for WebSocket
        # connections (Starlette limitation), so auth must be re-checked here.
        if auth_enabled() and not verify_request(websocket):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        try:
            meta = await _read_json_optional(rp / "meta.json") or {}
            manifest = await _read_json_optional(rp / "manifest.json") or {}
            await websocket.send_json(
                {
                    "type": "meta",
                    "payload": {
                        "run_path": str(rp),
                        **meta,
                        "manifest": manifest,
                    },
                }
            )

            # Poll continuously rather than branching once on "is this run live
            # right now" — a run that hasn't written global.arrow yet (e.g. the
            # browser connected the instant the server started, before the
            # first training step landed) must not get stuck forever on a
            # heartbeat-only path once it does go live.
            last_global_len = 0
            last_spikes: set[int] = set()
            last_layers: set[str] = set()
            while True:
                global_rows = await _read_arrow(rp / "global.arrow")
                if last_global_len == 0 and global_rows:
                    await websocket.send_json({"type": "global", "payload": global_rows})
                    last_global_len = len(global_rows)
                elif len(global_rows) > last_global_len:
                    new_rows = global_rows[last_global_len:]
                    await websocket.send_json({"type": "global_delta", "payload": new_rows})
                    last_global_len = len(global_rows)

                spikes = await _get_spike_steps(rp)
                new_spikes = spikes - last_spikes
                if new_spikes:
                    for step in sorted(new_spikes):
                        await websocket.send_json({"type": "spike", "payload": {"step": step}})
                    last_spikes = spikes

                layers = set(await _get_layers(rp))
                if layers != last_layers:
                    await websocket.send_json({"type": "layers", "payload": sorted(layers)})
                    last_layers = layers

                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            logger.debug("WebSocket client disconnected from %s", rp)
        except Exception:
            logger.exception("WebSocket error for run %s", rp)
            await websocket.close(code=1011)

    @app.get("/{full_path:path}")
    async def serve_static(full_path: str) -> Response:
        target = static_dir / full_path
        if full_path and target.exists() and target.is_file():
            return FileResponse(str(target))

        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))

        return HTMLResponse(
            content="""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>TrainScope UI</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; background: #0f172a; color: #f8fafc; display: flex; height: 100vh; align-items: center; justify-content: center; margin: 0; }
    .card { background: #1e293b; padding: 2.5rem; border-radius: 1rem; border: 1px solid #334155; max-width: 520px; text-align: center; box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5); }
    h1 { font-size: 1.5rem; margin-bottom: 1rem; color: #38bdf8; }
    p { color: #94a3b8; line-height: 1.6; font-size: 0.95rem; }
    code { background: #0f172a; padding: 0.3rem 0.6rem; border-radius: 0.375rem; color: #f43f5e; font-family: monospace; font-size: 0.9em; border: 1px solid #334155; }
  </style>
</head>
<body>
  <div class="card">
    <h1>TrainScope UI</h1>
    <p>React UI assets were not found in <code>trainscope/ui/static</code>.</p>
    <p>If you are running from a local git repository, build the frontend:</p>
    <p><code>cd frontend && npm install && npm run build</code></p>
  </div>
</body>
</html>"""
        )

    return app


def start_server(
    run_path: str,
    host: str = "127.0.0.1",
    port: int = 7007,
    log_level: str = "info",
    runs_root: str | None = None,
) -> None:
    import uvicorn

    uvicorn.run(
        create_app(run_path, runs_root=runs_root),
        host=host,
        port=port,
        log_level=log_level.lower(),
    )
