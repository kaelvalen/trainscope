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
import pyarrow.ipc as ipc
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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
    """Read an Arrow IPC file into a list of row dicts.

    Uses a context manager so the reader is closed explicitly even on older
    PyArrow versions where ``RecordBatchFileReader.close()`` is not exposed.
    """
    if not path.exists():
        return []
    with ipc.open_file(str(path)) as reader:
        table = reader.read_all()
    raw = table.to_pydict()
    if not raw:
        return []
    n = len(next(iter(raw.values())))
    keys = list(raw.keys())
    rows = []
    for i in range(n):
        row = {}
        for k in keys:
            val = raw[k][i]
            if hasattr(val, "as_py"):
                val = val.as_py()
            row[k] = val
        rows.append(row)
    return rows


async def _read_arrow(path: Path) -> list[dict]:
    try:
        return await anyio.to_thread.run_sync(_read_arrow_sync, path)
    except Exception as exc:
        logger.exception("Failed to read Arrow file %s", path)
        raise HTTPException(status_code=500, detail=f"Failed to read {path.name}") from exc


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


def create_app(run_path: str) -> FastAPI:
    rp = Path(run_path).resolve()
    static_dir = Path(__file__).parent / "static"
    fallback_path = Path(__file__).parent / "fallback.html"
    _fallback_html = (
        fallback_path.read_text(encoding="utf-8")
        if fallback_path.exists()
        else "<html><body><h1>TrainScope</h1><p>UI build not found.</p></body></html>"
    )
    _cache = _TTLCache(maxsize=128, ttl=60.0)

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        logger.info("Starting TrainScope UI for run: %s", rp)
        if not rp.exists():
            logger.warning("Run path does not exist: %s", rp)
        elif not rp.is_dir():
            logger.warning("Run path is not a directory: %s", rp)
        yield

    app = FastAPI(title="TrainScope UI", lifespan=_lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "run_path": str(rp),
            "exists": rp.exists(),
            "static_served": (static_dir / "index.html").exists(),
        }

    @app.get("/api/manifest")
    async def manifest() -> Any:
        path = rp / "manifest.json"
        if not await anyio.to_thread.run_sync(path.exists):
            raise HTTPException(status_code=404, detail="manifest.json not found")
        return await _read_json(path)

    @app.get("/api/meta")
    async def get_meta() -> Any:
        meta_file = rp / "meta.json"
        if not await anyio.to_thread.run_sync(meta_file.exists):
            raise HTTPException(status_code=404, detail="meta.json not found")
        return await _read_json(meta_file)

    @app.get("/api/global")
    async def get_global() -> list[dict]:
        return await _read_arrow(rp / "global.arrow")

    @app.get("/api/layers")
    async def get_layers() -> list[str]:
        layers_dir = rp / "layers"
        if not await anyio.to_thread.run_sync(layers_dir.exists):
            return []
        files = await anyio.to_thread.run_sync(lambda: sorted(layers_dir.glob("*.arrow")))
        return [_decode_layer_name(f.name) for f in files]

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
        spikes_dir = rp / "spikes"
        if not await anyio.to_thread.run_sync(spikes_dir.exists):
            return []
        files = await anyio.to_thread.run_sync(
            lambda: sorted(spikes_dir.glob("spike_step_*.arrow"))
        )
        result = []
        for f in files:
            match = re.search(r"spike_step_(\d+)\.arrow", f.name)
            if match:
                result.append({"step": int(match.group(1)), "file": f.name})
        return result

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

    if (static_dir / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
    else:

        @app.get("/")
        async def index() -> HTMLResponse:
            return HTMLResponse(content=_fallback_html)

    return app


def start_server(
    run_path: str,
    host: str = "127.0.0.1",
    port: int = 7007,
    log_level: str = "info",
) -> None:
    import uvicorn

    uvicorn.run(
        create_app(run_path),
        host=host,
        port=port,
        log_level=log_level.lower(),
    )
