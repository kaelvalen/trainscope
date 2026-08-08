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
) -> None:
    import uvicorn

    uvicorn.run(
        create_app(run_path),
        host=host,
        port=port,
        log_level=log_level.lower(),
    )
