"""Remote storage writer backed by ``fsspec``.

Mirrors the :class:`trainscope.io.writer.DiskWriter` interface while writing to
URI-backed storage such as ``s3://``, ``gs://``, ``az://`` or ``file://``.
"""

import io
import json
import logging
import pickle
import time
import urllib.parse
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.ipc as ipc
import torch

from trainscope.core.config import TrainScopeConfig
from trainscope.io.writer import (
    GLOBAL_SCHEMA,
    LAYER_SCHEMA,
    PLUGIN_METRICS_SCHEMA,
    _make_global_batch,
    _make_layer_batch,
    _make_plugin_metrics_batch,
    read_arrow_rows_bytes,
)

try:
    import fsspec
    import fsspec.core
except Exception:  # pragma: no cover
    fsspec = None

logger = logging.getLogger("trainscope")


def _encode_layer_name(name: str) -> str:
    return urllib.parse.quote(name, safe="")


def _decode_layer_name(filename: str) -> str:
    return urllib.parse.unquote(Path(filename).stem)


class RemoteWriter:
    """Write trainscope artifacts to a remote or URI-backed filesystem.

    The implementation uses ``fsspec`` so any scheme it supports (including
    ``memory://`` for tests) works without code changes.

    Unlike :class:`DiskWriter`, remote objects (e.g. S3 keys) cannot be
    appended to, so each flush rewrites the full row list as one IPC *stream*
    object. To bound write amplification the rewrite happens only every
    ``config.compaction_every_n_steps`` rows (default 1000) and on
    ``flush()``/``close()``; new rows are buffered in memory in between, so
    remote artifacts lag the training run by up to that many steps.
    """

    def __init__(
        self,
        run_path: str,
        trainscope_config: TrainScopeConfig,
        model_name: str | None = None,
        model_config: dict | None = None,
    ):
        if fsspec is None:
            raise ImportError("RemoteWriter requires 'fsspec'. Install it with: pip install fsspec")

        self._uri = str(run_path)
        self._fs, self._path = fsspec.core.url_to_fs(self._uri)
        self._config = trainscope_config
        self._closed = False

        self._fs.makedirs(self._path, exist_ok=True)
        for subdir in ("layers", "spikes", "rng_states", "checkpoints"):
            self._fs.makedirs(f"{self._path}/{subdir}", exist_ok=True)

        if model_name is not None:
            self.write_meta(model_name, model_config or {})

        self._global_buffer: list[dict] = []
        self._layer_buffers: dict[str, list[dict]] = {}
        self._plugin_metrics_buffer: list[dict] = []

        self._global_rows: list[dict] = []
        self._layer_rows: dict[str, list[dict]] = {}
        self._plugin_metrics_rows: list[dict] = []

        self._n_global_rows = 0
        self._last_step: int | None = None
        self._layer_files: set[str] = set()
        # Rows appended since the last full object rewrite.
        self._global_rows_since_write = 0
        self._layer_rows_since_write: dict[str, int] = {}
        self._plugin_rows_since_write = 0

        if self._config.resume:
            self._load_resume_state()

    def __enter__(self) -> "RemoteWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ------------------------------------------------------------------ #
    # Path helpers
    # ------------------------------------------------------------------ #
    def _global_path(self) -> str:
        return f"{self._path}/global.arrow"

    def _layer_path(self, layer_name: str) -> str:
        encoded = _encode_layer_name(layer_name)
        return f"{self._path}/layers/{encoded}.arrow"

    def _plugin_metrics_path(self) -> str:
        return f"{self._path}/plugin_metrics.arrow"

    def _meta_path(self) -> str:
        return f"{self._path}/meta.json"

    def _manifest_path(self) -> str:
        return f"{self._path}/manifest.json"

    # ------------------------------------------------------------------ #
    # Resume support
    # ------------------------------------------------------------------ #
    def _read_arrow_rows(self, path: str) -> list[dict]:
        if not self._fs.exists(path):
            return []
        with self._fs.open(path, "rb") as f:
            data = f.read()
        return read_arrow_rows_bytes(data)

    def _load_resume_state(self):
        global_rows = self._read_arrow_rows(self._global_path())
        if global_rows:
            self._global_rows = global_rows
            self._n_global_rows = len(global_rows)
            if global_rows:
                self._last_step = global_rows[-1].get("step")

        try:
            layer_files = self._fs.glob(f"{self._path}/layers/*.arrow")
        except Exception:
            layer_files = []
        for file_path in layer_files:
            name = _decode_layer_name(Path(file_path).name)
            try:
                rows = self._read_arrow_rows(file_path)
                self._layer_rows[name] = rows
                self._layer_files.add(name)
            except Exception:
                logger.exception("Failed to resume layer file %s", file_path)

        plugin_rows = self._read_arrow_rows(self._plugin_metrics_path())
        if plugin_rows:
            self._plugin_metrics_rows = plugin_rows

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #
    def write_meta(self, model_name: str, model_config: dict, detector_info: dict | None = None):
        meta = {
            "model_name": model_name,
            "model_config": model_config,
            "trainscope_config": self._config.to_dict(),
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if detector_info is not None:
            meta["detector"] = detector_info
        with self._fs.open(self._meta_path(), "wb") as f:
            f.write(json.dumps(meta, indent=2).encode("utf-8"))
        logger.debug("Wrote meta.json to %s", self._uri)

    def _write_manifest(self):
        layer_files = {}
        for layer_name in self._layer_files:
            layer_files[layer_name] = self._layer_path(layer_name).replace(f"{self._path}/", "")
        manifest = {
            "last_step": self._last_step,
            "n_global_rows": self._n_global_rows + len(self._global_buffer),
            "layer_files": layer_files,
            "n_plugin_metric_rows": len(self._plugin_metrics_rows)
            + len(self._plugin_metrics_buffer),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with self._fs.open(self._manifest_path(), "wb") as f:
            f.write(json.dumps(manifest, indent=2).encode("utf-8"))

    # ------------------------------------------------------------------ #
    # Arrow helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _write_stream_bytes(schema, rows: list[dict]) -> bytes:
        """Serialize ``rows`` as one IPC *stream* (matching DiskWriter's
        on-disk format for main metric streams)."""
        bio = io.BytesIO()
        with ipc.new_stream(bio, schema) as writer:
            if schema == GLOBAL_SCHEMA:
                writer.write_batch(_make_global_batch(rows))
            elif schema == LAYER_SCHEMA:
                writer.write_batch(_make_layer_batch(rows))
            elif schema == PLUGIN_METRICS_SCHEMA:
                writer.write_batch(_make_plugin_metrics_batch(rows))
            else:
                raise ValueError(f"Unsupported schema {schema}")
        return bio.getvalue()

    @staticmethod
    def _write_file_bytes(schema, rows: list[dict]) -> bytes:
        """Serialize ``rows`` as a legacy IPC *file* (spike windows, which are
        written once and never appended to)."""
        bio = io.BytesIO()
        with ipc.new_file(bio, schema) as writer:
            if schema == GLOBAL_SCHEMA:
                writer.write_batch(_make_global_batch(rows))
            elif schema == LAYER_SCHEMA:
                writer.write_batch(_make_layer_batch(rows))
            elif schema == PLUGIN_METRICS_SCHEMA:
                writer.write_batch(_make_plugin_metrics_batch(rows))
            else:
                raise ValueError(f"Unsupported schema {schema}")
        return bio.getvalue()

    def _write_global(self):
        rows = self._global_rows + self._global_buffer
        if not rows:
            return
        data = self._write_stream_bytes(GLOBAL_SCHEMA, rows)
        with self._fs.open(self._global_path(), "wb") as f:
            f.write(data)
        self._global_rows = rows
        self._global_buffer = []
        self._global_rows_since_write = 0
        self._n_global_rows = len(rows)

    def _write_layer(self, layer_name: str):
        rows = self._layer_rows.get(layer_name, []) + self._layer_buffers.get(layer_name, [])
        if not rows:
            return
        data = self._write_stream_bytes(LAYER_SCHEMA, rows)
        with self._fs.open(self._layer_path(layer_name), "wb") as f:
            f.write(data)
        self._layer_rows[layer_name] = rows
        self._layer_files.add(layer_name)
        if layer_name in self._layer_buffers:
            del self._layer_buffers[layer_name]
        self._layer_rows_since_write[layer_name] = 0

    def _write_plugin_metrics(self):
        rows = self._plugin_metrics_rows + self._plugin_metrics_buffer
        if not rows:
            return
        data = self._write_stream_bytes(PLUGIN_METRICS_SCHEMA, rows)
        with self._fs.open(self._plugin_metrics_path(), "wb") as f:
            f.write(data)
        self._plugin_metrics_rows = rows
        self._plugin_metrics_buffer = []
        self._plugin_rows_since_write = 0

    # ------------------------------------------------------------------ #
    # Public append API
    # ------------------------------------------------------------------ #
    def append_global(self, snap: dict):
        if self._closed:
            raise RuntimeError("RemoteWriter is closed")
        self._global_buffer.append(snap)
        step = snap.get("step")
        if step is not None:
            self._last_step = step
        self._global_rows_since_write += 1
        if self._global_rows_since_write >= self._config.compaction_every_n_steps:
            self._write_global()

    def append_layer(self, layer_name: str, snap: dict):
        if self._closed:
            raise RuntimeError("RemoteWriter is closed")
        if layer_name not in self._layer_buffers:
            self._layer_buffers[layer_name] = []
        self._layer_buffers[layer_name].append(snap)
        n_since = self._layer_rows_since_write.get(layer_name, 0) + 1
        self._layer_rows_since_write[layer_name] = n_since
        if n_since >= self._config.compaction_every_n_steps:
            self._write_layer(layer_name)

    def append_plugin_metrics(self, step: int, plugin_name: str, metrics: dict[str, float]):
        if self._closed:
            raise RuntimeError("RemoteWriter is closed")
        for metric_name, value in metrics.items():
            self._plugin_metrics_buffer.append(
                {
                    "step": step,
                    "plugin": plugin_name,
                    "metric": metric_name,
                    "value": float(value),
                }
            )
            self._plugin_rows_since_write += 1
        if self._plugin_rows_since_write >= self._config.compaction_every_n_steps:
            self._write_plugin_metrics()

    # ------------------------------------------------------------------ #
    # Spike windows
    # ------------------------------------------------------------------ #
    def write_spike_window(
        self,
        spike_step: int,
        window: list[dict],
        layer_windows: dict[str, list[dict]],
    ):
        spike_dir = f"{self._path}/spikes"
        self._fs.makedirs(spike_dir, exist_ok=True)

        global_rows = [entry["global"] for entry in window if "global" in entry]
        if global_rows:
            data = self._write_file_bytes(GLOBAL_SCHEMA, global_rows)
            with self._fs.open(f"{spike_dir}/spike_step_{spike_step}.arrow", "wb") as f:
                f.write(data)

        if layer_windows:
            layers_dir = f"{spike_dir}/spike_step_{spike_step}_layers"
            self._fs.makedirs(layers_dir, exist_ok=True)
            for layer_name, rows in layer_windows.items():
                if not rows:
                    continue
                encoded = _encode_layer_name(layer_name)
                data = self._write_file_bytes(LAYER_SCHEMA, rows)
                with self._fs.open(f"{layers_dir}/{encoded}.arrow", "wb") as f:
                    f.write(data)

        logger.debug("Wrote remote spike window for step %d", spike_step)

    # ------------------------------------------------------------------ #
    # Checkpoints / RNG state
    # ------------------------------------------------------------------ #
    def save_checkpoint(
        self,
        step: int,
        state_dict: dict[str, Any],
        optimizer_state: dict[str, Any] | None = None,
    ):
        ckpt_dir = f"{self._path}/checkpoints"
        self._fs.makedirs(ckpt_dir, exist_ok=True)
        payload = {"step": step, "model_state_dict": state_dict}
        if optimizer_state is not None:
            payload["optimizer_state_dict"] = optimizer_state
        bio = io.BytesIO()
        torch.save(payload, bio)
        with self._fs.open(f"{ckpt_dir}/{step}.pt", "wb") as f:
            f.write(bio.getvalue())
        logger.debug("Saved remote checkpoint for step %d", step)

    def save_rng_state(self, step: int):
        state = {
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
        }
        if torch.cuda.is_available():
            state["cuda_rng"] = torch.cuda.get_rng_state()
        bio = io.BytesIO()
        pickle.dump(state, bio)
        with self._fs.open(f"{self._path}/rng_states/step_{step}.pkl", "wb") as f:
            f.write(bio.getvalue())
        logger.debug("Saved remote RNG state for step %d", step)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def flush(self):
        if self._closed:
            return
        self._write_global()
        for layer_name in list(self._layer_buffers.keys()):
            self._write_layer(layer_name)
        self._write_plugin_metrics()
        self._write_manifest()
        logger.debug("Flushed remote writer for %s", self._uri)

    def close(self):
        if self._closed:
            return
        try:
            self.flush()
        finally:
            self._closed = True
