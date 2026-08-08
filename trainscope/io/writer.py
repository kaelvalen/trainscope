import io
import json
import logging
import os
import pickle
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch

from trainscope.core.config import TrainScopeConfig

logger = logging.getLogger("trainscope")

GLOBAL_SCHEMA = pa.schema(
    [
        pa.field("step", pa.int64()),
        pa.field("loss", pa.float64()),
        pa.field("grad_norm_before_clip", pa.float64()),
        pa.field("grad_norm_after_clip", pa.float64()),
        pa.field("lr", pa.float64()),
        pa.field("optimizer_v_norm", pa.float64()),
        pa.field("step_time_ms", pa.float64()),
        pa.field("batch_index", pa.int64()),
        pa.field("is_spike", pa.bool_()),
        pa.field("cpu_memory_mb", pa.float64()),
        pa.field("cuda_memory_mb", pa.float64()),
        pa.field("spike_score", pa.float64()),
    ]
)

LAYER_SCHEMA = pa.schema(
    [
        pa.field("step", pa.int64()),
        pa.field("grad_l2_norm", pa.float64()),
        pa.field("weight_l2_norm", pa.float64()),
        pa.field("act_mean", pa.float64()),
        pa.field("act_std", pa.float64()),
        pa.field("act_max_abs", pa.float64()),
        pa.field("act_kurtosis", pa.float64()),
        pa.field("grad_nan_inf_ratio", pa.float64()),
        pa.field("hist_counts", pa.list_(pa.float64())),
        pa.field("hist_edges", pa.list_(pa.float64())),
        pa.field("grad_max_abs", pa.float64()),
        pa.field("grad_mean", pa.float64()),
        pa.field("weight_mean", pa.float64()),
        pa.field("weight_std", pa.float64()),
        pa.field("weight_max_abs", pa.float64()),
        pa.field("weight_min", pa.float64()),
        pa.field("act_min", pa.float64()),
        pa.field("act_max", pa.float64()),
        pa.field("act_median", pa.float64()),
    ]
)

PLUGIN_METRICS_SCHEMA = pa.schema(
    [
        pa.field("step", pa.int64()),
        pa.field("plugin", pa.string()),
        pa.field("metric", pa.string()),
        pa.field("value", pa.float64()),
    ]
)

FLUSH_INTERVAL = 5


def _make_global_batch(rows: list[dict]) -> pa.RecordBatch:
    return pa.record_batch(
        {
            "step": pa.array([r.get("step", 0) for r in rows], type=pa.int64()),
            "loss": pa.array([r.get("loss", 0.0) for r in rows], type=pa.float64()),
            "grad_norm_before_clip": pa.array(
                [r.get("grad_norm_before_clip", 0.0) for r in rows], type=pa.float64()
            ),
            "grad_norm_after_clip": pa.array(
                [r.get("grad_norm_after_clip", 0.0) for r in rows], type=pa.float64()
            ),
            "lr": pa.array([r.get("lr", 0.0) for r in rows], type=pa.float64()),
            "optimizer_v_norm": pa.array(
                [r.get("optimizer_v_norm", 0.0) for r in rows], type=pa.float64()
            ),
            "step_time_ms": pa.array([r.get("step_time_ms", 0.0) for r in rows], type=pa.float64()),
            "batch_index": pa.array([r.get("batch_index", -1) for r in rows], type=pa.int64()),
            "is_spike": pa.array([r.get("is_spike", False) for r in rows], type=pa.bool_()),
            "cpu_memory_mb": pa.array(
                [r.get("cpu_memory_mb", 0.0) for r in rows], type=pa.float64()
            ),
            "cuda_memory_mb": pa.array(
                [r.get("cuda_memory_mb", 0.0) for r in rows], type=pa.float64()
            ),
            "spike_score": pa.array([r.get("spike_score", 0.0) for r in rows], type=pa.float64()),
        },
        schema=GLOBAL_SCHEMA,
    )


def _make_layer_batch(rows: list[dict]) -> pa.RecordBatch:
    return pa.record_batch(
        {
            "step": pa.array([r.get("step", 0) for r in rows], type=pa.int64()),
            "grad_l2_norm": pa.array([r.get("grad_l2_norm", 0.0) for r in rows], type=pa.float64()),
            "weight_l2_norm": pa.array(
                [r.get("weight_l2_norm", 0.0) for r in rows], type=pa.float64()
            ),
            "act_mean": pa.array([r.get("act_mean") for r in rows], type=pa.float64()),
            "act_std": pa.array([r.get("act_std") for r in rows], type=pa.float64()),
            "act_max_abs": pa.array([r.get("act_max_abs") for r in rows], type=pa.float64()),
            "act_kurtosis": pa.array([r.get("act_kurtosis") for r in rows], type=pa.float64()),
            "grad_nan_inf_ratio": pa.array(
                [r.get("grad_nan_inf_ratio", 0.0) for r in rows], type=pa.float64()
            ),
            "hist_counts": pa.array(
                [r.get("hist_counts", []) for r in rows], type=pa.list_(pa.float64())
            ),
            "hist_edges": pa.array(
                [r.get("hist_edges", []) for r in rows], type=pa.list_(pa.float64())
            ),
            "grad_max_abs": pa.array([r.get("grad_max_abs", 0.0) for r in rows], type=pa.float64()),
            "grad_mean": pa.array([r.get("grad_mean", 0.0) for r in rows], type=pa.float64()),
            "weight_mean": pa.array([r.get("weight_mean", 0.0) for r in rows], type=pa.float64()),
            "weight_std": pa.array([r.get("weight_std", 0.0) for r in rows], type=pa.float64()),
            "weight_max_abs": pa.array(
                [r.get("weight_max_abs", 0.0) for r in rows], type=pa.float64()
            ),
            "weight_min": pa.array([r.get("weight_min", 0.0) for r in rows], type=pa.float64()),
            "act_min": pa.array([r.get("act_min") for r in rows], type=pa.float64()),
            "act_max": pa.array([r.get("act_max") for r in rows], type=pa.float64()),
            "act_median": pa.array([r.get("act_median") for r in rows], type=pa.float64()),
        },
        schema=LAYER_SCHEMA,
    )


def _make_plugin_metrics_batch(rows: list[dict]) -> pa.RecordBatch:
    return pa.record_batch(
        {
            "step": pa.array([r.get("step", 0) for r in rows], type=pa.int64()),
            "plugin": pa.array([r.get("plugin", "") for r in rows], type=pa.string()),
            "metric": pa.array([r.get("metric", "") for r in rows], type=pa.string()),
            "value": pa.array([r.get("value", 0.0) for r in rows], type=pa.float64()),
        },
        schema=PLUGIN_METRICS_SCHEMA,
    )


def _table_to_rows(table: pa.Table) -> list[dict]:
    pydict = table.to_pydict()
    if not pydict:
        return []
    keys = list(pydict.keys())
    n = len(next(iter(pydict.values())))
    rows = []
    for i in range(n):
        row = {}
        for k in keys:
            val = pydict[k][i]
            if hasattr(val, "as_py"):
                val = val.as_py()
            row[k] = val
        rows.append(row)
    return rows


def _read_ipc_stream_rows_bytes(data: bytes) -> list[dict]:
    """Read rows from an IPC *stream*, tolerating a truncated tail.

    The DiskWriter appends batches to live stream files; a crash can leave the
    file without its end-of-stream marker (still readable) or cut a message in
    half (the incomplete tail must be dropped, not fail the whole read).
    """
    try:
        reader = ipc.open_stream(io.BytesIO(data))
    except Exception:
        return []
    batches = []
    try:
        while True:
            try:
                batch = reader.read_next_batch()
            except Exception:
                # Truncated mid-message: keep everything parsed so far.
                break
            if batch is None:
                break
            batches.append(batch)
    finally:
        close = getattr(reader, "close", None)
        if callable(close):
            close()
    if not batches:
        return []
    return _table_to_rows(pa.Table.from_batches(batches))


def read_arrow_rows_sync(path: Path) -> list[dict]:
    """Read rows from an Arrow file written in either on-disk format.

    Runs written before the append-only writer land as legacy IPC *files*
    (``ipc.open_file``); runs written by the append-only writer use the IPC
    *stream* format. Both are readable; ``path`` may also be a concurrently
    written file whose tail is momentarily incomplete.
    """
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        with open(path, "rb") as f:
            return read_arrow_rows_bytes(f.read())
    except Exception:
        return []


def read_arrow_rows_bytes(data: bytes) -> list[dict]:
    """Read rows from Arrow IPC bytes written in either on-disk format.

    Tolerates a truncated tail (a crash mid-write cuts the last message; the
    incomplete tail is dropped rather than failing the whole read).
    """
    if not data:
        return []
    try:
        reader = ipc.open_file(io.BytesIO(data))
        try:
            table = reader.read_all()
        finally:
            close = getattr(reader, "close", None)
            if callable(close):
                close()
        return _table_to_rows(table)
    except Exception:
        pass
    return _read_ipc_stream_rows_bytes(data)


class DiskWriter:
    """Persists training snapshots to Arrow, JSON, checkpoint, and RNG-state files.

    Global/layer/plugin-metrics streams are written in the Arrow IPC *stream*
    format with true appends: each flush writes only the new rows and flushes
    the sink, so the on-disk file stays readable by the UI server throughout a
    live run (IPC streams need no footer, unlike IPC files). Every
    ``config.compaction_every_n_steps`` steps the file is atomically rewritten
    from the full in-memory row list to keep the layout compact; without this,
    rewriting the whole history on every flush would cost O(n^2) total row
    writes on long runs.

    A crash mid-write can leave a truncated tail; readers (``read_arrow_rows_sync``)
    drop the incomplete tail instead of failing the whole read, and the missing
    end-of-stream marker is tolerated.

    Supports resuming an existing run: when ``config.resume`` is True and Arrow
    files already exist, existing rows (legacy IPC file or stream format) are
    read at initialization and merged with new rows; the next flush compacts
    the file into the stream format.
    """

    def __init__(
        self,
        run_path: Path,
        trainscope_config: TrainScopeConfig,
        model_name: str | None = None,
        model_config: dict | None = None,
    ):
        self._run_path = Path(run_path)
        self._config = trainscope_config

        self._run_path.mkdir(parents=True, exist_ok=True)
        (self._run_path / "layers").mkdir(exist_ok=True)
        (self._run_path / "spikes").mkdir(exist_ok=True)
        (self._run_path / "rng_states").mkdir(exist_ok=True)
        (self._run_path / "checkpoints").mkdir(exist_ok=True)

        # Clean up temp files left behind by a crash mid-compaction.
        for stale in self._run_path.glob("*.arrow.tmp"):
            stale.unlink(missing_ok=True)
        for stale in (self._run_path / "layers").glob("*.arrow.tmp"):
            stale.unlink(missing_ok=True)

        if model_name is not None:
            self.write_meta(model_name, model_config or {})

        self._global_buffer: list[dict] = []
        self._layer_buffers: dict[str, list[dict]] = {}
        self._plugin_metrics_buffer: list[dict] = []

        self._closed = False

        # Full in-memory row lists drive compactions (full rewrite every
        # compaction_every_n_steps rows) and hold rows reloaded on resume.
        self._all_global_rows: list[dict] = []
        self._all_layer_rows: dict[str, list[dict]] = {}
        self._all_plugin_metrics_rows: list[dict] = []
        # Total number of global rows persisted (including rows loaded for resume).
        self._n_global_rows = 0
        # Highest step number seen via append_global (or loaded for resume).
        self._last_step: int | None = None
        # Set of layer names for which a layer Arrow file exists or will exist.
        self._layer_files: set[str] = set()

        # Open IPC stream writers (sink, writer) per on-disk stream. Kept open
        # for the lifetime of the writer; closed on close() and reopened after
        # each compaction.
        self._global_stream: tuple[pa.OSFile, Any] | None = None
        self._layer_streams: dict[str, tuple[pa.OSFile, Any]] = {}
        self._plugin_stream: tuple[pa.OSFile, Any] | None = None
        # Rows appended to the on-disk stream since the last compaction.
        self._global_rows_since_compaction = 0
        self._layer_rows_since_compaction: dict[str, int] = {}
        self._plugin_rows_since_compaction = 0
        # Set when resume loads rows from a pre-existing file: those files are
        # in the legacy IPC file format and cannot be appended to, so the next
        # flush compacts them into the stream format.
        self._needs_compaction: set[str] = set()

        if self._config.resume:
            self._load_resume_state()

    def __enter__(self) -> "DiskWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # --------------------------------------------------------------------- #
    # Filename helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _encode_layer_name(name: str) -> str:
        """Return a filesystem-safe, reversible encoding of a layer name."""
        return urllib.parse.quote(name, safe="")

    @staticmethod
    def _decode_layer_name(filename: str) -> str:
        stem = Path(filename).stem
        return urllib.parse.unquote(stem)

    def _global_path(self) -> Path:
        return self._run_path / "global.arrow"

    def _layer_path(self, layer_name: str) -> Path:
        encoded = self._encode_layer_name(layer_name)
        return self._run_path / "layers" / f"{encoded}.arrow"

    def _plugin_metrics_path(self) -> Path:
        return self._run_path / "plugin_metrics.arrow"

    # --------------------------------------------------------------------- #
    # Resume support
    # --------------------------------------------------------------------- #
    def _load_resume_state(self):
        global_path = self._global_path()
        if global_path.exists():
            try:
                self._all_global_rows = read_arrow_rows_sync(global_path)
                self._n_global_rows = len(self._all_global_rows)
                if self._all_global_rows:
                    self._last_step = self._all_global_rows[-1].get("step")
                # The existing file cannot be appended to (legacy IPC file
                # format, or a stream whose segments can't be merged): the
                # next flush compacts it into a fresh stream file.
                self._needs_compaction.add("global")
            except Exception:
                logger.exception("Failed to resume global.arrow, falling back to overwrite")
                self._all_global_rows = []

        layers_dir = self._run_path / "layers"
        if layers_dir.is_dir():
            for path in layers_dir.glob("*.arrow"):
                layer_name = self._decode_layer_name(path.name)
                try:
                    rows = read_arrow_rows_sync(path)
                    self._all_layer_rows[layer_name] = rows
                    self._layer_files.add(layer_name)
                    self._needs_compaction.add(f"layer:{layer_name}")
                except Exception:
                    logger.exception("Failed to resume layer file %s", path)

        plugin_metrics_path = self._plugin_metrics_path()
        if plugin_metrics_path.exists():
            try:
                self._all_plugin_metrics_rows = read_arrow_rows_sync(plugin_metrics_path)
                self._needs_compaction.add("plugin")
            except Exception:
                logger.exception("Failed to resume plugin_metrics.arrow")

    # --------------------------------------------------------------------- #
    # Metadata
    # --------------------------------------------------------------------- #
    def write_meta(self, model_name: str, model_config: dict, detector_info: dict | None = None):
        meta = {
            "model_name": model_name,
            "model_config": model_config,
            "trainscope_config": self._config.to_dict(),
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if detector_info is not None:
            meta["detector"] = detector_info
        with open(self._run_path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.debug("Wrote meta.json for run %s", self._run_path.name)

    def _write_manifest(self):
        last_step = self._last_step

        layer_files = {}
        for layer_name in self._layer_files:
            layer_files[layer_name] = str(self._layer_path(layer_name).relative_to(self._run_path))

        manifest = {
            "last_step": last_step,
            "n_global_rows": self._n_global_rows + len(self._global_buffer),
            "layer_files": layer_files,
            "n_plugin_metric_rows": len(self._all_plugin_metrics_rows)
            + len(self._plugin_metrics_buffer),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(self._run_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    # --------------------------------------------------------------------- #
    # Public append API
    # --------------------------------------------------------------------- #
    def append_global(self, snap: dict):
        if self._closed:
            raise RuntimeError("DiskWriter is closed")
        self._global_buffer.append(snap)
        step = snap.get("step")
        if step is not None:
            self._last_step = step
        if len(self._global_buffer) >= FLUSH_INTERVAL:
            self._flush_global()

    def append_layer(self, layer_name: str, snap: dict):
        if self._closed:
            raise RuntimeError("DiskWriter is closed")
        if layer_name not in self._layer_buffers:
            self._layer_buffers[layer_name] = []
        self._layer_buffers[layer_name].append(snap)
        if len(self._layer_buffers[layer_name]) >= FLUSH_INTERVAL:
            self._flush_layer(layer_name)

    def append_plugin_metrics(self, step: int, plugin_name: str, metrics: dict[str, float]):
        """Append per-step metrics emitted by a :class:`MetricPlugin`.

        Each metric is stored as a separate row so the schema stays simple and
        new metrics can be added without schema changes.
        """
        if self._closed:
            raise RuntimeError("DiskWriter is closed")
        for metric_name, value in metrics.items():
            self._plugin_metrics_buffer.append(
                {
                    "step": step,
                    "plugin": plugin_name,
                    "metric": metric_name,
                    "value": float(value),
                }
            )
        if len(self._plugin_metrics_buffer) >= FLUSH_INTERVAL:
            self._flush_plugin_metrics()

    # --------------------------------------------------------------------- #
    # Flush logic: append-only, with periodic full compaction
    # --------------------------------------------------------------------- #
    @staticmethod
    def _open_stream(path: Path, schema: pa.Schema) -> tuple[pa.OSFile, Any]:
        sink = pa.OSFile(str(path), "wb")
        writer = ipc.new_stream(sink, schema)
        return sink, writer

    @staticmethod
    def _close_stream(stream: tuple[pa.OSFile, Any] | None) -> None:
        if stream is None:
            return
        sink, writer = stream
        try:
            writer.close()  # writes the end-of-stream marker
        finally:
            sink.close()

    @staticmethod
    def _reopen_compacted_stream(
        path: Path,
        schema: pa.Schema,
        rows: list[dict],
        make_batch: Callable[[list[dict]], pa.RecordBatch],
    ) -> tuple[pa.OSFile, Any]:
        """Atomically replace ``path`` with an IPC stream containing ``rows``.

        The returned open (sink, writer) keeps appending to the newly swapped
        file: it holds the only reference to the temp-file inode, which is
        renamed over ``path`` while still open. A crash before the rename
        leaves the old file untouched.
        """
        tmp_path = path.with_suffix(".arrow.tmp")
        tmp_path.unlink(missing_ok=True)
        sink = pa.OSFile(str(tmp_path), "wb")
        writer = ipc.new_stream(sink, schema)
        writer.write_batch(make_batch(rows))
        sink.flush()
        os.replace(str(tmp_path), str(path))
        return sink, writer

    def _flush_global(self):
        if not self._global_buffer:
            return

        rows = self._global_buffer
        self._global_buffer = []
        self._all_global_rows = self._all_global_rows + rows
        self._n_global_rows = len(self._all_global_rows)

        self._global_rows_since_compaction += len(rows)
        if (
            "global" in self._needs_compaction
            or self._global_rows_since_compaction >= self._config.compaction_every_n_steps
        ):
            self._needs_compaction.discard("global")
            self._close_stream(self._global_stream)
            self._global_stream = self._reopen_compacted_stream(
                self._global_path(), GLOBAL_SCHEMA, self._all_global_rows, _make_global_batch
            )
            self._global_rows_since_compaction = 0
            return

        if self._global_stream is None:
            self._global_stream = self._open_stream(self._global_path(), GLOBAL_SCHEMA)
        sink, writer = self._global_stream
        writer.write_batch(_make_global_batch(rows))
        sink.flush()

    def _flush_layer(self, layer_name: str):
        rows = self._layer_buffers.get(layer_name, [])
        if not rows:
            return
        del self._layer_buffers[layer_name]

        self._all_layer_rows[layer_name] = self._all_layer_rows.get(layer_name, []) + rows
        self._layer_files.add(layer_name)

        key = f"layer:{layer_name}"
        n_since = self._layer_rows_since_compaction.get(layer_name, 0) + len(rows)
        self._layer_rows_since_compaction[layer_name] = n_since
        if key in self._needs_compaction or n_since >= self._config.compaction_every_n_steps:
            self._needs_compaction.discard(key)
            self._close_stream(self._layer_streams.pop(layer_name, None))
            self._layer_streams[layer_name] = self._reopen_compacted_stream(
                self._layer_path(layer_name),
                LAYER_SCHEMA,
                self._all_layer_rows[layer_name],
                _make_layer_batch,
            )
            self._layer_rows_since_compaction[layer_name] = 0
            return

        stream = self._layer_streams.get(layer_name)
        if stream is None:
            stream = self._open_stream(self._layer_path(layer_name), LAYER_SCHEMA)
            self._layer_streams[layer_name] = stream
        sink, writer = stream
        writer.write_batch(_make_layer_batch(rows))
        sink.flush()

    def _flush_plugin_metrics(self):
        if not self._plugin_metrics_buffer:
            return

        rows = self._plugin_metrics_buffer
        self._plugin_metrics_buffer = []
        self._all_plugin_metrics_rows = self._all_plugin_metrics_rows + rows

        self._plugin_rows_since_compaction += len(rows)
        if (
            "plugin" in self._needs_compaction
            or self._plugin_rows_since_compaction >= self._config.compaction_every_n_steps
        ):
            self._needs_compaction.discard("plugin")
            self._close_stream(self._plugin_stream)
            self._plugin_stream = self._reopen_compacted_stream(
                self._plugin_metrics_path(),
                PLUGIN_METRICS_SCHEMA,
                self._all_plugin_metrics_rows,
                _make_plugin_metrics_batch,
            )
            self._plugin_rows_since_compaction = 0
            return

        if self._plugin_stream is None:
            self._plugin_stream = self._open_stream(
                self._plugin_metrics_path(), PLUGIN_METRICS_SCHEMA
            )
        sink, writer = self._plugin_stream
        writer.write_batch(_make_plugin_metrics_batch(rows))
        sink.flush()

    # --------------------------------------------------------------------- #
    # Batch builders
    # --------------------------------------------------------------------- #
    def _make_global_batch(self, rows: list[dict]) -> pa.RecordBatch:
        return _make_global_batch(rows)

    def _make_layer_batch(self, rows: list[dict]) -> pa.RecordBatch:
        return _make_layer_batch(rows)

    def _make_plugin_metrics_batch(self, rows: list[dict]) -> pa.RecordBatch:
        return _make_plugin_metrics_batch(rows)

    # --------------------------------------------------------------------- #
    # Spike windows
    # --------------------------------------------------------------------- #
    def write_spike_window(
        self,
        spike_step: int,
        window: list[dict],
        layer_windows: dict[str, list[dict]],
    ):
        spike_dir = self._run_path / "spikes"
        spike_dir.mkdir(exist_ok=True)

        global_rows = [entry["global"] for entry in window if "global" in entry]
        if global_rows:
            path = spike_dir / f"spike_step_{spike_step}.arrow"
            with pa.OSFile(str(path), "wb") as sink:
                w = ipc.new_file(sink, GLOBAL_SCHEMA)
                w.write_batch(_make_global_batch(global_rows))
                w.close()

        if layer_windows:
            layers_dir = spike_dir / f"spike_step_{spike_step}_layers"
            layers_dir.mkdir(exist_ok=True)
            for layer_name, rows in layer_windows.items():
                if not rows:
                    continue
                encoded = self._encode_layer_name(layer_name)
                path = layers_dir / f"{encoded}.arrow"
                with pa.OSFile(str(path), "wb") as sink:
                    w = ipc.new_file(sink, LAYER_SCHEMA)
                    w.write_batch(_make_layer_batch(rows))
                    w.close()

        logger.debug("Wrote spike window for step %d", spike_step)

    # --------------------------------------------------------------------- #
    # Checkpoints / RNG state
    # --------------------------------------------------------------------- #
    def save_checkpoint(
        self,
        step: int,
        state_dict: dict[str, Any],
        optimizer_state: dict[str, Any] | None = None,
    ):
        ckpt_dir = self._run_path / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        payload = {"step": step, "model_state_dict": state_dict}
        if optimizer_state is not None:
            payload["optimizer_state_dict"] = optimizer_state
        path = ckpt_dir / f"{step}.pt"
        torch.save(payload, path)
        logger.debug("Saved checkpoint for step %d", step)

    def save_rng_state(self, step: int):
        state = {
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
        }
        if torch.cuda.is_available():
            state["cuda_rng"] = torch.cuda.get_rng_state()
        path = self._run_path / "rng_states" / f"step_{step}.pkl"
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.debug("Saved RNG state for step %d", step)

    # --------------------------------------------------------------------- #
    # Lifecycle
    # --------------------------------------------------------------------- #
    def flush(self):
        if self._closed:
            return

        self._flush_global()
        for layer_name in list(self._layer_buffers.keys()):
            self._flush_layer(layer_name)
        self._flush_plugin_metrics()

        self._write_manifest()
        logger.debug("Flushed writer for run %s", self._run_path.name)

    def close(self):
        if self._closed:
            return
        try:
            self.flush()
        finally:
            # Closing each stream writer writes the end-of-stream marker so
            # the final files are fully terminated.
            self._close_stream(self._global_stream)
            self._global_stream = None
            for stream in self._layer_streams.values():
                self._close_stream(stream)
            self._layer_streams = {}
            self._close_stream(self._plugin_stream)
            self._plugin_stream = None
            self._closed = True
