import json
import logging
import os
import pickle
import time
import urllib.parse
from pathlib import Path
from typing import Any

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
            "act_mean": pa.array([r.get("act_mean", 0.0) for r in rows], type=pa.float64()),
            "act_std": pa.array([r.get("act_std", 0.0) for r in rows], type=pa.float64()),
            "act_max_abs": pa.array([r.get("act_max_abs", 0.0) for r in rows], type=pa.float64()),
            "act_kurtosis": pa.array([r.get("act_kurtosis", 0.0) for r in rows], type=pa.float64()),
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
            "act_min": pa.array([r.get("act_min", 0.0) for r in rows], type=pa.float64()),
            "act_max": pa.array([r.get("act_max", 0.0) for r in rows], type=pa.float64()),
            "act_median": pa.array([r.get("act_median", 0.0) for r in rows], type=pa.float64()),
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


class DiskWriter:
    """Persists training snapshots to Arrow, JSON, checkpoint, and RNG-state files.

    Because PyArrow IPC files do not support true append and only become
    readable once a footer is written, each flush atomically rewrites the
    global/layer/plugin-metrics Arrow files from their full in-memory row
    lists. This keeps files valid and readable by the UI server throughout a
    live run, not just after close().

    Supports resuming an existing run: when ``config.resume`` is True and Arrow
    files already exist, existing rows are read at initialization and merged
    with new rows on the first flush.
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

        if model_name is not None:
            self.write_meta(model_name, model_config or {})

        self._global_buffer: list[dict] = []
        self._layer_buffers: dict[str, list[dict]] = {}
        self._plugin_metrics_buffer: list[dict] = []

        self._closed = False

        # Every flush rewrites each Arrow file from this in-memory row list (see
        # _flush_global/_flush_layer/_flush_plugin_metrics). PyArrow's IPC file
        # format only becomes readable once a footer is written at close(), so a
        # long-lived open writer would leave the file unreadable to the UI server
        # for the entire duration of a live run. Rewriting atomically on every
        # flush keeps the on-disk file always valid, at the cost of O(rows) work
        # per flush. Also holds rows reloaded when resuming an existing run.
        self._resume = False
        self._all_global_rows: list[dict] = []
        self._all_layer_rows: dict[str, list[dict]] = {}
        self._all_plugin_metrics_rows: list[dict] = []
        # Total number of global rows persisted (including rows loaded for resume).
        self._n_global_rows = 0
        # Highest step number seen via append_global (or loaded for resume).
        self._last_step: int | None = None
        # Set of layer names for which a layer Arrow file exists or will exist.
        self._layer_files: set[str] = set()
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
                with ipc.open_file(str(global_path)) as reader:
                    table = reader.read_all()
                pydict = table.to_pydict()
                # to_pydict returns columns; convert to row dicts.
                self._all_global_rows = [
                    dict(zip(pydict.keys(), values)) for values in zip(*pydict.values())
                ]
                self._n_global_rows = len(self._all_global_rows)
                if self._all_global_rows:
                    self._last_step = self._all_global_rows[-1].get("step")
                self._resume = True
            except Exception:
                logger.exception("Failed to resume global.arrow, falling back to overwrite")
                self._all_global_rows = []

        layers_dir = self._run_path / "layers"
        if layers_dir.is_dir():
            for path in layers_dir.glob("*.arrow"):
                layer_name = self._decode_layer_name(path.name)
                try:
                    reader = ipc.open_file(str(path))
                    table = reader.read_all()
                    pydict = table.to_pydict()
                    rows = [dict(zip(pydict.keys(), values)) for values in zip(*pydict.values())]
                    self._all_layer_rows[layer_name] = rows
                    self._layer_files.add(layer_name)
                    self._resume = True
                except Exception:
                    logger.exception("Failed to resume layer file %s", path)

        plugin_metrics_path = self._plugin_metrics_path()
        if plugin_metrics_path.exists():
            try:
                with ipc.open_file(str(plugin_metrics_path)) as reader:
                    table = reader.read_all()
                pydict = table.to_pydict()
                self._all_plugin_metrics_rows = [
                    dict(zip(pydict.keys(), values)) for values in zip(*pydict.values())
                ]
                self._resume = True
            except Exception:
                logger.exception("Failed to resume plugin_metrics.arrow")

    # --------------------------------------------------------------------- #
    # Metadata
    # --------------------------------------------------------------------- #
    def write_meta(self, model_name: str, model_config: dict):
        meta = {
            "model_name": model_name,
            "model_config": model_config,
            "trainscope_config": self._config.to_dict(),
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
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
    # Flush logic
    # --------------------------------------------------------------------- #
    def _flush_global(self):
        if not self._global_buffer:
            return

        rows = self._all_global_rows + self._global_buffer
        self._atomic_rewrite_global(rows)
        self._all_global_rows = rows
        self._global_buffer = []
        self._n_global_rows = len(rows)

    def _flush_layer(self, layer_name: str):
        rows = self._layer_buffers.get(layer_name, [])
        if not rows:
            return

        combined = self._all_layer_rows.get(layer_name, []) + rows
        self._atomic_rewrite_layer(layer_name, combined)
        self._all_layer_rows[layer_name] = combined
        del self._layer_buffers[layer_name]

    def _flush_plugin_metrics(self):
        if not self._plugin_metrics_buffer:
            return

        rows = self._all_plugin_metrics_rows + self._plugin_metrics_buffer
        self._atomic_rewrite_plugin_metrics(rows)
        self._all_plugin_metrics_rows = rows
        self._plugin_metrics_buffer = []

    def _atomic_rewrite_global(self, rows: list[dict]):
        path = self._global_path()
        tmp_path = path.with_suffix(".arrow.tmp")
        with pa.OSFile(str(tmp_path), "wb") as sink:
            writer = ipc.new_file(sink, GLOBAL_SCHEMA)
            writer.write_batch(_make_global_batch(rows))
            writer.close()
        os.replace(str(tmp_path), str(path))

    def _atomic_rewrite_layer(self, layer_name: str, rows: list[dict]):
        path = self._layer_path(layer_name)
        tmp_path = path.with_suffix(".arrow.tmp")
        with pa.OSFile(str(tmp_path), "wb") as sink:
            writer = ipc.new_file(sink, LAYER_SCHEMA)
            writer.write_batch(_make_layer_batch(rows))
            writer.close()
        os.replace(str(tmp_path), str(path))
        self._layer_files.add(layer_name)

    def _atomic_rewrite_plugin_metrics(self, rows: list[dict]):
        path = self._plugin_metrics_path()
        tmp_path = path.with_suffix(".arrow.tmp")
        with pa.OSFile(str(tmp_path), "wb") as sink:
            writer = ipc.new_file(sink, PLUGIN_METRICS_SCHEMA)
            writer.write_batch(_make_plugin_metrics_batch(rows))
            writer.close()
        os.replace(str(tmp_path), str(path))

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
            self._closed = True
