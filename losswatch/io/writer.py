import json
import pickle
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch

from losswatch.core.config import LossWatchConfig

GLOBAL_SCHEMA = pa.schema([
    pa.field("step", pa.int64()),
    pa.field("loss", pa.float64()),
    pa.field("grad_norm_before_clip", pa.float64()),
    pa.field("grad_norm_after_clip", pa.float64()),
    pa.field("lr", pa.float64()),
    pa.field("optimizer_v_norm", pa.float64()),
    pa.field("step_time_ms", pa.float64()),
    pa.field("batch_index", pa.int64()),
    pa.field("is_spike", pa.bool_()),
])

LAYER_SCHEMA = pa.schema([
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
])

FLUSH_INTERVAL = 100


class DiskWriter:
    def __init__(self, run_path: Path, losswatch_config: LossWatchConfig):
        self._run_path = run_path
        self._config = losswatch_config

        self._run_path.mkdir(parents=True, exist_ok=True)
        (self._run_path / "layers").mkdir(exist_ok=True)
        (self._run_path / "spikes").mkdir(exist_ok=True)
        (self._run_path / "rng_states").mkdir(exist_ok=True)

        self._global_buffer: list[dict] = []
        self._layer_buffers: dict[str, list[dict]] = {}
        self._global_writer: ipc.RecordBatchFileWriter | None = None
        self._layer_writers: dict[str, ipc.RecordBatchFileWriter] = {}

    def _get_global_writer(self) -> ipc.RecordBatchFileWriter:
        if self._global_writer is None:
            path = self._run_path / "global.arrow"
            sink = pa.OSFile(str(path), "wb")
            self._global_writer = ipc.new_file(sink, GLOBAL_SCHEMA)
        return self._global_writer

    def _get_layer_writer(self, layer_name: str) -> ipc.RecordBatchFileWriter:
        if layer_name not in self._layer_writers:
            safe_name = layer_name.replace("/", "__")
            path = self._run_path / "layers" / f"{safe_name}.arrow"
            sink = pa.OSFile(str(path), "wb")
            self._layer_writers[layer_name] = ipc.new_file(sink, LAYER_SCHEMA)
        return self._layer_writers[layer_name]

    def write_meta(self, model_name: str, model_config: dict):
        meta = {
            "model_name": model_name,
            "model_config": model_config,
            "losswatch_config": {
                "run_dir": self._config.run_dir,
                "run_name": self._config.run_name,
                "full_resolution_window": self._config.full_resolution_window,
                "decimation_factor": self._config.decimation_factor,
                "spike_threshold": self._config.spike_threshold,
                "spike_window_before": self._config.spike_window_before,
                "spike_window_after": self._config.spike_window_after,
                "n_histogram_bins": self._config.n_histogram_bins,
                "histogram_every_n_steps": self._config.histogram_every_n_steps,
                "activation_metrics_every_n_steps": self._config.activation_metrics_every_n_steps,
                "activation_layer_filter": self._config.activation_layer_filter,
                "stop_on_spike": self._config.stop_on_spike,
                "trace_every_n_steps": self._config.trace_every_n_steps,
                "rank": self._config.rank,
            },
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(self._run_path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    def append_global(self, snap: dict):
        self._global_buffer.append(snap)
        if len(self._global_buffer) >= FLUSH_INTERVAL:
            self._flush_global()

    def append_layer(self, layer_name: str, snap: dict):
        if layer_name not in self._layer_buffers:
            self._layer_buffers[layer_name] = []
        self._layer_buffers[layer_name].append(snap)
        if len(self._layer_buffers[layer_name]) >= FLUSH_INTERVAL:
            self._flush_layer(layer_name)

    def _flush_global(self):
        if not self._global_buffer:
            return
        self._get_global_writer().write_batch(self._make_global_batch(self._global_buffer))
        self._global_buffer = []

    def _flush_layer(self, layer_name: str):
        rows = self._layer_buffers.get(layer_name, [])
        if not rows:
            return
        self._get_layer_writer(layer_name).write_batch(self._make_layer_batch(rows))
        self._layer_buffers[layer_name] = []

    def _make_global_batch(self, rows: list[dict]) -> pa.RecordBatch:
        return pa.record_batch(
            {
                "step": pa.array([r.get("step", 0) for r in rows], type=pa.int64()),
                "loss": pa.array([r.get("loss", 0.0) for r in rows], type=pa.float64()),
                "grad_norm_before_clip": pa.array([r.get("grad_norm_before_clip", 0.0) for r in rows], type=pa.float64()),
                "grad_norm_after_clip": pa.array([r.get("grad_norm_after_clip", 0.0) for r in rows], type=pa.float64()),
                "lr": pa.array([r.get("lr", 0.0) for r in rows], type=pa.float64()),
                "optimizer_v_norm": pa.array([r.get("optimizer_v_norm", 0.0) for r in rows], type=pa.float64()),
                "step_time_ms": pa.array([r.get("step_time_ms", 0.0) for r in rows], type=pa.float64()),
                "batch_index": pa.array([r.get("batch_index", -1) for r in rows], type=pa.int64()),
                "is_spike": pa.array([r.get("is_spike", False) for r in rows], type=pa.bool_()),
            },
            schema=GLOBAL_SCHEMA,
        )

    def _make_layer_batch(self, rows: list[dict]) -> pa.RecordBatch:
        return pa.record_batch(
            {
                "step": pa.array([r.get("step", 0) for r in rows], type=pa.int64()),
                "grad_l2_norm": pa.array([r.get("grad_l2_norm", 0.0) for r in rows], type=pa.float64()),
                "weight_l2_norm": pa.array([r.get("weight_l2_norm", 0.0) for r in rows], type=pa.float64()),
                "act_mean": pa.array([r.get("act_mean", 0.0) for r in rows], type=pa.float64()),
                "act_std": pa.array([r.get("act_std", 0.0) for r in rows], type=pa.float64()),
                "act_max_abs": pa.array([r.get("act_max_abs", 0.0) for r in rows], type=pa.float64()),
                "act_kurtosis": pa.array([r.get("act_kurtosis", 0.0) for r in rows], type=pa.float64()),
                "grad_nan_inf_ratio": pa.array([r.get("grad_nan_inf_ratio", 0.0) for r in rows], type=pa.float64()),
                "hist_counts": pa.array([r.get("hist_counts", []) for r in rows], type=pa.list_(pa.float64())),
                "hist_edges": pa.array([r.get("hist_edges", []) for r in rows], type=pa.list_(pa.float64())),
            },
            schema=LAYER_SCHEMA,
        )

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
                w.write_batch(self._make_global_batch(global_rows))
                w.close()

        if layer_windows:
            layers_dir = spike_dir / f"spike_step_{spike_step}_layers"
            layers_dir.mkdir(exist_ok=True)
            for layer_name, rows in layer_windows.items():
                if not rows:
                    continue
                safe_name = layer_name.replace("/", "__")
                path = layers_dir / f"{safe_name}.arrow"
                with pa.OSFile(str(path), "wb") as sink:
                    w = ipc.new_file(sink, LAYER_SCHEMA)
                    w.write_batch(self._make_layer_batch(rows))
                    w.close()

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

    def flush(self):
        self._flush_global()
        for layer_name in list(self._layer_buffers.keys()):
            self._flush_layer(layer_name)

    def close(self):
        self.flush()
        if self._global_writer is not None:
            self._global_writer.close()
            self._global_writer = None
        for layer_name, writer in self._layer_writers.items():
            writer.close()
        self._layer_writers = {}
