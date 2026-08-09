import json
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from trainscope.core.profiles import PRESETS


def _coerce_value(field_name: str, raw: str) -> Any:
    """Parse an environment-variable string into the right Python type."""
    raw = raw.strip()
    if raw == "":
        return None

    # Boolean fields.
    bool_fields = {
        "stop_on_spike",
        "track_memory",
        "resume",
    }
    if field_name in bool_fields:
        return raw.lower() in {"1", "true", "yes", "on"}

    # List fields.
    list_fields = {
        "activation_layer_filter",
        "metric_plugins",
        "detector_plugins",
        "alerts",
    }
    if field_name in list_fields:
        if raw.startswith("["):
            parsed = yaml.safe_load(raw)
            return parsed if isinstance(parsed, list) else []
        return [part.strip() for part in raw.split(",") if part.strip()]

    # Dict fields (detector config / integrations).
    if field_name == "detector":
        if raw.startswith(("{", "[")):
            parsed = yaml.safe_load(raw)
            return parsed if isinstance(parsed, dict) else {"name": str(parsed)}
        return {"name": raw}

    if field_name == "integrations":
        if raw.startswith(("{", "[")):
            parsed = yaml.safe_load(raw)
            return parsed if isinstance(parsed, dict) else {}
        return {}

    # Numeric fields.
    int_fields = {
        "full_resolution_window",
        "decimation_factor",
        "spike_window_before",
        "spike_window_after",
        "n_histogram_bins",
        "histogram_every_n_steps",
        "activation_metrics_every_n_steps",
        "trace_every_n_steps",
        "rank",
        "rng_every_n_steps",
        "compaction_every_n_steps",
    }
    if field_name in int_fields:
        return int(raw)

    # Fallback to string.
    return raw


@dataclass
class TrainScopeConfig:
    run_dir: str = "./trainscope_runs"
    run_name: str | None = None
    # Full-resolution data is kept for the last `full_resolution_window` steps.
    # Older steps are decimated (kept every `decimation_factor`-th step).
    full_resolution_window: int = 500
    decimation_factor: int = 10
    # spike_window_before must be <= full_resolution_window. If equal, the full
    # spike window fits into the full-resolution buffer.
    spike_window_before: int = 50
    spike_window_after: int = 10
    n_histogram_bins: int = 16
    # Weight histograms are expensive (torch.histogram per param per step).
    # Compute them every N steps; spike steps always get a histogram regardless.
    histogram_every_n_steps: int = 50
    # Activation kurtosis (pow4 + std per layer) is also non-trivial at scale.
    # Compute every N steps; spike steps always get full activation metrics.
    activation_metrics_every_n_steps: int = 5
    # Only capture activation metrics for layers whose name contains one of
    # these substrings. None means capture all leaf layers.
    activation_layer_filter: list[str] | None = None
    stop_on_spike: bool = False
    trace_every_n_steps: int = 1
    # DDP rank: when set, appends _rank{rank} to the run directory to avoid
    # file collisions when each process creates its own TrainScope.
    rank: int | None = None
    # Arrow files are written append-only (cheap per-flush writes); every
    # `compaction_every_n_steps` steps the whole file is rewritten from
    # memory to keep the file layout compact. Larger values reduce write
    # amplification at the cost of a sparser on-disk layout.
    compaction_every_n_steps: int = 1000

    # Device used for metric computation. None means "same device as the model
    # tensor being inspected"; explicit values force offloading to that device.
    device: str | torch.device | None = None
    # Record CPU/CUDA memory usage in the global snapshot.
    track_memory: bool = True
    # Save model state dict when a spike is detected. None/False disables this;
    # True uses "checkpoints/{step}.pt"; a string is treated as a path template
    # with a single {step} placeholder.
    checkpoint_on_spike: bool | str | None = None
    # Save RNG state every N steps (in addition to on spikes). 0 means only on
    # spikes (legacy behaviour).
    rng_every_n_steps: int = 0
    # If True and a run directory already contains Arrow files, resume by
    # appending new rows. Otherwise existing files are overwritten.
    resume: bool = False
    # Experiment-tracker integrations (wandb, tensorboard, mlflow).
    integrations: dict[str, Any] = field(default_factory=dict)
    # Alert backends (slack, email) triggered on spike detection.
    alerts: list[dict[str, Any]] = field(default_factory=list)

    # Remote storage URI. When unset, the local DiskWriter is used. Supported
    # schemes include ``s3://``, ``gs://``, ``az://`` and ``file://``.
    storage_uri: str | None = None
    # Anomaly detector configuration. A string selects a detector by name; a
    # dict provides ``name`` plus constructor kwargs. Defaults to the CUSUM
    # change-point detector (see ``ChangePointDetector``), which catches
    # subtle, persistent drift that a plain z-score threshold misses.
    detector: str | dict | None = None
    # Explicit plugin class paths (e.g. ``myplugin.MyMetric``) to load in
    # addition to entry-point discovered plugins.
    metric_plugins: list[str] = field(default_factory=list)
    detector_plugins: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.run_name is None:
            self.run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

        if self.full_resolution_window < 1:
            raise ValueError("full_resolution_window must be >= 1")
        if self.decimation_factor < 1:
            raise ValueError("decimation_factor must be >= 1")
        if self.spike_window_before > self.full_resolution_window:
            raise ValueError(
                f"spike_window_before={self.spike_window_before} exceeds "
                f"full_resolution_window={self.full_resolution_window}. "
                "Increase full_resolution_window or decrease spike_window_before."
            )
        if self.spike_window_after < 0:
            raise ValueError("spike_window_after must be >= 0")
        if self.histogram_every_n_steps < 1:
            raise ValueError("histogram_every_n_steps must be >= 1")
        if self.activation_metrics_every_n_steps < 1:
            raise ValueError("activation_metrics_every_n_steps must be >= 1")
        if self.trace_every_n_steps < 1:
            raise ValueError("trace_every_n_steps must be >= 1")
        if self.compaction_every_n_steps < 1:
            raise ValueError("compaction_every_n_steps must be >= 1")
        if self.n_histogram_bins < 2:
            raise ValueError("n_histogram_bins must be >= 2")
        if self.rng_every_n_steps < 0:
            raise ValueError("rng_every_n_steps must be >= 0")

        if self.activation_metrics_every_n_steps < self.trace_every_n_steps:
            warnings.warn(
                f"activation_metrics_every_n_steps={self.activation_metrics_every_n_steps} "
                f"< trace_every_n_steps={self.trace_every_n_steps}. "
                "Activation hooks will fire more often than steps are recorded; "
                "set activation_metrics_every_n_steps >= trace_every_n_steps to avoid waste.",
                stacklevel=2,
            )

        if isinstance(self.detector, str):
            self.detector = {"name": self.detector}
        if self.detector is None:
            # Each detector scales its own decision threshold; CUSUM's "h"
            # (cumulative-sum decision threshold) lives on a different scale
            # than the z_score detector's raw z-score cutoff, so detectors get
            # their own validated defaults instead of a shared threshold.
            self.detector = {"name": "changepoint"}
        if not isinstance(self.detector, dict) or "name" not in self.detector:
            raise ValueError("detector must be a detector name or a dict with a 'name' key")

        if self.storage_uri is not None and not isinstance(self.storage_uri, str):
            raise ValueError("storage_uri must be a string or None")

        if not isinstance(self.metric_plugins, list) or not all(
            isinstance(x, str) for x in self.metric_plugins
        ):
            raise ValueError("metric_plugins must be a list of strings")
        if not isinstance(self.detector_plugins, list) or not all(
            isinstance(x, str) for x in self.detector_plugins
        ):
            raise ValueError("detector_plugins must be a list of strings")

    def to_dict(self) -> dict:
        """Return a JSON-serializable snapshot of the config."""
        return {
            "run_dir": self.run_dir,
            "run_name": self.run_name,
            "full_resolution_window": self.full_resolution_window,
            "decimation_factor": self.decimation_factor,
            "spike_window_before": self.spike_window_before,
            "spike_window_after": self.spike_window_after,
            "n_histogram_bins": self.n_histogram_bins,
            "histogram_every_n_steps": self.histogram_every_n_steps,
            "activation_metrics_every_n_steps": self.activation_metrics_every_n_steps,
            "activation_layer_filter": self.activation_layer_filter,
            "stop_on_spike": self.stop_on_spike,
            "trace_every_n_steps": self.trace_every_n_steps,
            "rank": self.rank,
            "compaction_every_n_steps": self.compaction_every_n_steps,
            "device": str(self.device) if self.device is not None else None,
            "track_memory": self.track_memory,
            "checkpoint_on_spike": self.checkpoint_on_spike,
            "rng_every_n_steps": self.rng_every_n_steps,
            "resume": self.resume,
            "storage_uri": self.storage_uri,
            "detector": self.detector,
            "metric_plugins": self.metric_plugins,
            "detector_plugins": self.detector_plugins,
            "integrations": self.integrations,
            "alerts": self.alerts,
        }

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainScopeConfig":
        """Load a config from a YAML file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return load_config(data)

    @classmethod
    def from_env(cls, prefix: str = "TRAINSCOPE_") -> "TrainScopeConfig":
        """Build a config from ``TRAINSCOPE_*`` environment variables."""
        profile_name = os.environ.get(f"{prefix}PROFILE")
        base: dict[str, Any] = {}
        if profile_name:
            base = PRESETS.get(profile_name, PRESETS["default"])()

        overrides: dict[str, Any] = {}
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        for key, value in os.environ.items():
            if not key.startswith(prefix) or key == f"{prefix}PROFILE":
                continue
            field_name = key[len(prefix) :].lower()
            if field_name == "spike_threshold":
                raise ValueError(
                    f"{prefix}SPIKE_THRESHOLD was removed in 1.0. Configure the detector "
                    "threshold instead, e.g. "
                    f'{prefix}DETECTOR=\'{{"name": "z_score", "threshold": 3.5}}\''
                )
            if field_name not in field_names:
                continue
            overrides[field_name] = _coerce_value(field_name, value)

        merged = {**base, **overrides}
        return cls(**merged)


def load_config(path_or_dict: str | Path | dict[str, Any] | TrainScopeConfig) -> TrainScopeConfig:
    """Load a :class:`TrainScopeConfig` from a path, dict, or pass one through.

    Supports YAML files (``.yaml`` / ``.yml``) and JSON files. Dicts may
    contain a ``profile`` key that selects preset overrides before applying the
    rest of the configuration.
    """
    if isinstance(path_or_dict, TrainScopeConfig):
        return path_or_dict

    if isinstance(path_or_dict, (str, Path)):
        path = Path(path_or_dict)
        with open(path, "r", encoding="utf-8") as f:
            if path.suffix.lower() in {".yaml", ".yml"}:
                data = yaml.safe_load(f) or {}
            else:
                data = json.load(f)
    elif isinstance(path_or_dict, dict):
        data = dict(path_or_dict)
    else:
        raise TypeError(f"Expected Path, str, dict or TrainScopeConfig, got {type(path_or_dict)}")

    profile_name = data.pop("profile", None)
    base = PRESETS.get(profile_name, PRESETS["default"])() if profile_name else {}
    merged = {**base, **data}

    if "spike_threshold" in merged:
        raise ValueError(
            "spike_threshold was removed in 1.0. Configure the detector threshold "
            "instead, e.g. detector={'name': 'z_score', 'threshold': 3.5}."
        )

    return TrainScopeConfig(**merged)
