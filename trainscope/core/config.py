import warnings
from dataclasses import dataclass
from datetime import datetime

import torch


@dataclass
class TrainScopeConfig:
    run_dir: str = "./trainscope_runs"
    run_name: str | None = None
    # Full-resolution data is kept for the last `full_resolution_window` steps.
    # Older steps are decimated (kept every `decimation_factor`-th step).
    full_resolution_window: int = 500
    decimation_factor: int = 10
    spike_threshold: float = 3.5
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

    def to_dict(self) -> dict:
        """Return a JSON-serializable snapshot of the config."""
        return {
            "run_dir": self.run_dir,
            "run_name": self.run_name,
            "full_resolution_window": self.full_resolution_window,
            "decimation_factor": self.decimation_factor,
            "spike_threshold": self.spike_threshold,
            "spike_window_before": self.spike_window_before,
            "spike_window_after": self.spike_window_after,
            "n_histogram_bins": self.n_histogram_bins,
            "histogram_every_n_steps": self.histogram_every_n_steps,
            "activation_metrics_every_n_steps": self.activation_metrics_every_n_steps,
            "activation_layer_filter": self.activation_layer_filter,
            "stop_on_spike": self.stop_on_spike,
            "trace_every_n_steps": self.trace_every_n_steps,
            "rank": self.rank,
            "device": str(self.device) if self.device is not None else None,
            "track_memory": self.track_memory,
            "checkpoint_on_spike": self.checkpoint_on_spike,
            "rng_every_n_steps": self.rng_every_n_steps,
            "resume": self.resume,
        }
