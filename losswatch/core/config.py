import warnings
from dataclasses import dataclass
from datetime import datetime


@dataclass
class LossWatchConfig:
    run_dir: str = "./losswatch_runs"
    run_name: str | None = None
    # Full-resolution data is kept for the last `full_resolution_window` steps.
    # Older steps are decimated (kept every `decimation_factor`-th step).
    full_resolution_window: int = 500
    decimation_factor: int = 10
    spike_threshold: float = 3.5
    # spike_window_before should be <= full_resolution_window.
    # If larger, the extra range is served from decimated history (every
    # decimation_factor-th step), so spike windows will contain far more
    # entries than spike_window_before alone suggests.
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
    # file collisions when each process creates its own LossWatch.
    rank: int | None = None

    def __post_init__(self):
        if self.run_name is None:
            self.run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        if self.activation_metrics_every_n_steps < self.trace_every_n_steps:
            warnings.warn(
                f"activation_metrics_every_n_steps={self.activation_metrics_every_n_steps} "
                f"< trace_every_n_steps={self.trace_every_n_steps}. "
                "Activation hooks will fire more often than steps are recorded; "
                "set activation_metrics_every_n_steps >= trace_every_n_steps to avoid waste.",
                stacklevel=2,
            )
        if self.spike_window_before > self.full_resolution_window:
            warnings.warn(
                f"spike_window_before={self.spike_window_before} exceeds "
                f"full_resolution_window={self.full_resolution_window}. "
                "The extra range will be served from decimated history, "
                "resulting in a much larger spike window than expected.",
                stacklevel=2,
            )
