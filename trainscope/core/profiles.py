"""Preset configuration profiles for common TrainScope use cases."""

from typing import Callable


def minimal() -> dict:
    """Minimal overhead profile suitable for very large models.

    Reduces histogram and activation computation frequency, filters to
    attention and MLP layers only, and disables memory tracking.
    """
    return {
        "full_resolution_window": 250,
        "decimation_factor": 20,
        "histogram_every_n_steps": 100,
        "activation_metrics_every_n_steps": 50,
        "activation_layer_filter": ["attn", "mlp"],
        "track_memory": False,
    }


def default() -> dict:
    """Default profile with no overrides."""
    return {}


def debug() -> dict:
    """Verbose debugging profile that records everything every step."""
    return {
        "full_resolution_window": 2000,
        "histogram_every_n_steps": 1,
        "activation_metrics_every_n_steps": 1,
        "track_memory": True,
    }


def production() -> dict:
    """Production profile with longer retention and safety features enabled."""
    return {
        "full_resolution_window": 1000,
        "decimation_factor": 10,
        "track_memory": True,
        "checkpoint_on_spike": True,
        "rng_every_n_steps": 100,
    }


PRESETS: dict[str, Callable[[], dict]] = {
    "minimal": minimal,
    "default": default,
    "debug": debug,
    "production": production,
}


def get_profile(name: str) -> dict:
    """Return the overrides for a named profile, or an empty dict if unknown."""
    return PRESETS.get(name, default)()
