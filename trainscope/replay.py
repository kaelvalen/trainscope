from __future__ import annotations

import pickle
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch


class SkippingDataLoader:
    """Wraps any iterable DataLoader, yielding batches whose index is not in skip_batches.

    The batch counter resets each time ``__iter__`` is called, so the loader
    behaves consistently across epochs.  Negative skip indices are rejected at
    construction time.

    Usage::

        from trainscope.replay import SkippingDataLoader
        import json

        with open("replay_config.json") as f:
            cfg = json.load(f)

        loader = SkippingDataLoader(original_loader, skip_batches=cfg["skip_batches"])
        for batch in loader:
            loss = model(batch)
            ...
    """

    def __init__(self, loader: Iterable[Any], skip_batches: list[int]):
        if any(b < 0 for b in skip_batches):
            raise ValueError("skip_batches must be non-negative")
        self._loader = loader
        self._skip = set(skip_batches)

    def __iter__(self) -> Iterator[Any]:
        for i, batch in enumerate(self._loader):
            if i not in self._skip:
                yield batch

    def __len__(self) -> int:
        try:
            base = len(self._loader)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError("underlying loader does not support len()") from exc
        skipped = sum(1 for i in self._skip if i < base)
        return max(0, base - skipped)


def load_rng_state(path: str | Path) -> None:
    """Restore PyTorch, NumPy and (optionally) CUDA RNG states from a pickle file.

    The file is expected to contain a dictionary with at least a
    ``torch_rng`` key.  Optional ``numpy_rng`` and ``cuda_rng`` keys are also
    honoured when present.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"RNG state file not found: {path}")
    with open(path, "rb") as f:
        state = pickle.load(f)
    if not isinstance(state, dict):
        raise ValueError(f"RNG state file must contain a dict: {path}")
    if "torch_rng" not in state:
        raise ValueError(f"RNG state missing 'torch_rng' key: {path}")
    torch.set_rng_state(state["torch_rng"])
    if "numpy_rng" in state:
        np.random.set_state(state["numpy_rng"])
    if torch.cuda.is_available() and "cuda_rng" in state:
        torch.cuda.set_rng_state(state["cuda_rng"])


def _move_to_device(batch: Any, device: torch.device | str) -> Any:
    """Recursively move tensors in a batch to ``device``."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, tuple):
        return tuple(_move_to_device(item, device) for item in batch)
    if isinstance(batch, list):
        return [_move_to_device(item, device) for item in batch]
    if isinstance(batch, dict):
        return {k: _move_to_device(v, device) for k, v in batch.items()}
    return batch


def replay_step(
    model: torch.nn.Module,
    batch: Any,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.nn.Module, Any], torch.Tensor],
    *,
    rng_state_path: str | Path | None = None,
    device: torch.device | str | None = None,
    zero_grad: bool = True,
    backward: bool = True,
    step: bool = True,
) -> float:
    """Run a single training step using a reproducible RNG state.

    ``loss_fn`` receives the model and the current batch and must return a
    scalar tensor.  Optional toggles control whether gradients are zeroed,
    ``loss.backward()`` is called and ``optimizer.step()`` is applied.
    """
    if rng_state_path is not None:
        load_rng_state(rng_state_path)
    if device is not None:
        batch = _move_to_device(batch, device)
    if zero_grad:
        optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model, batch)
    if backward and isinstance(loss, torch.Tensor):
        loss.backward()
    if step:
        optimizer.step()
    return float(loss.detach()) if isinstance(loss, torch.Tensor) else float(loss)
