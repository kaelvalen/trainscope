"""Tests for trainscope.replay."""

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainscope.replay import (
    SkippingDataLoader,
    load_rng_state,
    replay_step,
)


class _DummyLoader:
    """Minimal loader for SkippingDataLoader tests."""

    def __init__(self, items):
        self._items = list(items)

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)


def test_skipping_loader_yields_expected_batches():
    loader = SkippingDataLoader(_DummyLoader([0, 1, 2, 3, 4]), skip_batches=[1, 3])
    assert list(loader) == [0, 2, 4]


def test_skipping_loader_len():
    loader = SkippingDataLoader(_DummyLoader(list(range(10))), skip_batches=[0, 5, 9])
    assert len(loader) == 7


def test_skipping_loader_len_with_out_of_range_skips():
    loader = SkippingDataLoader(_DummyLoader(list(range(3))), skip_batches=[5, 10])
    assert len(loader) == 3


def test_skipping_loader_empty_skip():
    loader = SkippingDataLoader(_DummyLoader([0, 1, 2]), skip_batches=[])
    assert list(loader) == [0, 1, 2]
    assert len(loader) == 3


def test_skipping_loader_rejects_negative_skip():
    with pytest.raises(ValueError, match="non-negative"):
        SkippingDataLoader(_DummyLoader([]), skip_batches=[-1])


def test_skipping_loader_counter_resets_per_iteration():
    loader = SkippingDataLoader(_DummyLoader([10, 20, 30]), skip_batches=[1])
    assert list(loader) == [10, 30]
    assert list(loader) == [10, 30]


def test_skipping_loader_rejects_unsized_underlying():
    class Unsized:
        def __iter__(self):
            return iter([1, 2, 3])

    loader = SkippingDataLoader(Unsized(), skip_batches=[0])
    with pytest.raises(TypeError, match="len"):
        len(loader)


def test_load_rng_state_restores_torch_and_numpy(tmp_path: Path):
    torch.manual_seed(1234)
    np.random.seed(1234)

    state_path = tmp_path / "rng.pkl"
    with open(state_path, "wb") as f:
        pickle.dump(
            {
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
            },
            f,
        )

    torch.manual_seed(0)
    np.random.seed(0)
    load_rng_state(state_path)

    a_torch = torch.rand(4)
    a_numpy = np.random.rand(4)

    torch.manual_seed(1234)
    np.random.seed(1234)
    expected_torch = torch.rand(4)
    expected_numpy = np.random.rand(4)

    assert torch.allclose(a_torch, expected_torch)
    np.testing.assert_allclose(a_numpy, expected_numpy)


def test_load_rng_state_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_rng_state(tmp_path / "missing.pkl")


def test_load_rng_state_missing_torch_key(tmp_path: Path):
    path = tmp_path / "bad.pkl"
    with open(path, "wb") as f:
        pickle.dump({"numpy_rng": np.random.get_state()}, f)
    with pytest.raises(ValueError, match="torch_rng"):
        load_rng_state(path)


def test_load_rng_state_non_dict_file(tmp_path: Path):
    path = tmp_path / "bad.pkl"
    with open(path, "wb") as f:
        pickle.dump([1, 2, 3], f)
    with pytest.raises(ValueError, match="dict"):
        load_rng_state(path)


def test_replay_step_runs_full_training_step():
    model = nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    batch = torch.randn(8, 4)
    target = torch.randn(8, 1)

    initial_weight = model.weight.detach().clone()

    def loss_fn(model, batch):
        return F.mse_loss(model(batch), target)

    loss = replay_step(model, batch, optimizer, loss_fn)

    assert isinstance(loss, float)
    assert loss >= 0.0
    assert not torch.equal(model.weight, initial_weight)


def test_replay_step_respects_toggles():
    model = nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    batch = torch.randn(8, 4)
    target = torch.randn(8, 1)

    def loss_fn(model, batch):
        return F.mse_loss(model(batch), target)

    initial_weight = model.weight.detach().clone()
    loss = replay_step(
        model,
        batch,
        optimizer,
        loss_fn,
        zero_grad=False,
        backward=False,
        step=False,
    )
    assert isinstance(loss, float)
    assert torch.equal(model.weight, initial_weight)


def test_replay_step_uses_rng_state(tmp_path: Path):
    torch.manual_seed(7)
    state_path = tmp_path / "rng.pkl"
    with open(state_path, "wb") as f:
        pickle.dump(
            {
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
            },
            f,
        )

    model = nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = torch.randn(8, 4)
    target = torch.randn(8, 1)

    # Capture the random noise consumed inside the loss function so we can
    # verify that replay_step restored the RNG state before the forward pass.
    captured_noise = None

    def loss_fn(model, batch):
        nonlocal captured_noise
        captured_noise = torch.rand(batch.shape)
        return F.mse_loss(model(batch), target) + captured_noise.sum()

    replay_step(model, batch, optimizer, loss_fn, rng_state_path=state_path)

    torch.manual_seed(7)
    expected_noise = torch.rand(batch.shape)
    assert captured_noise is not None
    assert torch.allclose(captured_noise, expected_noise)
