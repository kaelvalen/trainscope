import json
import math
import warnings

import pyarrow.ipc as ipc
import pytest
import torch
import torch.nn as nn

from trainscope import TrainScope, TrainScopeConfig


def test_attach_writes_meta(tmp_path):
    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    cfg = TrainScopeConfig(run_dir=str(tmp_path), run_name="scope_meta")
    scope = TrainScope(model, opt, cfg).attach()
    scope.detach()

    meta_file = tmp_path / "scope_meta" / "meta.json"
    assert meta_file.exists()


def test_meta_records_detector_warmup_info(tmp_path):
    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    cfg = TrainScopeConfig(run_dir=str(tmp_path), run_name="scope_detector_meta")
    scope = TrainScope(model, opt, cfg).attach()
    scope.detach()

    with open(tmp_path / "scope_detector_meta" / "meta.json") as f:
        meta = json.load(f)
    assert meta["detector"]["name"] == "changepoint"
    assert meta["detector"]["min_observations"] == 30


def test_step_records_global_and_layers(tmp_path):
    model = nn.Sequential(nn.Linear(4, 2), nn.ReLU(), nn.Linear(2, 1))
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    cfg = TrainScopeConfig(
        run_dir=str(tmp_path),
        run_name="scope_step",
        full_resolution_window=10,
        spike_window_before=5,
        trace_every_n_steps=1,
        activation_metrics_every_n_steps=1,
    )
    scope = TrainScope(model, opt, cfg).attach()

    x = torch.randn(2, 4)
    y = torch.randn(2, 1)
    opt.zero_grad()
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()
    scope.step(loss.item(), batch_index=0)
    scope.detach()

    run_path = tmp_path / "scope_step"
    assert (run_path / "global.arrow").exists()
    assert any((run_path / "layers").glob("*.arrow"))


def test_clip_grad_norm_deprecation(tmp_path):
    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    cfg = TrainScopeConfig(run_dir=str(tmp_path), run_name="scope_clip")
    scope = TrainScope(model, opt, cfg).attach()

    x = torch.randn(2, 4)
    y = torch.randn(2, 1)
    opt.zero_grad()
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        scope.step(loss.item(), clip_grad_norm=1.0)
        assert any(issubclass(x.category, DeprecationWarning) for x in w)

    scope.detach()


def test_inf_loss_does_not_poison_detector_or_report_spike(tmp_path):
    """A non-finite loss (inf or nan) must not be handed to the anomaly
    detector: it would corrupt the running baseline (mean/median/MAD) for
    every subsequent step. math.isnan alone misses +inf/-inf, which are
    just as capable of poisoning the baseline as NaN is."""
    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    cfg = TrainScopeConfig(run_dir=str(tmp_path), run_name="scope_inf")
    scope = TrainScope(model, opt, cfg).attach()

    x = torch.randn(2, 4)
    y = torch.randn(2, 1)

    def do_step(loss_value, step):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        return scope.step(loss_value if loss_value is not None else loss.item(), batch_index=step)

    for step in range(35):
        result = do_step(None, step)
        assert result is None

    inf_result = do_step(math.inf, 35)
    assert inf_result is None

    finite_result = do_step(None, 36)
    assert finite_result is None

    scope.detach()


def test_unmeasured_activation_metrics_are_null_not_zero(tmp_path):
    """Steps between activation_metrics_every_n_steps intervals must persist
    null activation metrics so the UI can tell 'not measured' apart from
    'measured and zero'."""
    model = nn.Sequential(nn.Linear(4, 2), nn.ReLU(), nn.Linear(2, 1))
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    cfg = TrainScopeConfig(
        run_dir=str(tmp_path),
        run_name="scope_act_nulls",
        trace_every_n_steps=1,
        activation_metrics_every_n_steps=5,
        full_resolution_window=20,
        spike_window_before=5,
    )
    scope = TrainScope(model, opt, cfg).attach()

    x = torch.randn(2, 4)
    y = torch.randn(2, 1)
    for step in range(6):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        scope.step(loss.item(), batch_index=step)
    scope.detach()

    layer_path = next((tmp_path / "scope_act_nulls" / "layers").glob("*.arrow"))
    reader = ipc.open_file(str(layer_path))
    table = reader.read_all()
    d = table.to_pydict()
    steps = d["step"]
    kurtosis = d["act_kurtosis"]

    # Steps 0 and 5 are multiples of 5: measured (a real float).
    for row_idx, step in enumerate(steps):
        if step % 5 == 0:
            assert isinstance(kurtosis[row_idx], float)
        else:
            assert kurtosis[row_idx] is None


def test_step_records_custom_grad_norm_after_clip(tmp_path):
    """Callers that clip externally can pass the real post-clip norm via
    grad_norm_after_clip; it must be recorded instead of mirroring the
    pre-clip reading."""
    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    cfg = TrainScopeConfig(run_dir=str(tmp_path), run_name="scope_after_clip")
    scope = TrainScope(model, opt, cfg).attach()

    x = torch.randn(2, 4)
    y = torch.randn(2, 1)
    opt.zero_grad()
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()
    scope.step(loss.item(), grad_norm_after_clip=0.123, batch_index=0)
    scope.detach()

    reader = ipc.open_file(str(tmp_path / "scope_after_clip" / "global.arrow"))
    table = reader.read_all()
    d = table.to_pydict()
    assert d["grad_norm_after_clip"][0] == pytest.approx(0.123)
    assert d["grad_norm_before_clip"][0] != pytest.approx(0.123)
