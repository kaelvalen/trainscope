import json
import math
import warnings

import pytest
import torch
import torch.nn as nn

from trainscope import TrainScope, TrainScopeConfig
from trainscope.io.writer import read_arrow_rows_sync


def _grad_l2_norm(model: nn.Module) -> float:
    """Total L2 norm of all non-None parameter gradients of ``model``."""
    with torch.no_grad():
        return float(
            torch.sqrt(
                sum(
                    (
                        p.grad.detach().pow(2).sum()
                        for p in model.parameters()
                        if p.grad is not None
                    ),
                    torch.zeros(()),
                )
            ).item()
        )


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
    layer_rows = read_arrow_rows_sync(layer_path)
    steps = [r["step"] for r in layer_rows]
    kurtosis = [r["act_kurtosis"] for r in layer_rows]

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

    rows = read_arrow_rows_sync(tmp_path / "scope_after_clip" / "global.arrow")
    assert rows[0]["grad_norm_after_clip"] == pytest.approx(0.123)
    assert rows[0]["grad_norm_before_clip"] != pytest.approx(0.123)


def test_step_records_realistic_external_clip_flow(tmp_path):
    """End-to-end external-clip flow: the caller clips with
    clip_grad_norm_, then passes the real post-clip norm. Since scope reads
    the model's (already clipped) gradients as its 'before' reading, both
    fields must agree with the actual post-clip norm."""
    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    cfg = TrainScopeConfig(run_dir=str(tmp_path), run_name="scope_real_clip")
    scope = TrainScope(model, opt, cfg).attach()

    # Deterministic gradients with norm sqrt(5) ≈ 2.24 > max_norm so the
    # clip always fires and the post-clip norm lands exactly on max_norm.
    model.weight.grad = torch.ones_like(model.weight)
    model.bias.grad = torch.ones_like(model.bias)
    assert _grad_l2_norm(model) > 0.1

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
    post_clip_norm = _grad_l2_norm(model)

    scope.step(1.0, grad_norm_after_clip=post_clip_norm, batch_index=0)
    scope.detach()

    rows = read_arrow_rows_sync(tmp_path / "scope_real_clip" / "global.arrow")
    assert rows[0]["grad_norm_after_clip"] == pytest.approx(post_clip_norm)
    assert rows[0]["grad_norm_before_clip"] == pytest.approx(post_clip_norm)
    assert post_clip_norm == pytest.approx(0.1, abs=1e-6)


def test_step_defaults_grad_norm_after_clip_to_before_when_omitted(tmp_path):
    """When grad_norm_after_clip is omitted, the recorded value must mirror
    grad_norm_before_clip (and equal the actually measured norm) — guarding
    against regressions where the field becomes None or diverges."""
    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    cfg = TrainScopeConfig(run_dir=str(tmp_path), run_name="scope_default_after_clip")
    scope = TrainScope(model, opt, cfg).attach()

    x = torch.randn(2, 4)
    y = torch.randn(2, 1)
    opt.zero_grad()
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()

    measured_norm = _grad_l2_norm(model)

    scope.step(loss.item(), batch_index=0)
    scope.detach()

    rows = read_arrow_rows_sync(tmp_path / "scope_default_after_clip" / "global.arrow")
    assert rows[0]["grad_norm_after_clip"] == pytest.approx(measured_norm)
    assert rows[0]["grad_norm_before_clip"] == pytest.approx(measured_norm)
    assert rows[0]["grad_norm_after_clip"] == rows[0]["grad_norm_before_clip"]


def test_optimizer_v_norm_detects_adam_subclasses(tmp_path):
    """The v-norm metric must work for Adam-family optimizers regardless of
    their concrete class name (fused/8-bit variants, user subclasses)."""

    class MyAdamW(torch.optim.AdamW):
        pass

    model = nn.Linear(4, 1)
    opt = MyAdamW(model.parameters(), lr=0.01)
    cfg = TrainScopeConfig(run_dir=str(tmp_path), run_name="scope_vnorm_adam_subclass")
    scope = TrainScope(model, opt, cfg).attach()

    x = torch.randn(2, 4)
    y = torch.randn(2, 1)
    opt.zero_grad()
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()
    opt.step()  # Adam-family state (exp_avg_sq) is only populated on step()
    scope.step(loss.item(), batch_index=0)
    scope.detach()

    rows = read_arrow_rows_sync(tmp_path / "scope_vnorm_adam_subclass" / "global.arrow")
    assert rows[0]["optimizer_v_norm"] > 0.0


def test_optimizer_v_norm_zero_for_non_adam(tmp_path):
    """Non-Adam optimizers have no exp_avg_sq state; the metric must be 0.0."""
    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    cfg = TrainScopeConfig(run_dir=str(tmp_path), run_name="scope_vnorm_sgd")
    scope = TrainScope(model, opt, cfg).attach()

    x = torch.randn(2, 4)
    y = torch.randn(2, 1)
    opt.zero_grad()
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()
    scope.step(loss.item(), batch_index=0)
    scope.detach()

    rows = read_arrow_rows_sync(tmp_path / "scope_vnorm_sgd" / "global.arrow")
    assert rows[0]["optimizer_v_norm"] == 0.0
