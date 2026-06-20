import warnings

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
