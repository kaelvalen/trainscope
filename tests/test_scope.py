import json
import math
import warnings

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class TestMoEIntegration:
    """Mini MoE: routing shares are persisted and feed the drift detector."""

    def _make_moe_model(self):
        import torch.nn.functional as F

        class Expert(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))

            def forward(self, x):
                return self.net(x)

        class MoEBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.router = nn.Linear(8, 4)
                self.experts = nn.ModuleList([Expert() for _ in range(4)])

            def forward(self, x):
                logits = self.router(x)  # (..., 4)
                probs = F.softmax(logits, dim=-1)
                out = torch.zeros_like(x)
                for i, expert in enumerate(self.experts):
                    out = out + probs[..., i : i + 1] * expert(x)
                return out

        class MoEModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Linear(8, 8)
                self.blocks = nn.ModuleList([MoEBlock(), MoEBlock()])
                self.head = nn.Linear(8, 1)

            def forward(self, x):
                h = self.embed(x)
                for block in self.blocks:
                    h = block(h)
                return self.head(h)

        return MoEModel()

    def test_moe_shares_written_to_arrow(self, tmp_path):

        from trainscope.io.writer import read_arrow_rows_sync

        model = self._make_moe_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        config = TrainScopeConfig(
            run_dir=str(tmp_path),
            run_name="moe_run",
            detector={"name": "expert_utilization_drift", "threshold": 0.85},
            track_memory=False,
        )
        scope = TrainScope(model, optimizer, config).attach()

        x = torch.randn(4, 8)
        y = torch.randn(4, 1)
        for _ in range(5):
            optimizer.zero_grad()
            loss = F.mse_loss(model(x), y)
            loss.backward()
            scope.step(loss.item(), batch_index=0)
            optimizer.step()

        scope.writer.flush()
        scope.detach()

        moe_path = tmp_path / "moe_run" / "moe.arrow"
        assert moe_path.exists()
        rows = read_arrow_rows_sync(moe_path)
        assert len(rows) == 10  # 5 steps x 2 blocks
        assert {r["block"] for r in rows} == {"blocks.0.router", "blocks.1.router"}
        for row in rows:
            assert len(row["shares"]) == 4
            assert abs(sum(row["shares"]) - 1.0) < 1e-6

    def test_detector_receives_max_share_not_loss(self, tmp_path):
        """The expert detector consumes the routing signal, not the loss."""
        from trainscope.core.detectors.expert_utilization import (
            ExpertUtilizationDriftDetector,
        )

        model = self._make_moe_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        config = TrainScopeConfig(
            run_dir=str(tmp_path),
            run_name="moe_detect",
            detector={"name": "expert_utilization_drift", "threshold": 0.85, "min_observations": 3},
            track_memory=False,
        )
        scope = TrainScope(model, optimizer, config).attach()
        assert isinstance(scope._detector, ExpertUtilizationDriftDetector)

        # Force concentration: a near-one-hot router output means max share
        # approaches 1.0. Override the router weights directly.
        for block in model.blocks:
            with torch.no_grad():
                block.router.weight.zero_()
                block.router.weight[0, :] = 10.0
                block.router.bias.zero_()

        x = torch.randn(4, 8)
        y = torch.randn(4, 1)
        spike_steps = []
        for step in range(10):
            optimizer.zero_grad()
            loss = F.mse_loss(model(x), y)
            loss.backward()
            spike = scope.step(loss.item(), batch_index=step)
            optimizer.step()
            if spike is not None:
                spike_steps.append((step, spike["spike_score"]))

        scope.detach()

        # After warmup (3 obs), a consistently concentrated router must
        # trigger the expert detector even though the loss is stable.
        assert spike_steps, "expert detector never fired on concentrated routing"
        _, first_score = spike_steps[0]
        assert first_score >= 0.85


class TestAddressorIntegration:
    """Mini memory-augmented model: slot shares feed the addressor detector."""

    def _make_addressor_model(self):
        class MemoryBank(nn.Module):
            def __init__(self, n_slots):
                super().__init__()
                self.slots = nn.Parameter(torch.randn(n_slots, 8) * 0.1)

            def read(self, weights):
                return torch.einsum("...s,sd->...d", weights, self.slots)

        class AddressorBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))
                self.addressor = nn.Linear(8, 16)  # 16 memory slots
                self.memory = MemoryBank(16)

            def forward(self, x):
                x = x + self.mlp(x)
                weights = F.softmax(self.addressor(x), dim=-1)
                return x + self.memory.read(weights)

        class MemoryModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Linear(8, 8)
                self.blocks = nn.ModuleList([AddressorBlock(), AddressorBlock()])
                self.head = nn.Linear(8, 1)

            def forward(self, x):
                h = self.embed(x)
                for block in self.blocks:
                    h = block(h)
                return self.head(h)

        return MemoryModel()

    def test_addressor_shares_written_to_arrow(self, tmp_path):
        from trainscope.io.writer import read_arrow_rows_sync

        model = self._make_addressor_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        config = TrainScopeConfig(
            run_dir=str(tmp_path),
            run_name="mem_run",
            detector={"name": "addressor_concentration_drift"},
            track_memory=False,
        )
        scope = TrainScope(model, optimizer, config).attach()

        x = torch.randn(4, 8)
        y = torch.randn(4, 1)
        for _ in range(5):
            optimizer.zero_grad()
            loss = F.mse_loss(model(x), y)
            loss.backward()
            scope.step(loss.item(), batch_index=0)
            optimizer.step()

        scope.writer.flush()
        scope.detach()

        moe_path = tmp_path / "mem_run" / "moe.arrow"
        assert moe_path.exists()
        rows = read_arrow_rows_sync(moe_path)
        assert len(rows) == 10  # 5 steps x 2 blocks
        blocks = {r["block"] for r in rows}
        assert blocks == {"blocks.0.addressor", "blocks.1.addressor"}
        for row in rows:
            assert len(row["shares"]) == 16
            assert abs(sum(row["shares"]) - 1.0) < 1e-6

    def test_addressor_detector_fires_on_concentration(self, tmp_path):
        from trainscope.core.detectors.addressor_concentration import (
            AddressorConcentrationDriftDetector,
        )

        model = self._make_addressor_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        config = TrainScopeConfig(
            run_dir=str(tmp_path),
            run_name="mem_detect",
            detector={"name": "addressor_concentration_drift", "min_observations": 3},
            track_memory=False,
        )
        scope = TrainScope(model, optimizer, config).attach()
        assert isinstance(scope._detector, AddressorConcentrationDriftDetector)

        # Force concentration: addressor biases push all tokens to slot 0.
        # A large bias guarantees slot-0 dominance regardless of the sign of
        # the (random) input features — weight-only manipulation leaves ~half
        # of tokens with a negative slot-0 logit.
        for block in model.blocks:
            with torch.no_grad():
                block.addressor.weight.zero_()
                block.addressor.bias.zero_()
                block.addressor.bias[0] = 50.0

        x = torch.randn(4, 8)
        y = torch.randn(4, 1)
        spike_steps = []
        for step in range(10):
            optimizer.zero_grad()
            loss = F.mse_loss(model(x), y)
            loss.backward()
            spike = scope.step(loss.item(), batch_index=step)
            optimizer.step()
            if spike is not None:
                spike_steps.append((step, spike["spike_score"]))

        scope.detach()

        assert spike_steps, "addressor detector never fired on concentrated addressing"
        _, first_score = spike_steps[0]
        assert first_score >= 0.6
