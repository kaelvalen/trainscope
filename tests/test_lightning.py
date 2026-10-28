import pytest
import torch

L = pytest.importorskip("lightning.pytorch")

from lightning.pytorch import LightningModule  # noqa: E402

from trainscope import TrainScopeConfig  # noqa: E402
from trainscope.integrations.lightning import TrainScopeCallback  # noqa: E402
from trainscope.io.writer import read_arrow_rows_sync  # noqa: E402


class _Regressor(LightningModule):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1)
        )
        self.micro_losses: list[float] = []

    def training_step(self, batch, _idx):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self.net(x).float(), y)
        self.micro_losses.append(loss.item())
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _fit(tmp_path, run_name, **trainer_kwargs):
    torch.manual_seed(0)
    data = torch.utils.data.TensorDataset(torch.randn(32, 4), torch.randn(32, 1))
    module = _Regressor()
    cfg = TrainScopeConfig(run_dir=str(tmp_path), run_name=run_name, track_memory=False)
    trainer = L.Trainer(
        max_epochs=1,
        accumulate_grad_batches=2,
        callbacks=[TrainScopeCallback(cfg)],
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
        **trainer_kwargs,
    )
    trainer.fit(module, torch.utils.data.DataLoader(data, batch_size=4))
    return module, read_arrow_rows_sync(tmp_path / run_name / "global.arrow")


def _expected_step_losses(micro_losses, accum=2):
    return [sum(micro_losses[i : i + accum]) / accum for i in range(0, len(micro_losses), accum)]


def test_records_accumulated_loss_per_optimizer_step(tmp_path):
    module, rows = _fit(tmp_path, "pl_run")
    expected = _expected_step_losses(module.micro_losses)
    assert [r["step"] for r in rows] == list(range(len(expected)))
    assert [r["loss"] for r in rows] == pytest.approx(expected, rel=1e-5)
    assert all(r["grad_norm_before_clip"] > 0 for r in rows)


def test_fp16_loss_and_grads_are_unscaled(tmp_path):
    from lightning.pytorch.plugins import MixedPrecision

    scaler = torch.amp.GradScaler("cpu", init_scale=1024.0)
    plugin = MixedPrecision("16-mixed", "cpu", scaler=scaler)
    module, rows = _fit(tmp_path, "pl_amp", plugins=[plugin])
    expected = _expected_step_losses(module.micro_losses)
    assert [r["loss"] for r in rows] == pytest.approx(expected, rel=1e-3)
    # Unscaled grads of a tiny regressor are O(1), never O(scale).
    assert all(0 < r["grad_norm_before_clip"] < 100 for r in rows)
