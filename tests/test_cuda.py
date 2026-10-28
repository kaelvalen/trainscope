"""Mixed-precision checks on a real GPU (skipped without CUDA)."""

import math

import pytest
import torch
import torch.nn as nn

from trainscope import TrainScope, TrainScopeConfig
from trainscope.io.writer import read_arrow_rows_sync

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _reference_grad_norm(model: nn.Module) -> float:
    return math.sqrt(
        sum(float(p.grad.norm()) ** 2 for p in model.parameters() if p.grad is not None)
    )


def test_fp16_autocast_grad_scaler_records_unscaled_norms(tmp_path):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 1)).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    cfg = TrainScopeConfig(run_dir=str(tmp_path), run_name="cuda_amp", track_memory=True)
    scope = TrainScope(model, opt, cfg, scaler=scaler).attach()

    expected = []
    for _ in range(3):
        x, y = torch.randn(16, 32, device="cuda"), torch.randn(16, 1, device="cuda")
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = nn.functional.mse_loss(model(x).float(), y)
        scaler.scale(loss).backward()
        expected.append(_reference_grad_norm(model) / scaler.get_scale())
        scope.step(loss.item())  # before unscale_: scope must divide by the scale
        scaler.step(opt)
        scaler.update()
    scope.detach()

    rows = read_arrow_rows_sync(tmp_path / "cuda_amp" / "global.arrow")
    assert [r["grad_norm_before_clip"] for r in rows] == pytest.approx(expected, rel=1e-4)
    assert all(r["cuda_memory_mb"] > 0 for r in rows)


def test_hf_trainer_fp16_on_gpu(tmp_path):
    transformers = pytest.importorskip("transformers")
    from trainscope.integrations.huggingface import TrainScopeCallback

    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(n_layer=2, n_embd=64, n_head=2, vocab_size=128, n_positions=32)
    )
    data = [{"input_ids": ids, "labels": ids} for ids in torch.randint(0, 128, (32, 32))]
    args = transformers.TrainingArguments(
        output_dir=str(tmp_path / "hf_out"),
        max_steps=6,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        fp16=True,
        report_to=[],
        save_strategy="no",
        disable_tqdm=True,
    )
    cfg = TrainScopeConfig(run_dir=str(tmp_path), run_name="hf_fp16", track_memory=False)
    trainer = transformers.Trainer(
        model=model, args=args, train_dataset=data, callbacks=[TrainScopeCallback(cfg)]
    )
    trainer.train()

    rows = read_arrow_rows_sync(tmp_path / "hf_fp16" / "global.arrow")
    train_loss = trainer.state.log_history[-1]["train_loss"]
    assert sum(r["loss"] for r in rows) / len(rows) == pytest.approx(train_loss, rel=1e-3)
    # Unscaled grads of a tiny GPT-2 are O(1); scaled ones would be ~65536x.
    # Non-finite norms are fp16 overflow steps the scaler skips (shown as gaps).
    finite = [r["grad_norm_before_clip"] for r in rows if math.isfinite(r["grad_norm_before_clip"])]
    assert finite and all(0 < v < 100 for v in finite)


def test_lightning_16_mixed_on_gpu(tmp_path):
    L = pytest.importorskip("lightning.pytorch")
    from lightning.pytorch import LightningModule

    from trainscope.integrations.lightning import TrainScopeCallback

    class Regressor(LightningModule):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))
            self.micro_losses: list[float] = []

        def training_step(self, batch, _idx):
            x, y = batch
            loss = nn.functional.mse_loss(self.net(x).float(), y)
            self.micro_losses.append(loss.item())
            return loss

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    torch.manual_seed(0)
    data = torch.utils.data.TensorDataset(torch.randn(32, 16), torch.randn(32, 1))
    module = Regressor()
    cfg = TrainScopeConfig(run_dir=str(tmp_path), run_name="pl_gpu", track_memory=False)
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        accumulate_grad_batches=2,
        callbacks=[TrainScopeCallback(cfg)],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
    )
    trainer.fit(module, torch.utils.data.DataLoader(data, batch_size=4))

    rows = read_arrow_rows_sync(tmp_path / "pl_gpu" / "global.arrow")
    m = module.micro_losses
    expected = [(m[i] + m[i + 1]) / 2 for i in range(0, len(m), 2)]
    assert [r["loss"] for r in rows] == pytest.approx(expected, rel=1e-3)
    finite = [r["grad_norm_before_clip"] for r in rows if math.isfinite(r["grad_norm_before_clip"])]
    assert finite and all(0 < v < 100 for v in finite)
