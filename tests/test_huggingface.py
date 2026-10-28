import math

import pytest
import torch

transformers = pytest.importorskip("transformers")

from trainscope import TrainScopeConfig  # noqa: E402
from trainscope.integrations.huggingface import TrainScopeCallback  # noqa: E402
from trainscope.io.writer import read_arrow_rows_sync  # noqa: E402


class _PlainLM(torch.nn.Module):
    """A model without **kwargs: the Trainer passes no num_items_in_batch, so
    each micro-batch loss is a per-micro-batch mean."""

    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(64, 32)
        self.head = torch.nn.Linear(32, 64)

    def forward(self, input_ids, labels):
        logits = self.head(self.emb(input_ids))
        return {"loss": torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten())}


def _tiny_trainer(tmp_path, callback, max_steps=6, grad_accum=2, plain=False):
    torch.manual_seed(0)
    model = (
        _PlainLM()
        if plain
        else transformers.GPT2LMHeadModel(
            transformers.GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=64, n_positions=16)
        )
    )
    data = [{"input_ids": ids, "labels": ids} for ids in torch.randint(0, 64, (32, 16))]
    args = transformers.TrainingArguments(
        output_dir=str(tmp_path / "hf_out"),
        max_steps=max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=grad_accum,
        report_to=[],
        save_strategy="no",
        use_cpu=True,
        disable_tqdm=True,
    )
    return transformers.Trainer(model=model, args=args, train_dataset=data, callbacks=[callback])


@pytest.mark.parametrize("plain", [False, True], ids=["num_items_in_batch", "plain_mean"])
def test_callback_records_one_row_per_optimizer_step(tmp_path, plain):
    cfg = TrainScopeConfig(run_dir=str(tmp_path), run_name="hf_run", track_memory=False)
    trainer = _tiny_trainer(tmp_path, TrainScopeCallback(cfg), plain=plain)
    trainer.train()

    rows = read_arrow_rows_sync(tmp_path / "hf_run" / "global.arrow")
    assert [r["step"] for r in rows] == list(range(6))
    assert all(math.isfinite(r["loss"]) and r["loss"] > 0 for r in rows)
    assert all(r["grad_norm_before_clip"] > 0 for r in rows)
    # Mean of recorded per-step losses matches the Trainer's own train_loss.
    train_loss = trainer.state.log_history[-1]["train_loss"]
    assert sum(r["loss"] for r in rows) / len(rows) == pytest.approx(train_loss, rel=1e-4)


def test_stop_on_spike_stops_training_gracefully(tmp_path):
    cfg = TrainScopeConfig(
        run_dir=str(tmp_path),
        run_name="hf_stop",
        track_memory=False,
        stop_on_spike=True,
        detector={"name": "z_score", "threshold": 0.0, "min_observations": 2},
    )
    trainer = _tiny_trainer(tmp_path, TrainScopeCallback(cfg), max_steps=20)
    trainer.train()
    assert trainer.state.global_step < 20
