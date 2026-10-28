"""Synthetic runs for ``trainscope demo``.

Trains a few tiny transformers on a learnable synthetic sequence task: some
stay stable, others diverge organically under a learning-rate ramp. Together
they give every UI view (loss, layers, spike story, run clustering, stable
counterexample) real data to show, in seconds on a CPU.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainscope.core.config import TrainScopeConfig
from trainscope.scope import TrainScope

VOCAB, SEQ, BATCH, D_MODEL = 64, 32, 16, 64
BASE_LR = 1e-3
WARMUP = 80
N_STEPS = 160
# (run name, seed, LR multiplier per step after WARMUP; None = constant LR)
RUNS = [
    ("stable-seed0", 0, None),
    ("stable-seed1", 1, None),
    ("lr-ramp-seed0", 0, 1.15),
    ("lr-ramp-seed1", 1, 1.15),
]


class _Attention(nn.Module):
    # Leaf Linear projections (not nn.MultiheadAttention, whose weights sit on a
    # non-leaf module) so every parameter has activation metrics in the UI.
    def __init__(self, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = (
            t.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
            for t in self.qkv(x).split(C, dim=2)
        )
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = _Attention()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.mlp = nn.Sequential(
            nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(), nn.Linear(4 * D_MODEL, D_MODEL)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class DemoLM(nn.Module):
    def __init__(self, n_layers: int = 2):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, D_MODEL)
        self.pos_emb = nn.Embedding(SEQ, D_MODEL)
        self.blocks = nn.Sequential(*[_Block() for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB)

    def forward(self, x):
        h = self.tok_emb(x) + self.pos_emb(torch.arange(x.shape[1]))
        return self.head(self.ln_f(self.blocks(h)))


def _batch(gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    # Arithmetic sequences with 10% token noise: learnable, never trivially zero-loss.
    start = torch.randint(0, VOCAB, (BATCH, 1), generator=gen)
    seq = (start + 3 * torch.arange(SEQ + 1)) % VOCAB
    noise = torch.rand(seq.shape, generator=gen) < 0.1
    seq = torch.where(noise, torch.randint(0, VOCAB, seq.shape, generator=gen), seq)
    return seq[:, :-1], seq[:, 1:]


def train_demo_run(root: Path, name: str, seed: int, lr_ramp: float | None) -> dict:
    """Train one demo run into ``root/name``; return a short summary."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = DemoLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.0)
    config = TrainScopeConfig(run_dir=str(root), run_name=name, track_memory=False)
    scope = TrainScope(model, optimizer, config).attach()

    # Record the hyperparameters that differ between runs so the Runs view can
    # name them as the distinguishing trait of the diverging cluster.
    meta_path = root / name / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["model_config"] = {"lr": BASE_LR, "lr_ramp": lr_ramp or 1.0, "seed": seed}
    meta_path.write_text(json.dumps(meta, indent=2))

    spikes, first_spike = 0, None
    for step in range(N_STEPS):
        if lr_ramp is not None and step >= WARMUP:
            for group in optimizer.param_groups:
                group["lr"] = BASE_LR * lr_ramp ** (step - WARMUP)
        x, y = _batch(gen)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        loss.backward()
        if scope.step(loss.item(), batch_index=step) is not None:
            spikes += 1
            first_spike = step if first_spike is None else first_spike
        # Past the spike window there is nothing left to learn from a blown-up run.
        if not math.isfinite(loss.item()) or (first_spike is not None and step >= first_spike + 15):
            break
        optimizer.step()
    scope.detach()
    return {"name": name, "steps": step + 1, "spikes": spikes, "first_spike": first_spike}


def generate_demo_runs(root: Path) -> list[dict]:
    root.mkdir(parents=True, exist_ok=True)
    return [train_demo_run(root, name, seed, ramp) for name, seed, ramp in RUNS]
