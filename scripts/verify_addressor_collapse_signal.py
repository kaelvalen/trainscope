#!/usr/bin/env python3
"""Empirical verification of the addressor-collapse early-warning signal.

Phase 2's second claim, tested with the same methodology as the MoE
experiment (``verify_expert_collapse_signal.py``): a small
memory-augmented transformer with an addressor (softmax addressing over a
memory bank) is trained on wikitext-2, and we ask whether addressor
concentration — the model locking onto a single memory slot — precedes the
loss divergence.

Two conditions, identical to the MoE experiment:

1. LR-ramp divergence: learning rate raised multiplicatively after warmup
   until training diverges. Does address concentration precede the loss
   explosion?
2. Stable control: identical setup without the ramp. Concentration must
   NOT happen here, or the signal is not specific to instability.

Measured per step: per-slot mean addressing weight (mean over tokens of
softmax weights) and max-slot share. The collapse point is the first step
where max-slot share durably exceeds ``COLLAPSE_SHARE`` for
``COLLAPSE_RUN`` consecutive steps; the lead is collapse_step -
explosion_step.

Explosion is defined objectively (loss > 10x baseline mean, or non-finite),
independent of any detector, exactly as in the CUSUM/kurtosis/MoE claims.

Threshold choice (why 0.6 and not MoE's 0.85): the threshold is not
proportional to slot count but sits above the observed healthy ceiling of
the max-share signal. With 16 soft-addressed slots the stable control's
max-share peaks at 0.24-0.32 (measured below); 0.6 is a durable ~2x margin
above that. The MoE experiment used 0.85 because its healthy control
already peaks at 0.60-0.74 (top-2-of-4 routing naturally concentrates), so
a lower threshold would not separate the conditions. Both thresholds are
therefore "control max + margin", validated by running the control first.

Dead-slot signal (measured, rejected): a slot whose mean addressing weight
stays below 2% is NOT a collapse signal. With 16 slots such a slot exists
in *every* step of both the stable control (140/140 steps, 3/3 seeds) and
the LR-ramp condition (87-90/88-91) — identical to the MoE experiment's
dead-expert finding. Only concentration (one slot dominating) separates
the conditions, so only max-share is used for detection.

Usage:
    python scripts/verify_addressor_collapse_signal.py [--seeds 1,7,42] [--data PATH]
"""

import argparse
import time

import numpy as np
import pyarrow.ipc as ipc
import torch
import torch.nn as nn
import torch.nn.functional as F
from _wikitext import find_wikitext_arrow

SEQ_LEN = 64
BATCH = 12
D_MODEL = 96
N_HEADS = 4
N_LAYERS = 2
N_SLOTS = 16
D_FF = 192

WARMUP = 120
RAMP_FACTOR = 1.2
N_STEPS = 260
BASE_LR = 1e-3

COLLAPSE_SHARE = 0.6
COLLAPSE_RUN = 3
# A slot whose mean addressing weight stays below this is "dead" (measured
# and reported, but NOT used for detection — see the module docstring).
DEAD_SHARE = 0.02


class MemoryBank(nn.Module):
    """A soft-addressed memory bank: ``read(x) = W^T softmax-addressed slots``."""

    def __init__(self, d_model, n_slots):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(n_slots, d_model) * 0.1)

    def read(self, weights):
        # weights: (..., n_slots) -> (..., d_model)
        return torch.einsum("...s,sd->...d", weights, self.slots)


class AddressorBlock(nn.Module):
    """Transformer block with a soft-addressed memory read (addressor)."""

    def __init__(self, d_model, n_heads, n_slots, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.ln3 = nn.LayerNorm(d_model)
        self.addressor = nn.Linear(d_model, n_slots)
        self.memory = MemoryBank(d_model, n_slots)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        # Addressor: softmax over slots, read weighted memory, residual add.
        weights = F.softmax(self.addressor(self.ln3(x)), dim=-1)
        x = x + self.memory.read(weights)
        return x, weights.detach()


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.register_buffer(
            "mask", torch.tril(torch.ones(SEQ_LEN, SEQ_LEN)).view(1, 1, SEQ_LEN, SEQ_LEN)
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        head = C // self.n_heads
        q = q.view(B, T, self.n_heads, head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (head**0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class AddressorLM(nn.Module):
    def __init__(self, vocab, d_model, n_heads, n_slots, d_ff):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(SEQ_LEN, d_model)
        self.blocks = nn.ModuleList(
            [AddressorBlock(d_model, n_heads, n_slots, d_ff) for _ in range(N_LAYERS)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        weights = []
        for block in self.blocks:
            h, w = block(h)
            weights.append(w)
        h = self.ln_f(h)
        return self.head(h), weights


def load_wikitext_chars(path, max_chars=600_000):
    with open(path, "rb") as f:
        table = ipc.open_stream(f).read_all()
    texts = [t for t in table.column(0).to_pylist() if t.strip()]
    text = " ".join(texts)[:max_chars]
    chars = sorted(set(text))
    ctoi = {c: i for i, c in enumerate(chars)}
    tokens = np.array([ctoi[c] for c in text], dtype=np.int64)
    return tokens, len(chars)


def make_batches(tokens, n_steps):
    pos = 0
    while pos + BATCH * SEQ_LEN <= tokens.size - SEQ_LEN:
        block = tokens[pos : pos + BATCH * SEQ_LEN].reshape(BATCH, SEQ_LEN)
        x = torch.from_numpy(block[:, :-1]).contiguous()
        y = torch.from_numpy(block[:, 1:]).contiguous()
        yield x, y
        pos += BATCH * SEQ_LEN
    raise StopIteration


def train(seed, vocab, tokens, scenario, n_steps=N_STEPS):
    """Run one addressor scenario; return (losses, max_share, dead_slots)."""
    torch.manual_seed(seed)
    model = AddressorLM(vocab, D_MODEL, N_HEADS, N_SLOTS, D_FF)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.0)
    losses: list[float] = []
    max_share: list[float] = []
    dead_slots: list[bool] = []

    gen = make_batches(tokens, n_steps)
    for step in range(n_steps):
        try:
            x, y = next(gen)
        except StopIteration:
            break

        if scenario == "lr_ramp" and step >= WARMUP:
            multiplier = RAMP_FACTOR ** (step - WARMUP)
            for group in optimizer.param_groups:
                group["lr"] = BASE_LR * multiplier

        optimizer.zero_grad()
        logits, weights = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1))
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        # Mean over tokens of each block's slot weights; keep the max share.
        per_block_max = []
        per_block_min = []
        for w in weights:
            mean_w = w.reshape(-1, N_SLOTS).mean(dim=0)
            per_block_max.append(float(mean_w.max().item()))
            per_block_min.append(float(mean_w.min().item()))
        max_share.append(max(per_block_max))
        dead_slots.append(min(per_block_min) < DEAD_SHARE)

        if not torch.isfinite(loss):
            break

    return losses, max_share, dead_slots


def analyze(seed, scenario, losses, max_share, dead_slots):
    base = np.array(losses[WARMUP - 40 : WARMUP])
    base_mean = float(base.mean())

    explosion = None
    for i, loss in enumerate(losses):
        if not np.isfinite(loss) or loss > 10.0 * base_mean:
            explosion = i
            break

    collapse = None
    run = 0
    limit = explosion if explosion is not None else len(losses)
    for i in range(WARMUP, limit):
        if max_share[i] > COLLAPSE_SHARE:
            run += 1
        else:
            run = 0
        if run >= COLLAPSE_RUN:
            collapse = i - COLLAPSE_RUN + 1
            break

    # Dead-slot diagnostic: first step (post-warmup) with a slot below the
    # dead threshold. Measured but NOT used for detection — the experiment
    # showed it fires in the stable control too (see module docstring).
    dead_at = None
    for i in range(WARMUP, limit):
        if dead_slots[i]:
            dead_at = i
            break

    return {
        "seed": seed,
        "scenario": scenario,
        "explosion": explosion,
        "collapse": collapse,
        "lead": (explosion - collapse)
        if (collapse is not None and explosion is not None)
        else None,
        "max_share_peak": float(np.nanmax(max_share[WARMUP:])) if WARMUP < len(max_share) else None,
        "dead_slot_at": dead_at,
        "baseline": round(base_mean, 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default="1,7,42")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to wikitext-train.arrow (default: resolve from the HF cache)",
    )
    args = parser.parse_args()

    tokens, vocab = load_wikitext_chars(args.data or find_wikitext_arrow())
    print(
        f"wikitext-2: {len(tokens)} chars, vocab={vocab}, "
        f"slots={N_SLOTS}, collapse_share={COLLAPSE_SHARE}",
        flush=True,
    )

    seeds = [int(s) for s in args.seeds.split(",")]
    results = []
    for scenario in ["stable_control", "lr_ramp"]:
        print(f"\n=== scenario: {scenario} ===", flush=True)
        for seed in seeds:
            t0 = time.time()
            losses, max_share, dead_slots = train(seed, vocab, tokens, scenario)
            r = analyze(seed, scenario, losses, max_share, dead_slots)
            r["train_seconds"] = round(time.time() - t0, 1)
            results.append(r)
            print(r, flush=True)

    print("\n=== summary ===")
    for scenario in ["stable_control", "lr_ramp"]:
        rows = [r for r in results if r["scenario"] == scenario]
        expl_count = sum(1 for r in rows if r["explosion"] is not None)
        col_count = sum(1 for r in rows if r["collapse"] is not None)
        leads = [r["lead"] for r in rows if r.get("lead") is not None]
        print(f"{scenario}: explosions={expl_count}/{len(rows)}, collapse={col_count}/{len(rows)}")
        if leads:
            print(f"  lead (steps): min={min(leads)} max={max(leads)} mean={np.mean(leads):.1f}")
        else:
            print("  no collapse detected")


if __name__ == "__main__":
    main()
