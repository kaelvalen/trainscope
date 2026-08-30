#!/usr/bin/env python3
"""Empirical verification of the activation-kurtosis early-warning claim (README).

The README claims activation kurtosis "rises 1-5 steps before catastrophic
loss explosion". This script tests that claim with the same organic scenario
as ``verify_cusum_early_warning.py``: a mini GPT-2 trained on wikitext-2 whose
loss genuinely diverges after a learning-rate ramp crosses the stability
threshold. Per block, the excess kurtosis of the post-block activations is
computed every step (same formula as ``compute_activation_metrics``).

Measures, per seed:
  - first step where any block's kurtosis exceeds its warmup baseline by a
    robust margin (median + k·σ, σ = 1.4826·MAD)
  - explosion step (loss > 10x baseline mean, or non-finite) — same objective
    definition as the CUSUM experiment, independent of any detector
  - early-warning lead (explosion - first kurtosis rise)

Usage:
    python scripts/verify_kurtosis_early_warning.py [--seeds 1,7,42] [--data PATH]
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

WARMUP = 120
RAMP_FACTOR = 1.2
N_STEPS = 260
BASE_LR = 1e-3

# Kurtosis must exceed baseline_median + K * baseline_sigma to count as
# "risen", where sigma = 1.4826 * MAD (the robust sigma estimate the CUSUM
# detector and trainscope.analysis.first_crossing_step use).
KURTOSIS_MARGIN_K = 3.0
# A rise must occur after warmup; baseline is the last 40 warmup steps.
BASELINE_STEPS = 40


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


class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab, d_model, n_heads):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(SEQ_LEN, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        block_outputs = []
        for block in self.blocks:
            h = block(h)
            block_outputs.append(h.detach())
        h = self.ln_f(h)
        return self.head(h), block_outputs


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


def excess_kurtosis(act: torch.Tensor) -> float:
    flat = act.detach().float().flatten()
    mean = flat.mean()
    std = flat.std(unbiased=False)
    if std.item() == 0.0:
        return 0.0
    normalized = (flat - mean) / std
    return float((normalized**4).mean().item()) - 3.0


def train_with_ramp(seed, vocab, tokens):
    """Train with an LR ramp after warmup; return (losses, kurtosis per block)."""
    torch.manual_seed(seed)
    model = TransformerLM(vocab, D_MODEL, N_HEADS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.0)
    losses = []
    block_kurtosis = []  # list of per-step lists, one entry per block

    gen = make_batches(tokens, N_STEPS)
    for step in range(N_STEPS):
        try:
            x, y = next(gen)
        except StopIteration:
            break

        if step >= WARMUP:
            multiplier = RAMP_FACTOR ** (step - WARMUP)
            for group in optimizer.param_groups:
                group["lr"] = BASE_LR * multiplier

        optimizer.zero_grad()
        logits, block_outputs = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1))
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        block_kurtosis.append([excess_kurtosis(act) for act in block_outputs])

        if not torch.isfinite(loss):
            break

    return losses, block_kurtosis


def analyze(seed, losses, block_kurtosis):
    baseline = np.array(losses[WARMUP - BASELINE_STEPS : WARMUP])
    base_mean = float(baseline.mean())

    explosion = None
    for i, loss in enumerate(losses):
        if not np.isfinite(loss) or loss > 10.0 * base_mean:
            explosion = i
            break

    if explosion is None:
        return {
            "seed": seed,
            "status": "no explosion",
            "max_loss": float(max(losses)),
        }

    # Per-block kurtosis baseline: median and MAD over the last BASELINE_STEPS
    # of warmup (robust, mirrors the detector's median/MAD philosophy).
    per_block = np.array(block_kurtosis).T  # (n_blocks, n_steps)
    baseline_win = per_block[:, WARMUP - BASELINE_STEPS : WARMUP]
    baseline_med = np.median(baseline_win, axis=1)
    baseline_mad = np.median(np.abs(baseline_win - baseline_med[:, None]), axis=1)

    first_rise = None
    for step in range(WARMUP, explosion):
        for b in range(per_block.shape[0]):
            threshold = baseline_med[b] + KURTOSIS_MARGIN_K * baseline_mad[b] * 1.4826
            if np.isfinite(per_block[b, step]) and per_block[b, step] > threshold:
                first_rise = step
                break
        if first_rise is not None:
            break

    return {
        "seed": seed,
        "status": "explosion",
        "explosion_step": explosion,
        "first_kurtosis_rise": first_rise,
        "lead": (explosion - first_rise) if first_rise is not None else None,
        "baseline_kurtosis": [round(float(x), 2) for x in baseline_med],
        "baseline": round(base_mean, 3),
        "n_blocks": per_block.shape[0],
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
    parser.add_argument("--k", type=float, default=KURTOSIS_MARGIN_K)
    args = parser.parse_args()

    tokens, vocab = load_wikitext_chars(args.data or find_wikitext_arrow())
    print(f"wikitext-2: {len(tokens)} chars, vocab={vocab}, margin k={args.k}", flush=True)

    results = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        t0 = time.time()
        losses, block_kurtosis = train_with_ramp(seed, vocab, tokens)
        result = analyze(seed, losses, block_kurtosis)
        result["train_seconds"] = round(time.time() - t0, 1)
        results.append(result)
        print(result, flush=True)

    explosions = [r for r in results if r["status"] == "explosion"]
    leads = [r["lead"] for r in explosions if r.get("lead") is not None]
    print("\n=== summary ===")
    print(f"seeds: {args.seeds}, explosions: {len(explosions)}/{len(results)}")
    if leads:
        print(
            f"kurtosis early-warning lead (steps): min={min(leads)} max={max(leads)} mean={np.mean(leads):.1f}"
        )
        print(f"all leads in [1, 5]: {all(1 <= lead <= 5 for lead in leads)}")
    else:
        print("no kurtosis rise detected before any explosion")


if __name__ == "__main__":
    main()
