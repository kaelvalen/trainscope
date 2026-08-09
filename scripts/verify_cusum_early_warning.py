#!/usr/bin/env python3
"""Empirical verification of CUSUM's early-warning claim (README).

The README claims the change-point detector "catches subtle, persistent loss
drifts 5-10 steps before loss explodes". This script tests that claim on a
real, organic loss explosion: a mini GPT-2 trained on wikitext-2 whose loss
genuinely diverges after a learning-rate ramp crosses the stability
threshold. The detector sees only the loss stream.

Measures, per seed:
  - first CUSUM detection step
  - explosion step (loss > 10x baseline mean, or non-finite)
  - early-warning lead (explosion - detection)
  - pre-explosion drift rate in sigma/step

Usage:
    python scripts/verify_cusum_early_warning.py [--seeds 1,2,3] [--data PATH]
"""

import argparse
import time

import numpy as np
import pyarrow.ipc as ipc
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainscope.core.detectors.changepoint import ChangePointDetector

SEQ_LEN = 64
BATCH = 12
D_MODEL = 96
N_HEADS = 4
N_LAYERS = 2

WARMUP = 120
RAMP_FACTOR = 1.2
N_STEPS = 260
BASE_LR = 1e-3

DEFAULT_DATA = (
    "/home/kael/.cache/huggingface/datasets/Salesforce___wikitext/"
    "wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/"
    "wikitext-train.arrow"
)


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
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.head(h)


def load_wikitext_chars(path, max_chars=600_000):
    """Char-level tokenizer over wikitext-2 raw text (arrow cache)."""
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


def train_with_ramp(seed, vocab, tokens):
    """Train with an LR ramp after warmup; return (losses, grad_norms)."""
    torch.manual_seed(seed)
    model = TransformerLM(vocab, D_MODEL, N_HEADS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.0)
    losses, grad_norms = [], []

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
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1))
        loss.backward()
        grad_norm = (
            sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        )
        optimizer.step()

        losses.append(loss.item())
        grad_norms.append(grad_norm)

        if not torch.isfinite(loss) or not torch.isfinite(torch.tensor(grad_norm)):
            break

    return losses, grad_norms


def analyze(seed, losses):
    """Run the real ChangePointDetector over the loss stream; measure the claim."""
    baseline = np.array(losses[WARMUP - 40 : WARMUP])
    base_mean = float(baseline.mean())
    base_std = float(baseline.std())

    detector = ChangePointDetector(threshold=6.0, slack=1.0, window=200, min_observations=30)
    first_detection = None
    for i, loss in enumerate(losses):
        if detector.update(loss) is not None:
            first_detection = i
            break

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
            "first_detection": first_detection,
        }

    drift_window = [(losses[i] - base_mean) / base_std for i in range(WARMUP, explosion)]
    drift_rate = (
        float(np.polyfit(np.arange(len(drift_window)), drift_window, 1)[0])
        if len(drift_window) >= 3
        else None
    )

    return {
        "seed": seed,
        "status": "explosion",
        "first_detection": first_detection,
        "explosion_step": explosion,
        "lead": (explosion - first_detection) if first_detection is not None else None,
        "drift_rate_sigma": drift_rate,
        "loss_at_detection": float(losses[first_detection])
        if first_detection is not None
        else None,
        "baseline": round(base_mean, 3),
        "baseline_std": round(base_std, 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default="1,7,42")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA)
    args = parser.parse_args()

    tokens, vocab = load_wikitext_chars(args.data)
    print(f"wikitext-2: {len(tokens)} chars, vocab={vocab}", flush=True)

    results = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        t0 = time.time()
        losses, _ = train_with_ramp(seed, vocab, tokens)
        result = analyze(seed, losses)
        result["train_seconds"] = round(time.time() - t0, 1)
        results.append(result)
        print(result, flush=True)

    explosions = [r for r in results if r["status"] == "explosion"]
    print("\n=== summary ===")
    print(f"seeds: {args.seeds}, explosions: {len(explosions)}/{len(results)}")
    leads = [r["lead"] for r in explosions if r.get("lead") is not None]
    rates = [r["drift_rate_sigma"] for r in explosions if r.get("drift_rate_sigma") is not None]
    if leads:
        print(
            f"early-warning lead (steps): min={min(leads)} max={max(leads)} mean={np.mean(leads):.1f}"
        )
    if rates:
        print(f"pre-explosion drift rate (sigma/step): {[round(r, 2) for r in rates]}")
    print(f"all leads >= 5: {all(lead >= 5 for lead in leads) if leads else False}")


if __name__ == "__main__":
    main()
