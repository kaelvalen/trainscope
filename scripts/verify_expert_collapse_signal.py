#!/usr/bin/env python3
"""Empirical verification of the expert-collapse early-warning signal.

Phase 2 (architecture-aware diagnostics) must be proven before it is built,
exactly like the CUSUM claim was. This script trains a small Mixtral-style
MoE (4 experts, top-2 routing) on wikitext-2 and asks: does expert
utilization collapse *before* the loss degrades, and by how many steps?

Two conditions are exercised, following the CUSUM experiment's methodology
(the same objective explosion definition — loss > 10x baseline mean or
non-finite — independent of any detector):

1. LR-ramp divergence: the learning rate is raised multiplicatively after
   warmup until training diverges. Does routing concentration precede the
   loss explosion?
2. Stable control: identical setup without the LR ramp. Collapse must NOT
   happen here, or the signal is not specific to instability.

Measured per step: per-expert utilization (fraction of tokens routed to each
expert) and max-expert share. The collapse point is the first step where
max-expert share durably exceeds a high threshold (COLLAPSE_SHARE for
COLLAPSE_RUN consecutive steps); the lead is collapse_step - explosion_step.

Caveats established while designing this experiment:
- Scaling router logits ("softmax sharpening") cannot change routing at all
  for positive scales — argmax (and therefore utilization) is invariant, so
  that pathology was discarded as a failure mechanism.
- A single expert dropping to low share ("dead expert") is NOT a collapse
  signal: it also occurs in the stable control (top-2 of 4 naturally leaves
  one expert near-zero), so it is not specific to instability. Only
  concentration toward a single dominant expert (max share > 0.85) is
  specific.

Usage:
    python scripts/verify_expert_collapse_signal.py [--seeds 1,7,42] [--data PATH]
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
N_EXPERTS = 4
TOP_K = 2
D_FF = 192

WARMUP = 120
RAMP_FACTOR = 1.2
N_STEPS = 260
BASE_LR = 1e-3

# A router is considered "collapsed" once max-expert share durably exceeds
# this fraction of tokens for COLLAPSE_RUN consecutive steps.
COLLAPSE_SHARE = 0.85
COLLAPSE_RUN = 3
# An expert with share below this for the whole window is "dead".
DEAD_SHARE = 0.02


class ExpertFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


class MoEBlock(nn.Module):
    """Mixtral-style block: attention + MoE FFN with top-k routing."""

    def __init__(self, d_model, n_heads, n_experts, top_k, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([ExpertFFN(d_model, d_ff) for _ in range(n_experts)])
        self.top_k = top_k

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        h = self.ln2(x)
        # Router over the flattened token dimension.
        logits = self.router(h).reshape(-1, len(self.experts))
        probs = F.softmax(logits, dim=-1)
        top_probs, top_idx = torch.topk(probs, self.top_k, dim=-1)
        out = torch.zeros_like(h.reshape(-1, h.shape[-1]))
        for k in range(self.top_k):
            chosen = top_idx[:, k]
            weight = top_probs[:, k].unsqueeze(-1)
            for e_idx, expert in enumerate(self.experts):
                mask = chosen == e_idx
                if mask.any():
                    out[mask] += weight[mask] * expert(h.reshape(-1, h.shape[-1])[mask])
        x = x + out.reshape(h.shape)
        return x, probs.detach().cpu().numpy()


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


class MoELM(nn.Module):
    def __init__(self, vocab, d_model, n_heads, n_experts, top_k, d_ff):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(SEQ_LEN, d_model)
        self.blocks = nn.ModuleList(
            [MoEBlock(d_model, n_heads, n_experts, top_k, d_ff) for _ in range(N_LAYERS)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        routing = []
        for block in self.blocks:
            h, probs = block(h)
            routing.append(probs)
        h = self.ln_f(h)
        return self.head(h), routing


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


def utilization_from_routing(routing, n_experts):
    """Per-block max-expert share and dead-expert count from router probs."""
    per_block = []
    for probs in routing:
        # Fraction of tokens whose argmax routes to each expert.
        argmax = probs.argmax(axis=-1)
        counts = np.bincount(argmax, minlength=n_experts).astype(float)
        share = counts / counts.sum()
        per_block.append(share)
    return per_block  # list of per-block share vectors


def train(seed, vocab, tokens, scenario, n_steps=N_STEPS):
    """Run one MoE training scenario; return (losses, max_share, dead_counts)."""
    torch.manual_seed(seed)
    model = MoELM(vocab, D_MODEL, N_HEADS, N_EXPERTS, TOP_K, D_FF)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.0)
    losses: list[float] = []
    max_share: list[float] = []
    dead_counts: list[int] = []

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
        logits, routing = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1))
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        shares = utilization_from_routing(routing, N_EXPERTS)
        max_share.append(max(s.max() for s in shares))
        dead_counts.append(sum(int((s < DEAD_SHARE).sum() >= 1) for s in shares))

        if not torch.isfinite(loss):
            break

    return losses, max_share, dead_counts


def analyze(seed, scenario, losses, max_share, dead_counts):
    base = np.array(losses[WARMUP - 40 : WARMUP])
    base_mean = float(base.mean())

    explosion = None
    for i, loss in enumerate(losses):
        if not np.isfinite(loss) or loss > 10.0 * base_mean:
            explosion = i
            break

    # Durable collapse: max-expert share > COLLAPSE_SHARE for COLLAPSE_RUN
    # consecutive steps (after warmup, before/at explosion).
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

    # Any expert effectively dead (share < DEAD_SHARE) across the whole step?
    dead_at = None
    for i in range(WARMUP, limit):
        if dead_counts[i] > 0:
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
        "dead_expert_at": dead_at,
        "max_share_peak": float(np.nanmax(max_share[WARMUP:])) if WARMUP < len(max_share) else None,
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
        f"wikitext-2: {len(tokens)} chars, vocab={vocab}, experts={N_EXPERTS} top_k={TOP_K}",
        flush=True,
    )

    seeds = [int(s) for s in args.seeds.split(",")]
    results = []
    # A control condition is mandatory: routing collapse is only an early
    # warning if it does NOT happen in a stable run. Otherwise the signal is
    # just normal MoE behavior and Phase 2 has no claim to build on.
    for scenario in ["stable_control", "lr_ramp"]:
        print(f"\n=== scenario: {scenario} ===", flush=True)
        for seed in seeds:
            t0 = time.time()
            losses, max_share, dead_counts = train(seed, vocab, tokens, scenario)
            r = analyze(seed, scenario, losses, max_share, dead_counts)
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
