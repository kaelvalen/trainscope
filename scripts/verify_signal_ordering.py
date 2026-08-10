#!/usr/bin/env python3
"""Empirical verification of the signal-ordering claim.

Four signals have been verified independently (CUSUM/loss, kurtosis/
activations, routing concentration, gradient norm), each with its own
early-warning lead (9-11, 14-18, 4-12, 7-11 steps). The open question:
do these signals *predict each other* — is there a mechanical order of
failure — or are they independent?

This script measures all four signals in the SAME organic run (a single
mini MoE+memory transformer under an LR ramp on wikitext-2) and reports,
per seed:

  - the first step each signal durably crosses its threshold (the same
    robust "control max + margin" definition used in earlier experiments),
  - the lead of each signal (explosion - first_crossing),
  - the pairwise order of the signals (which fires first).

Signals measured per step:
  - loss CUSUM: the real ChangePointDetector (threshold 6.0, slack 1.0)
  - activation kurtosis: excess kurtosis of block outputs (baseline
    median + 3*MAD, same rule as verify_kurtosis_early_warning.py)
  - gradient norm: L2 norm of all grads (baseline median + 3*MAD)
  - routing concentration: max expert share (threshold 0.85, same as the
    MoE experiment)

If the order is consistent across seeds (e.g. kurtosis always precedes
CUSUM), that supports a mechanical cascade and suggests the UI's Spike
Inspector should present the earliest signal first. If the order flips
between seeds, the signals are independent indicators.

Usage:
    python scripts/verify_signal_ordering.py [--seeds 1,7,42] [--data PATH]
"""

import argparse
import time

import numpy as np
import pyarrow.ipc as ipc
import torch
import torch.nn as nn
import torch.nn.functional as F

SEQ_LEN = 64
BATCH = 12
D_MODEL = 96
N_HEADS = 4
N_LAYERS = 2
N_EXPERTS = 4
TOP_K = 2
D_FF = 192
N_SLOTS = 16

WARMUP = 120
RAMP_FACTOR = 1.2
N_STEPS = 260
BASE_LR = 1e-3

# Robust "crossing" rule shared by all four signals: the signal must exceed
# baseline_median + MARGIN_K * baseline_MAD for MIN_RUN consecutive steps.
MARGIN_K = 3.0
MIN_RUN = 3

DEFAULT_DATA = (
    "/home/kael/.cache/huggingface/datasets/Salesforce___wikitext/"
    "wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/"
    "wikitext-train.arrow"
)


class ExpertFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


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


class MemoryBank(nn.Module):
    def __init__(self, d_model, n_slots):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(n_slots, d_model) * 0.1)

    def read(self, weights):
        return torch.einsum("...s,sd->...d", weights, self.slots)


class HybridBlock(nn.Module):
    """Block with MoE FFN (router) + memory read (addressor)."""

    def __init__(self, d_model, n_heads, n_experts, top_k, d_ff, n_slots):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([ExpertFFN(d_model, d_ff) for _ in range(n_experts)])
        self.top_k = top_k
        self.ln3 = nn.LayerNorm(d_model)
        self.addressor = nn.Linear(d_model, n_slots)
        self.memory = MemoryBank(d_model, n_slots)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        h = self.ln2(x)
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

        # Memory read.
        weights = F.softmax(self.addressor(self.ln3(x)), dim=-1)
        x = x + self.memory.read(weights)
        return x, probs.detach().cpu().numpy()


class HybridLM(nn.Module):
    def __init__(self, vocab, d_model, n_heads, n_experts, top_k, d_ff, n_slots):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(SEQ_LEN, d_model)
        self.blocks = nn.ModuleList(
            [
                HybridBlock(d_model, n_heads, n_experts, top_k, d_ff, n_slots)
                for _ in range(N_LAYERS)
            ]
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


def excess_kurtosis(act: torch.Tensor) -> float:
    flat = act.detach().float().flatten()
    mean = flat.mean()
    std = flat.std(unbiased=False)
    if std.item() == 0.0:
        return 0.0
    normalized = (flat - mean) / std
    return float((normalized**4).mean().item()) - 3.0


def train(seed, vocab, tokens, scenario, n_steps=N_STEPS):
    """Run the hybrid model; return per-step signal dicts."""
    from trainscope.core.detectors.changepoint import ChangePointDetector

    torch.manual_seed(seed)
    model = HybridLM(vocab, D_MODEL, N_HEADS, N_EXPERTS, TOP_K, D_FF, N_SLOTS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.0)
    cusum = ChangePointDetector(threshold=6.0, slack=1.0, window=200, min_observations=30)

    losses: list[float] = []
    kurtosis: list[float] = []
    grad_norms: list[float] = []
    concentration: list[float] = []
    cusum_scores: list[float] = []

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
        gn = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        optimizer.step()

        losses.append(loss.item())
        kurtosis.append(max(excess_kurtosis(b.attn.proj.weight) for b in model.blocks))
        grad_norms.append(gn)
        shares = []
        for probs in routing:
            argmax = probs.argmax(axis=-1)
            counts = np.bincount(argmax, minlength=N_EXPERTS).astype(float)
            shares.append((counts / counts.sum()).max())
        concentration.append(max(shares))
        cusum_scores.append(cusum.update(loss.item()) or 0.0)

        if not torch.isfinite(loss):
            break

    return {
        "loss": losses,
        "kurtosis": kurtosis,
        "grad_norm": grad_norms,
        "concentration": concentration,
        "cusum": cusum_scores,
    }


def _first_durable_crossing(
    signal: list[float], baseline: list[float], k: float, run: int, start: int = 0
) -> int | None:
    """First step >= start where signal exceeds baseline median + k*MAD for `run` steps."""
    arr = np.array(signal, dtype=float)
    base = np.array(baseline, dtype=float)
    med = np.median(base)
    mad = np.median(np.abs(base - med))
    threshold = med + k * mad
    run_count = 0
    for i in range(start, len(arr)):
        if np.isfinite(arr[i]) and arr[i] > threshold:
            run_count += 1
        else:
            run_count = 0
        if run_count >= run:
            return i - run + 1
    return None


def analyze(seed, scenario, signals):
    base_loss = np.array(signals["loss"][WARMUP - 40 : WARMUP])
    base_mean = float(base_loss.mean())

    explosion = None
    for i, loss in enumerate(signals["loss"]):
        if not np.isfinite(loss) or loss > 10.0 * base_mean:
            explosion = i
            break

    if explosion is None:
        return {"seed": seed, "scenario": scenario, "status": "no explosion"}

    # Baseline windows (last 40 warmup steps) per signal.
    base_windows = {
        "kurtosis": signals["kurtosis"][WARMUP - 40 : WARMUP],
        "grad_norm": signals["grad_norm"][WARMUP - 40 : WARMUP],
        "concentration": signals["concentration"][WARMUP - 40 : WARMUP],
    }

    # CUSUM fires when its own score >= 6 (its own detector semantics).
    cusum_fire = next((i for i in range(WARMUP, explosion) if signals["cusum"][i] >= 6.0), None)
    # Other signals use the robust crossing rule, scanned only after warmup
    # (baseline window ends at WARMUP; earlier crossings are warmup transients).
    fires = {
        "kurtosis": _first_durable_crossing(
            signals["kurtosis"], base_windows["kurtosis"], MARGIN_K, MIN_RUN, start=WARMUP
        ),
        "grad_norm": _first_durable_crossing(
            signals["grad_norm"], base_windows["grad_norm"], MARGIN_K, MIN_RUN, start=WARMUP
        ),
        "concentration": _first_durable_crossing(
            signals["concentration"],
            base_windows["concentration"],
            MARGIN_K,
            MIN_RUN,
            start=WARMUP,
        ),
        "cusum": cusum_fire,
    }

    leads = {name: (explosion - step) if step is not None else None for name, step in fires.items()}

    # Order by fire step (earliest first), skipping None.
    ordered = sorted(
        [(name, fires[name]) for name in fires if fires[name] is not None],
        key=lambda x: x[1],
    )
    order = [name for name, _ in ordered]

    return {
        "seed": seed,
        "scenario": scenario,
        "status": "explosion",
        "explosion": explosion,
        "fires": fires,
        "leads": leads,
        "order": order,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default="1,7,42")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA)
    args = parser.parse_args()

    tokens, vocab = load_wikitext_chars(args.data)
    print(
        f"wikitext-2: {len(tokens)} chars, vocab={vocab}, experts={N_EXPERTS} slots={N_SLOTS}",
        flush=True,
    )

    seeds = [int(s) for s in args.seeds.split(",")]
    results = []
    for seed in seeds:
        t0 = time.time()
        signals = train(seed, vocab, tokens, "lr_ramp")
        r = analyze(seed, "lr_ramp", signals)
        r["train_seconds"] = round(time.time() - t0, 1)
        results.append(r)
        print(r, flush=True)

    print("\n=== summary ===")
    explosions = [r for r in results if r["status"] == "explosion"]
    if not explosions:
        print("no explosions produced")
        return

    for signal in ["cusum", "kurtosis", "grad_norm", "concentration"]:
        leads = [r["leads"][signal] for r in explosions if r["leads"].get(signal) is not None]
        if leads:
            print(f"{signal}: leads {min(leads)}-{max(leads)} (mean {np.mean(leads):.1f})")
        else:
            print(f"{signal}: never fired before explosion")

    orders = [tuple(r["order"]) for r in explosions]
    print("\nper-seed order (earliest -> latest):")
    for r in explosions:
        print(f"  seed {r['seed']}: {' -> '.join(r['order'])}")

    # Is the order consistent across seeds?
    distinct = set(orders)
    if len(distinct) == 1:
        print(f"\norder is CONSISTENT across all seeds: {' -> '.join(orders[0])}")
    else:
        print(f"\norder VARIES across seeds ({len(distinct)} distinct orders)")


if __name__ == "__main__":
    main()
