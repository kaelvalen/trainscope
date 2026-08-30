#!/usr/bin/env python3
"""Empirical verification of the attention-collapse early-warning signal.

Phase 2's third candidate claim, tested with the same methodology as the MoE
(``verify_expert_collapse_signal.py``) and addressor
(``verify_addressor_collapse_signal.py``) experiments: a small transformer
(2 layers, 4 heads) is trained on wikitext-2, and we ask whether *attention*
concentration or *attention* uniformization — the failure modes of the
"lazy-head" / rank-collapse class — precede the loss divergence.

Two conditions, identical to the previous experiments:

1. LR-ramp divergence: learning rate raised multiplicatively after warmup
   until training diverges. Does an attention signal precede the loss
   explosion?
2. Stable control: identical setup without the ramp. The signal must NOT
   occur here, or it is not specific to instability.

Two candidate statistics are measured per step, per head (mean over token
positions, then the worst head across all blocks — the maximum over heads,
mirroring the max-share conventions of the MoE/addressor experiments):

- ``max_p`` — the largest attention weight. A head "locking onto" a single
  token pushes this toward 1.0 (concentrated/peaky collapse).
- ``norm_entropy`` — Shannon entropy normalized by the uniform entropy
  ``log(T)``. A head that stops attending (lazy-head / rank-collapse)
  pushes this toward 1.0, the fully uniform distribution.

The collapse point is the first step where either statistic durably exceeds
its threshold for ``COLLAPSE_RUN`` consecutive steps; the lead is
collapse_step - explosion_step. Explosion is defined objectively (loss >
10x baseline mean, or non-finite), independent of any detector.

Threshold choice: the candidate thresholds below (``MAXP_COLLAPSE`` and
``ENTROPY_COLLAPSE``) follow the "control max + margin" rule established in
the previous experiments, but are NOT yet calibrated: the rule requires
running the stable control first and setting each threshold above the
measured healthy ceiling. The summary therefore prints each condition's
observed max for both statistics so the thresholds can be revised against
real data before this signal is promoted to production.

Caveats established while designing this experiment:
- A lazy head is NOT necessarily a collapse signal on its own: with four
  heads, one head drifting toward uniform attention may be normal network
  specialization (the MoE "dead expert" finding generalized to attention).
  Only a *durable* concentration or uniformization that separates the
  conditions is a candidate signal.
- Causal masking biases raw entropy: early positions attend to one token
  (entropy 0), so normalization uses ``log(T)`` and the mean over positions
  smooths the bias identically in both conditions.

Usage:
    python scripts/verify_attention_collapse_signal.py [--seeds 1,7,42] [--data PATH]
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
D_FF = 192

WARMUP = 120
RAMP_FACTOR = 1.2
N_STEPS = 260
BASE_LR = 1e-3

# Candidate collapse thresholds (see module docstring: "control max +
# margin", to be calibrated against the control's observed ceiling).
MAXP_COLLAPSE = 0.85
ENTROPY_COLLAPSE = 0.95
COLLAPSE_RUN = 3

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
        # Populated every forward with the raw attention matrix (B, H, T, T),
        # detached, so the caller can measure concentration per head.
        self.last_attn = None

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
        self.last_attn = att.detach().cpu().numpy()
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniTransformerLM(nn.Module):
    def __init__(self, vocab, d_model, n_heads, d_ff):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(SEQ_LEN, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(N_LAYERS)]
        )
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

    def last_attention(self):
        """Per-block attention matrices from the most recent forward."""
        return [b.attn.last_attn for b in self.blocks]


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


def attention_signals(attention_mats):
    """Per-step (max_p, norm_entropy) from per-block attention matrices.

    Each matrix is (B, H, T, T). We reduce over token positions (mean) and
    then take the worst head across every block and batch item — the max,
    mirroring the max-share convention of the MoE/addressor experiments so a
    single collapsing head is caught rather than averaged away.
    """
    max_ps = []
    norm_ents = []
    for attn in attention_mats:
        if attn is None:
            continue
        B, H, T, _ = attn.shape
        eps = 1e-12
        # Mean over token positions of the per-token max attention weight.
        max_p = attn.max(axis=-1).mean(axis=-1)  # (B, H)
        # Shannon entropy (nats) per (B, H, T), then mean over positions.
        ent = -(attn * np.log(attn + eps)).sum(axis=-1)  # (B, H, T)
        norm_ent = ent.mean(axis=-1) / np.log(T)  # (B, H), 1.0 = uniform
        max_ps.append(max_p)
        norm_ents.append(norm_ent)
    if not max_ps:
        return 0.0, 0.0
    all_max = np.concatenate(max_ps, axis=1)  # (B, H_total)
    all_ent = np.concatenate(norm_ents, axis=1)
    return float(all_max.max()), float(all_ent.max())


def train(seed, vocab, tokens, scenario, n_steps=N_STEPS):
    """Run one attention training scenario; return losses + attention signals."""
    torch.manual_seed(seed)
    model = MiniTransformerLM(vocab, D_MODEL, N_HEADS, D_FF)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.0)
    losses: list[float] = []
    max_ps: list[float] = []
    norm_ents: list[float] = []

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
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1))
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        max_p, norm_ent = attention_signals(model.last_attention())
        max_ps.append(max_p)
        norm_ents.append(norm_ent)

        if not torch.isfinite(loss):
            break

    return losses, max_ps, norm_ents


def _first_durable_crossing(values, threshold, run_len, start, limit):
    """First index where ``values`` exceeds ``threshold`` for ``run_len``
    consecutive steps (after ``start``, before ``limit``), or None."""
    run = 0
    for i in range(start, limit):
        if values[i] > threshold:
            run += 1
        else:
            run = 0
        if run >= run_len:
            return i - run_len + 1
    return None


def analyze(seed, scenario, losses, max_ps, norm_ents):
    base = np.array(losses[WARMUP - 40 : WARMUP])
    base_mean = float(base.mean())

    explosion = None
    for i, loss in enumerate(losses):
        if not np.isfinite(loss) or loss > 10.0 * base_mean:
            explosion = i
            break

    limit = explosion if explosion is not None else len(losses)
    maxp_collapse = _first_durable_crossing(max_ps, MAXP_COLLAPSE, COLLAPSE_RUN, WARMUP, limit)
    entropy_collapse = _first_durable_crossing(
        norm_ents, ENTROPY_COLLAPSE, COLLAPSE_RUN, WARMUP, limit
    )

    return {
        "seed": seed,
        "scenario": scenario,
        "explosion": explosion,
        "maxp_collapse": maxp_collapse,
        "maxp_lead": (explosion - maxp_collapse)
        if (maxp_collapse is not None and explosion is not None)
        else None,
        "entropy_collapse": entropy_collapse,
        "entropy_lead": (explosion - entropy_collapse)
        if (entropy_collapse is not None and explosion is not None)
        else None,
        # Observed ceilings (control max + margin calibration data).
        "max_p_peak": round(float(np.nanmax(max_ps[WARMUP:])), 3) if WARMUP < len(max_ps) else None,
        "norm_entropy_peak": (
            round(float(np.nanmax(norm_ents[WARMUP:])), 3) if WARMUP < len(norm_ents) else None
        ),
        "baseline": round(base_mean, 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default="1,7,42")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA)
    args = parser.parse_args()

    tokens, vocab = load_wikitext_chars(args.data)
    print(
        f"wikitext-2: {len(tokens)} chars, vocab={vocab}, heads={N_HEADS} layers={N_LAYERS}",
        flush=True,
    )

    seeds = [int(s) for s in args.seeds.split(",")]
    results = []
    # A control condition is mandatory: attention concentration/uniformization
    # is only an early warning if it does NOT happen in a stable run.
    for scenario in ["stable_control", "lr_ramp"]:
        print(f"\n=== scenario: {scenario} ===", flush=True)
        for seed in seeds:
            t0 = time.time()
            losses, max_ps, norm_ents = train(seed, vocab, tokens, scenario)
            r = analyze(seed, scenario, losses, max_ps, norm_ents)
            r["train_seconds"] = round(time.time() - t0, 1)
            results.append(r)
            print(r, flush=True)

    print("\n=== summary ===")
    for scenario in ["stable_control", "lr_ramp"]:
        rows = [r for r in results if r["scenario"] == scenario]
        expl_count = sum(1 for r in rows if r["explosion"] is not None)
        mp_count = sum(1 for r in rows if r["maxp_collapse"] is not None)
        ent_count = sum(1 for r in rows if r["entropy_collapse"] is not None)
        mp_leads = [r["maxp_lead"] for r in rows if r.get("maxp_lead") is not None]
        ent_leads = [r["entropy_lead"] for r in rows if r.get("entropy_lead") is not None]
        mp_peaks = [r["max_p_peak"] for r in rows if r.get("max_p_peak") is not None]
        ent_peaks = [r["norm_entropy_peak"] for r in rows if r.get("norm_entropy_peak") is not None]
        print(
            f"{scenario}: explosions={expl_count}/{len(rows)}, "
            f"max_p collapse={mp_count}/{len(rows)}, entropy collapse={ent_count}/{len(rows)}"
        )
        if mp_leads:
            print(
                f"  max_p lead (steps): min={min(mp_leads)} max={max(mp_leads)} "
                f"mean={np.mean(mp_leads):.1f}"
            )
        else:
            print("  no max_p collapse detected")
        if ent_leads:
            print(
                f"  entropy lead (steps): min={min(ent_leads)} max={max(ent_leads)} "
                f"mean={np.mean(ent_leads):.1f}"
            )
        else:
            print("  no entropy collapse detected")
        # Control max + margin calibration data: the observed healthy ceilings.
        if mp_peaks:
            print(f"  max_p peak (control ceiling data): max={max(mp_peaks):.3f}")
        if ent_peaks:
            print(f"  norm_entropy peak (control ceiling data): max={max(ent_peaks):.3f}")


if __name__ == "__main__":
    main()
