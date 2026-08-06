#!/usr/bin/env python3
"""
TrainScope Real-Life End-to-End Showcase Script

Trains a 4-layer Transformer model, simulates a multi-stage training failure cascade
(Stable convergence -> Subtle Loss Drift -> Gradient Explosion -> NaN Collapse),
saves full per-layer Arrow artifacts, and automatically launches the TrainScope UI server
at http://localhost:7007 for live inspection and demo recording.

Usage:
    python scripts/run_realife_demo.py
"""

import os
import sys
import time
import subprocess
import signal
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainscope import TrainScope
from trainscope.core.config import TrainScopeConfig

# Hyperparameters
VOCAB = 256
SEQ_LEN = 64
BATCH = 16
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 4
N_STEPS = 150
DRIFT_START = 80
SPIKE_STEP = 100
LR = 1e-3


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        self.n_heads = N_HEADS
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(SEQ_LEN, SEQ_LEN)).view(1, 1, SEQ_LEN, SEQ_LEN),
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(D_MODEL, dim=2)
        head = C // self.n_heads
        q = q.view(B, T, self.n_heads, head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (head ** 0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.mlp = nn.Sequential(
            nn.Linear(D_MODEL, 4 * D_MODEL),
            nn.GELU(),
            nn.Linear(4 * D_MODEL, D_MODEL),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, D_MODEL)
        self.pos_emb = nn.Embedding(SEQ_LEN, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.head(h)


def main():
    print("=" * 70)
    print(" 🚀 TrainScope Real-Life End-to-End Showcase ")
    print("=" * 70)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerLM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    run_dir = "./trainscope_runs/realife_demo_run"
    config = TrainScopeConfig(
        run_dir=run_dir,
        spike_threshold=3.5,
        stop_on_spike=False,
        full_resolution_window=500,
        histogram_every_n_steps=10,
        activation_metrics_every_n_steps=2,
        track_memory=True,
    )

    scope = TrainScope(model, optimizer, config=config).attach()
    scope.writer.write_meta(
        "TransformerLM-4L",
        {"vocab": VOCAB, "d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS},
    )

    print(f"\n[1/3] Training 4-Layer Transformer ({N_STEPS} steps)...")
    print(f"      Run Directory: {config.run_name}\n")

    for step in range(N_STEPS):
        x = torch.randint(0, VOCAB, (BATCH, SEQ_LEN), device=device)
        targets = torch.randint(0, VOCAB, (BATCH, SEQ_LEN), device=device)

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB), targets.view(-1))

        # Phase 2: Inject gradual loss drift (Steps 80-99)
        if DRIFT_START <= step < SPIKE_STEP:
            loss = loss * (1.0 + 0.15 * (step - DRIFT_START + 1))

        # Phase 3: Inject gradient explosion & spike (Step 100)
        if step == SPIKE_STEP:
            loss = loss * 100.0
            # Induce gradient explosion in last block
            for p in model.blocks[-1].parameters():
                if p.grad is not None:
                    p.grad.data.mul_(50.0)

        loss.backward()

        # Step recorder
        spike = scope.step(loss.item(), batch_index=step)
        optimizer.step()

        if spike:
            print(
                f"  ⚠️  [DETECTION] Step {step:3d} | Loss: {loss.item():8.4f} | "
                f"Anomaly Score (z): {spike.get('z_score', 0):6.2f}"
            )
        elif step % 20 == 0:
            print(f"  step {step:3d} | Loss: {loss.item():7.4f}")

    scope.writer.flush()
    scope.writer.close()
    scope.detach()

    print("\n[2/3] Run artifacts successfully generated & persisted to Arrow format.")

    # Start UI server automatically
    print("\n[3/3] Launching TrainScope UI Server on http://localhost:7007 ...")
    print("=" * 70)
    print(" 🌟 Open your browser & record your demo GIF:")
    print(" 👉 http://localhost:7007")
    print("=" * 70)
    print(" Press Ctrl+C anytime to stop the UI server.\n")

    cmd = [
        sys.executable,
        "-m",
        "trainscope.cli",
        "ui",
        "--run",
        os.path.join(config.run_dir, config.run_name),
        "--port",
        "7007",
    ]

    try:
        proc = subprocess.Popen(cmd)
        proc.wait()
    except KeyboardInterrupt:
        print("\nStopping UI server...")
        if proc.poll() is None:
            proc.terminate()
            proc.wait()
        print("Done.")


if __name__ == "__main__":
    main()
