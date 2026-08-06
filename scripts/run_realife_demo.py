#!/usr/bin/env python3
"""
TrainScope Real-Life End-to-End Showcase Script

Trains a 4-layer Transformer model through a reproducible, noisy stress process,
saves full per-layer Arrow artifacts, and automatically launches the TrainScope UI
server at http://localhost:7007 for live inspection and demo recording.

Usage:
    python scripts/run_realife_demo.py
"""

import os
import random
import signal
import subprocess
import sys
import time

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
        att = (q @ k.transpose(-2, -1)) / (head**0.5)
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


def evolve_training_stress(stress: float, rng: random.Random) -> float:
    """Advance a noisy latent stress process without a scripted failure step."""
    return max(0.0, stress * 0.985 + rng.gauss(0.012, 0.018))


def apply_training_stress(loss: torch.Tensor, stress: float) -> torch.Tensor:
    """Model gradual distribution and optimizer instability as loss amplification."""
    drift = max(0.0, stress - 0.18)
    instability = max(0.0, drift - 0.35)
    multiplier = (
        1.0
        + 0.25 * drift
        + 1.2 * drift**2
        + torch.expm1(torch.tensor(3.0 * instability, device=loss.device))
    )
    return loss * multiplier


def free_port(port=7007):
    import socket

    current_pid = os.getpid()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            try:
                out = subprocess.check_output(f"lsof -t -i:{port}", shell=True).decode().strip()
                if out:
                    for pid_str in out.split():
                        try:
                            pid = int(pid_str)
                            if pid != current_pid:
                                os.kill(pid, signal.SIGKILL)
                        except Exception:
                            pass
                    time.sleep(0.8)
            except Exception:
                pass


def main():
    print("=" * 70)
    print(" 🚀 TrainScope Real-Life End-to-End Showcase ")
    print("=" * 70)

    free_port(7007)

    run_dir = "./trainscope_runs/realife_demo_run"
    config = TrainScopeConfig(
        run_dir=run_dir,
        run_name="realife_demo",
        spike_threshold=10.0,
        detector={"name": "changepoint", "threshold": 10.0, "slack": 1.0},
        stop_on_spike=False,
        full_resolution_window=500,
        histogram_every_n_steps=10,
        activation_metrics_every_n_steps=2,
        track_memory=True,
    )

    run_path = os.path.abspath(os.path.join(config.run_dir, config.run_name))
    os.makedirs(run_path, exist_ok=True)

    print("\n[1/3] Launching TrainScope UI Server FIRST...")
    print(f"      Run Directory: {run_path}\n")

    cmd = [
        sys.executable,
        "-m",
        "trainscope.cli",
        "ui",
        "--run",
        run_path,
        "--port",
        "7007",
    ]

    proc = subprocess.Popen(cmd)
    time.sleep(1.5)

    print("=" * 70)
    print(" 🌟 Open your browser NOW to watch live telemetry & record demo:")
    print(" 👉 http://localhost:7007")
    print("=" * 70)

    torch.manual_seed(42)
    stress_rng = random.Random("realife-stress")
    training_stress = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerLM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    scope = TrainScope(model, optimizer, config=config).attach()
    scope.writer.write_meta(
        "TransformerLM-4L",
        {"vocab": VOCAB, "d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS},
    )

    print(f"\n[2/3] Training 4-Layer Transformer ({N_STEPS} steps)...")

    try:
        for step in range(N_STEPS):
            time.sleep(0.04)  # Small pacing delay for real-time visual streaming
            x = torch.randint(0, VOCAB, (BATCH, SEQ_LEN), device=device)
            targets = torch.randint(0, VOCAB, (BATCH, SEQ_LEN), device=device)

            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB), targets.view(-1))

            # The failure point emerges from cumulative noisy stress rather than
            # from a hardcoded step. The detector only sees the resulting loss.
            training_stress = evolve_training_stress(training_stress, stress_rng)
            loss = apply_training_stress(loss, training_stress)

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

        print("\n[3/3] Training run complete! Artifacts persisted.")
        print("=" * 70)
        print(" 🌟 Server is live at http://localhost:7007")
        print(" Press Ctrl+C anytime to stop the UI server.\n")

        proc.wait()
    except KeyboardInterrupt:
        print("\nStopping UI server...")
        if proc.poll() is None:
            proc.terminate()
            proc.wait()
        print("Done.")


if __name__ == "__main__":
    main()
