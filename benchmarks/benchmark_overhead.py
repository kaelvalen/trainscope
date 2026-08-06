"""Benchmark script to measure TrainScope overhead on PyTorch transformer models (100M to 7B parameters).

Usage:
    python benchmarks/benchmark_overhead.py [--params 1B] [--steps 20] [--warmup 5]
"""

import argparse
import sys
import time
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from torch.optim import AdamW

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trainscope import TrainScope, TrainScopeConfig


class TransformerLayer(nn.Module):
    """Standard Transformer Block (Self-Attention + MLP)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class BenchModel(nn.Module):
    """Configurable Transformer Model for benchmarking."""

    def __init__(self, d_model: int, n_layers: int, n_heads: int, vocab_size: int = 32000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [TransformerLayer(d_model, n_heads, d_ff=d_model * 4) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits: torch.Tensor = cast(torch.Tensor, self.head(x))
        return logits


MODEL_PRESETS = {
    "100M": {"d_model": 768, "n_layers": 12, "n_heads": 12},
    "350M": {"d_model": 1024, "n_layers": 24, "n_heads": 16},
    "1B": {"d_model": 2048, "n_layers": 24, "n_heads": 16},
    "3B": {"d_model": 3072, "n_layers": 32, "n_heads": 24},
    "7B": {"d_model": 4096, "n_layers": 32, "n_heads": 32},
}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_benchmark(
    preset_name: str,
    steps: int = 20,
    warmup: int = 5,
    seq_len: int = 256,
    batch_size: int = 2,
    device: str = "cpu",
    output_dir: Path | None = None,
):
    preset = MODEL_PRESETS[preset_name]
    print(f"\n--- Benchmarking Preset: {preset_name} on {device.upper()} ---")

    device_obj = torch.device(device)

    # Initialize model and optimizer
    model = BenchModel(
        d_model=preset["d_model"],
        n_layers=preset["n_layers"],
        n_heads=preset["n_heads"],
    ).to(device_obj)

    num_params = count_parameters(model)
    print(f"Model Parameters: {num_params / 1e6:.1f}M ({num_params:,})")

    optimizer = AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Create dummy batch
    input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device_obj)
    targets = torch.randint(0, 32000, (batch_size, seq_len), device=device_obj)

    # Baseline Timing (without TrainScope)
    baseline_times = []
    print("Measuring Baseline performance...")
    for step in range(warmup + steps):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits.view(-1, 32000), targets.view(-1))
        loss.backward()
        optimizer.step()

        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        if step >= warmup:
            baseline_times.append((t1 - t0) * 1000.0)

    mean_baseline_ms = sum(baseline_times) / len(baseline_times)
    print(f"Baseline Mean Step Time: {mean_baseline_ms:.2f} ms")

    # TrainScope Timing
    if output_dir is None:
        output_dir = Path("./benchmark_runs")
    output_dir.mkdir(exist_ok=True)

    config = TrainScopeConfig(
        run_name=f"bench_{preset_name}",
        run_dir=str(output_dir),
        track_memory=True,
    )

    scope = TrainScope(model=model, optimizer=optimizer, config=config)

    trainscope_times = []
    print("Measuring TrainScope performance...")
    for step in range(warmup + steps):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits.view(-1, 32000), targets.view(-1))
        loss.backward()

        scope.step(loss=loss.item())
        optimizer.step()

        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        if step >= warmup:
            trainscope_times.append((t1 - t0) * 1000.0)

    scope.detach()

    mean_scope_ms = sum(trainscope_times) / len(trainscope_times)
    overhead_ms = mean_scope_ms - mean_baseline_ms
    overhead_pct = (overhead_ms / mean_baseline_ms) * 100.0

    print(f"TrainScope Mean Step Time: {mean_scope_ms:.2f} ms")
    print(f"Overhead: {overhead_ms:+.2f} ms ({overhead_pct:+.2f}%)")

    return {
        "preset": preset_name,
        "params_m": num_params / 1e6,
        "baseline_ms": mean_baseline_ms,
        "scope_ms": mean_scope_ms,
        "overhead_ms": overhead_ms,
        "overhead_pct": overhead_pct,
    }


def main():
    parser = argparse.ArgumentParser(description="TrainScope Model Scale Overhead Benchmark")
    parser.add_argument(
        "--preset",
        choices=list(MODEL_PRESETS.keys()) + ["all"],
        default="100M",
        help="Model scale preset to benchmark",
    )
    parser.add_argument("--steps", type=int, default=15, help="Number of benchmark steps")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup steps")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    args = parser.parse_args()

    presets_to_run = list(MODEL_PRESETS.keys()) if args.preset == "all" else [args.preset]
    results = []

    for preset in presets_to_run:
        res = run_benchmark(
            preset_name=preset,
            steps=args.steps,
            warmup=args.warmup,
            device=args.device,
        )
        results.append(res)

    print("\n" + "=" * 60)
    print("FINAL OVERHEAD BENCHMARK SUMMARY")
    print("=" * 60)
    print(
        f"| {'Model Preset':<12} | {'Params (M)':<10} | {'Baseline (ms)':<14} | {'TrainScope (ms)':<15} | {'Overhead (%)':<12} |"
    )
    print("|" + "-" * 14 + "|" + "-" * 12 + "|" + "-" * 16 + "|" + "-" * 17 + "|" + "-" * 14 + "|")
    for r in results:
        print(
            f"| {r['preset']:<12} | {r['params_m']:<10.1f} | {r['baseline_ms']:<14.2f} | {r['scope_ms']:<15.2f} | {r['overhead_pct']:<+11.2f}% |"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
