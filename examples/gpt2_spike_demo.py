"""
Mini GPT-2 spike demo.

Trains a 2-layer GPT-2-style model on random token sequences, artificially
injects a loss spike at step SPIKE_STEP, and shows losswatch detecting it.

Usage:
    python examples/gpt2_spike_demo.py

Then open the UI:
    losswatch ui --run ./losswatch_runs/<run-name>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from losswatch import LossWatch
from losswatch.core.config import LossWatchConfig

VOCAB = 256
SEQ_LEN = 32
BATCH = 8
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
N_STEPS = 150
SPIKE_STEP = 50
SPIKE_MULTIPLIER = 50.0
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


class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, D_MODEL)
        self.pos_emb = nn.Embedding(SEQ_LEN, D_MODEL)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.blocks(h)
        h = self.ln_f(h)
        return self.head(h)


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MiniGPT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Overhead profiles (measured on CPU, 2-layer GPT-2, 144 params):
    #   default  (hist/50, act/5):             ~121% overhead
    #   recommended (+ activation_layer_filter): ~89%
    #   minimal  (hist/50, act/50, filter):      ~52%
    # GPU overhead is significantly lower (~3-8%) due to parallelism.
    config = LossWatchConfig(
        run_dir="./losswatch_runs",
        spike_threshold=3.5,
        stop_on_spike=False,
        full_resolution_window=500,
        histogram_every_n_steps=50,
        activation_metrics_every_n_steps=5,
        # Uncomment to enable recommended profile:
        # activation_layer_filter=["attn", "mlp"],
    )

    watch = LossWatch(model, optimizer, config=config).attach()
    watch.writer.write_meta(
        "MiniGPT",
        {"vocab": VOCAB, "d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS},
    )

    print(f"Device: {device}")
    print(f"Run:    ./losswatch_runs/{config.run_name}")
    print(f"Steps:  {N_STEPS}  (spike injected at step {SPIKE_STEP})\n")

    for step in range(N_STEPS):
        x = torch.randint(0, VOCAB, (BATCH, SEQ_LEN), device=device)
        targets = torch.randint(0, VOCAB, (BATCH, SEQ_LEN), device=device)

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB), targets.view(-1))

        if step == SPIKE_STEP:
            loss = loss * SPIKE_MULTIPLIER
            print(f"  [inject] step={step}  loss={loss.item():.4f}  (×{SPIKE_MULTIPLIER})")

        loss.backward()
        optimizer.step()

        spike = watch.step(loss.item(), batch_index=step)

        if spike:
            print(
                f"\n  *** SPIKE DETECTED ***  step={spike['step']}  "
                f"loss={spike['loss']:.4f}  z={spike['z_score']:.2f}\n"
            )
        elif step % 20 == 0:
            print(f"  step={step:3d}  loss={loss.item():.4f}")

    watch.writer.flush()
    watch.writer.close()
    watch.detach()

    print(f"\nDone. Open the UI with:")
    print(f"  losswatch ui --run ./losswatch_runs/{config.run_name}")


if __name__ == "__main__":
    main()
