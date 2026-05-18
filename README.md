# trainscope
[![PyPI](https://img.shields.io/pypi/v/trainscope)](https://pypi.org/project/trainscope)
[![Python](https://img.shields.io/pypi/pyversions/trainscope)](https://pypi.org/project/trainscope)
[![CI](https://github.com/kaelvalen/trainscope/actions/workflows/ci.yml/badge.svg)](https://github.com/kaelvalen/trainscope/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Post-mortem debugger for LLM training loss spikes.

When a spike hits, you usually know *that* it happened but not *why*. trainscope records per-layer gradients, weight distributions, and activation kurtosis at every step, then lets you scrub back through the event in a browser UI.

## Install

```bash
pip install -e .
```

Dependencies: `torch`, `pyarrow`, `fastapi`, `uvicorn`, `click`, `numpy`.

## Quickstart

```python
from trainscope import TrainScope
from trainscope.core.config import TrainScopeConfig

scope = TrainScope(model, optimizer, config=TrainScopeConfig()).attach()

for step, batch in enumerate(dataloader):
    loss = forward_and_backward(batch)
    optimizer.step()

    spike = scope.step(loss.item(), batch_index=step)
    if spike:
        print(f"Spike at step {spike['step']}, z={spike['z_score']:.2f}")

scope.writer.close()
scope.detach()
```

Then open the UI:

```bash
trainscope ui --run ./trainscope_runs/<run-name>
```

## What gets recorded

**Per step (global)**
- Train loss, global grad norm (pre- and post-clip), learning rate
- Adam second-moment (v) norm — stale momentum indicator
- Step time, batch index

**Per step, per layer**
- Gradient L2 norm
- Weight L2 norm
- Activation mean / std / max-abs / kurtosis — kurtosis is the earliest spike signal
- NaN/Inf ratio in gradients
- 16-bin weight histogram

**On spike**
- Full snapshot of the surrounding window (configurable before/after)
- Per-layer data for the same window
- RNG state at the spike step (for exact replay)

## Overhead

Measured on CPU with a 2-layer GPT-2 (144 parameters). GPU overhead is ~3–8× lower.

| Config | CPU overhead | GPU overhead |
|--------|-------------|-------------|
| Default (`hist/50`, `act/5`) | ~55% | ~4% |
| + `activation_layer_filter=["attn","mlp"]` | ~38% | ~2% |
| Minimal (`hist/50`, `act/50`, filter) | ~18% | ~1% |

CPU measured on 2-layer mini-GPT (144 params), Apple M2. GPU measured on the same model with CUDA. Results will differ on larger models — histogram cost scales with parameter count, activation cost scales with layer count × sequence length.

## UI

Four views, one command:

| View | What it shows |
|------|---------------|
| **Timeline** | Loss + grad norm, top-8 layers by grad variance |
| **Layer Drill-down** | Kurtosis / grad norm / weight norm per layer; histogram scrubber |
| **Diff View** | KL divergence of weight distributions between any two steps |
| **Spike Inspector** | Per-spike window: loss+grad timeline and layer kurtosis/grad breakdown |

The UI works immediately after `pip install` — a built-in fallback HTML with Plotly CDN is served when the React build is absent. For the full React build:

```bash
cd frontend && npm install && npm run build
```

## CLI

```bash
# Open UI for a completed or in-progress run
trainscope ui --run ./trainscope_runs/run_20250516_143022 [--host 127.0.0.1] [--port 7007]

# Generate replay_config.json (does NOT resume training automatically)
trainscope replay --checkpoint ./checkpoints/step_4400.pt --skip-batches 4521,4522,4523 [--resume]
```

To actually skip batches, use `SkippingDataLoader` in your training script:

```python
from trainscope.replay import SkippingDataLoader
import json

with open("replay_config.json") as f:
    cfg = json.load(f)

loader = SkippingDataLoader(original_loader, skip_batches=cfg["skip_batches"])
for batch in loader:
    ...
```

## Configuration

```python
TrainScopeConfig(
    run_dir="./trainscope_runs",                 # output root
    spike_threshold=3.5,                     # z-score threshold (rolling window baseline)
    full_resolution_window=500,              # last N steps at full resolution
    decimation_factor=10,                    # older steps: keep every Nth
    spike_window_before=50,                  # steps before spike to save (≤ full_resolution_window)
    spike_window_after=10,                   # steps after spike to save
    histogram_every_n_steps=50,             # weight histograms are expensive; sample them
    activation_metrics_every_n_steps=5,     # kurtosis sampling; always captured at spike
    activation_layer_filter=["attn", "mlp"],# None = all leaf layers
    stop_on_spike=False,                     # raise StopTraining on detection
    trace_every_n_steps=1,                   # subsample for very large models
    rank=None,                               # DDP rank → adds _rank{N} suffix to run dir
)
```

## Demo

```bash
python examples/gpt2_spike_demo.py
```

Trains a 2-layer mini-GPT, injects a ×50 loss spike at step 50, and shows trainscope detecting it. Run `trainscope ui` on the output directory to explore the event.

## Storage layout

```
trainscope_runs/<run-name>/
    meta.json                          model config + trainscope config
    global.arrow                       step-level scalars (Arrow IPC)
    layers/<param-name>.arrow          per-layer metrics
    spikes/spike_step_<N>.arrow        global window around spike N
    spikes/spike_step_<N>_layers/      per-layer data for that window
    rng_states/step_<N>.pkl            RNG state for replay
```

Estimated storage: ~10 MB/step at full resolution. Rolling 500-step window → ~5 GB max for a 1B-param model. Spike windows are small.

## Publishing

CI runs on every push to `main` and every PR (`pytest` + `ruff`, Python 3.11 + 3.12, Vite build).

To publish a release to PyPI:
1. Set up [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) on PyPI for this repo (environment name: `pypi`).
2. Tag and push: `git tag v0.1.0 && git push origin v0.1.0`

The publish workflow builds the React frontend, bundles it into the wheel, and uploads via OIDC — no API token needed.
