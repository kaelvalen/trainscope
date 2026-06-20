# trainscope

[![PyPI](https://img.shields.io/pypi/v/trainscope)](https://pypi.org/project/trainscope)
[![Python](https://img.shields.io/pypi/pyversions/trainscope)](https://pypi.org/project/trainscope)
[![CI](https://github.com/kaelvalen/trainscope/actions/workflows/ci.yml/badge.svg)](https://github.com/kaelvalen/trainscope/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Post-mortem debugger for LLM training loss spikes.**

When a loss spike hits, you know *that* it happened — trainscope tells you *why*. It records per-layer gradients, weight distributions, activation statistics, and optimizer state at every training step, then serves a browser UI to scrub back through the event.

```bash
pip install trainscope
```

## Table of contents

- [Why trainscope?](#why-trainscope)
- [Quick start](#quick-start)
- [What gets recorded](#what-gets-recorded)
- [Browser UI](#browser-ui)
- [Command-line interface](#command-line-interface)
- [Configuration](#configuration)
- [Storage layout](#storage-layout)
- [Overhead](#overhead)
- [Development](#development)
- [Publishing](#publishing)
- [License](#license)

## Why trainscope?

Loss spikes in large language model training are expensive. Existing tools log aggregates; trainscope logs the *mechanism*:

- **Activation kurtosis** often rises 1–5 steps before loss explodes.
- **Gradient L2 norms** per layer show exactly where the update became unstable.
- **Weight histograms + KL divergence** let you compare distributions before and after the spike.
- **RNG state + optional checkpoint** at the spike step enable exact replay.

All data is written to local Arrow files; the UI is a standalone FastAPI server that can be started on the training node or your laptop.

## Quick start

```python
from trainscope import TrainScope
from trainscope.core.config import TrainScopeConfig

scope = TrainScope(model, optimizer, config=TrainScopeConfig()).attach()

for step, batch in enumerate(dataloader):
    optimizer.zero_grad()
    loss = forward_and_backward(batch)

    # Record metrics between backward and optimizer step so gradient norms are
    # measured before the optimizer mutates parameters.
    spike = scope.step(loss.item(), batch_index=step)
    optimizer.step()

    if spike:
        print(f"Spike at step {spike['step']}, z={spike['z_score']:.2f}")

scope.detach()
```

Open the run in the browser UI:

```bash
trainscope ui --run ./trainscope_runs/<run-name>
```

For a self-contained example with an injected spike:

```bash
python examples/gpt2_spike_demo.py
trainscope ui --run ./trainscope_runs/<run-name>
```

## What gets recorded

### Per step (global)

- Train loss, global grad norm (pre- and post-clip), learning rate
- Adam second-moment (`v`) norm — stale momentum indicator
- Step time, batch index
- CPU/CUDA memory usage when `track_memory=True`

### Per step, per layer

- Gradient L2 norm, max absolute gradient, gradient mean
- Weight L2 norm, mean, std, min, max absolute value
- Activation mean / std / min / max / median / max-abs / kurtosis
- NaN/Inf ratio in gradients
- 16-bin weight histogram

### On spike

- Full snapshot of the surrounding window (`spike_window_before` + `spike_window_after`)
- Per-layer data for the same window
- RNG state at the spike step for exact replay
- Optional model checkpoint when `checkpoint_on_spike` is enabled

## Browser UI

Four views, one command:

| View | What it shows |
|------|---------------|
| **Timeline** | Loss + grad norm, top-8 layers by gradient variance |
| **Layer Drill-down** | Kurtosis / grad norm / weight norm per layer with histogram scrubber |
| **Diff View** | KL divergence of weight distributions between any two steps |
| **Spike Inspector** | Per-spike window: loss + grad timeline and layer breakdown |

The UI works immediately after `pip install trainscope` — a built-in fallback HTML UI is served when the React build is absent. For the full React build:

```bash
cd frontend && npm install && npm run build
```

## Command-line interface

```bash
# Open UI for a completed or in-progress run
trainscope ui --run ./trainscope_runs/run_20250516_143022 \
    [--host 127.0.0.1] [--port 7007] [--log-level INFO]

# Print version
trainscope --version

# Generate replay_config.json for exact batch skipping
trainscope replay --checkpoint ./checkpoints/step_4400.pt \
    --skip-batches 4521,4522,4523 [--resume]

# Read skip batches from a file (one index or comma-separated list per line)
trainscope replay --checkpoint ./checkpoints/step_4400.pt \
    --skip-batches @batches.txt
```

Use the generated config with `SkippingDataLoader` in your training script:

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
    run_dir="./trainscope_runs",            # output root
    run_name=None,                          # defaults to run_YYYYMMDD_HHMMSS
    spike_threshold=3.5,                    # z-score threshold (rolling baseline)
    full_resolution_window=500,             # last N steps at full resolution
    decimation_factor=10,                   # older steps: keep every Nth
    spike_window_before=50,                 # steps before spike to save
    spike_window_after=10,                  # steps after spike to save
    histogram_every_n_steps=50,             # weight histograms are expensive
    activation_metrics_every_n_steps=5,     # kurtosis sampling
    activation_layer_filter=["attn", "mlp"],# None = all leaf modules
    stop_on_spike=False,                    # raise StopTraining on detection
    trace_every_n_steps=1,                  # subsample for very large models
    rank=None,                              # DDP rank → _rank{N} suffix
    device=None,                            # metric compute device; None = CPU
    track_memory=True,                      # CPU/CUDA memory in global snapshot
    checkpoint_on_spike=None,               # True, path template, or None/False
    rng_every_n_steps=0,                    # save RNG every N steps (0 = only spikes)
    resume=False,                           # append to existing Arrow files
)
```

### Notable options

- **`device`** — `None` computes metrics on CPU to avoid GPU synchronization; set to `"cuda"` to force GPU.
- **`checkpoint_on_spike`** — Save `model.state_dict()` (and optimizer state if available) on spike. `True` writes `checkpoints/{step}.pt`; a string is a `{step}` path template.
- **`rng_every_n_steps`** — Save RNG state periodically in addition to spike steps.
- **`resume`** — Append to existing Arrow files instead of overwriting.

## Storage layout

```
trainscope_runs/<run-name>/
    meta.json                          model config + trainscope config
    manifest.json                      summary of files and latest step
    global.arrow                       step-level scalars (Arrow IPC)
    layers/<param-name>.arrow          per-layer metrics (percent-encoded filenames)
    spikes/spike_step_<N>.arrow        global window around spike N
    spikes/spike_step_<N>_layers/      per-layer data for that window
    rng_states/step_<N>.pkl            RNG state for replay
    checkpoints/<N>.pt                 model checkpoint on spike (optional)
```

Estimated storage: ~10 MB/step at full resolution for a 1B-parameter model. The default 500-step rolling window caps typical retention at ~5 GB. Spike windows are small.

## Overhead

Measured on CPU with a 2-layer GPT-2 (~430K parameters). GPU overhead is ~3–8× lower.

| Config | CPU overhead | GPU overhead |
|--------|-------------|-------------|
| Default (`hist/50`, `act/5`) | ~55% | ~4% |
| + `activation_layer_filter=["attn","mlp"]` | ~38% | ~2% |
| Minimal (`hist/50`, `act/50`, filter) | ~18% | ~1% |

CPU measured on 2-layer mini-GPT (~430K params), Apple M2. GPU measured on the same model with CUDA. Results scale with parameter count and layer count.

## Development

The project uses a Nix flake for the development shell:

```bash
nix develop
```

Or with a local virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Quick commands:

```bash
make test      # pytest tests/ -q
make lint      # ruff + mypy
make format    # ruff format
make frontend-build
```

Install pre-commit hooks:

```bash
pre-commit install
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for coding style and pull request guidelines.

## License

[MIT](LICENSE)
