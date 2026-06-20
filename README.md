# trainscope
[![PyPI](https://img.shields.io/pypi/v/trainscope)](https://pypi.org/project/trainscope)
[![Python](https://img.shields.io/pypi/pyversions/trainscope)](https://pypi.org/project/trainscope)
[![CI](https://github.com/kaelvalen/trainscope/actions/workflows/ci.yml/badge.svg)](https://github.com/kaelvalen/trainscope/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Post-mortem debugger for LLM training loss spikes.

When a spike hits, you usually know *that* it happened but not *why*. trainscope records per-layer gradients, weight distributions, and activation kurtosis at every step, then lets you scrub back through the event in a browser UI.

## Install

```bash
pip install -e ".[dev]"
```

Dependencies: `torch`, `pyarrow`, `fastapi`, `uvicorn`, `click`, `numpy`.

## Quickstart

```python
from trainscope import TrainScope
from trainscope.core.config import TrainScopeConfig

scope = TrainScope(model, optimizer, config=TrainScopeConfig()).attach()

for step, batch in enumerate(dataloader):
    optimizer.zero_grad()
    loss = forward_and_backward(batch)
    spike = scope.step(loss.item(), batch_index=step)  # before optimizer.step()
    optimizer.step()

    if spike:
        print(f"Spike at step {spike['step']}, z={spike['z_score']:.2f}")

scope.writer.flush()
scope.writer.close()
scope.detach()
```

`scope.step()` should be called **between `loss.backward()` and `optimizer.step()`** so gradient norms are recorded before the optimizer mutates parameters. Calling it after `optimizer.step()` is supported for backward compatibility but less accurate.

Then open the UI:

```bash
trainscope ui --run ./trainscope_runs/<run-name>
```

## What gets recorded

**Per step (global)**
- Train loss, global grad norm (pre- and post-clip), learning rate
- Adam second-moment (v) norm — stale momentum indicator
- Step time, batch index
- CPU/CUDA memory usage when `track_memory=True`

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
- Optional model checkpoint when `checkpoint_on_spike` is enabled

## Overhead

Measured on CPU with a 2-layer GPT-2 (~430K parameters). GPU overhead is ~3–8× lower.

| Config | CPU overhead | GPU overhead |
|--------|-------------|-------------|
| Default (`hist/50`, `act/5`) | ~55% | ~4% |
| + `activation_layer_filter=["attn","mlp"]` | ~38% | ~2% |
| Minimal (`hist/50`, `act/50`, filter) | ~18% | ~1% |

CPU measured on 2-layer mini-GPT (~430K params), Apple M2. GPU measured on the same model with CUDA. Results will differ on larger models — histogram cost scales with parameter count, activation cost scales with layer count × sequence length.

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
trainscope ui --run ./trainscope_runs/run_20250516_143022 [--host 127.0.0.1] [--port 7007] [--log-level INFO]

# Print version
trainscope --version

# Generate replay_config.json (does NOT resume training automatically)
trainscope replay --checkpoint ./checkpoints/step_4400.pt --skip-batches 4521,4522,4523 [--resume]

# Read skip batches from a file (one index or comma-separated list per line)
trainscope replay --checkpoint ./checkpoints/step_4400.pt --skip-batches @batches.txt
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
    run_dir="./trainscope_runs",            # output root
    run_name=None,                          # defaults to run_YYYYMMDD_HHMMSS
    spike_threshold=3.5,                    # z-score threshold (rolling window baseline)
    full_resolution_window=500,             # last N steps at full resolution
    decimation_factor=10,                   # older steps: keep every Nth
    spike_window_before=50,                 # steps before spike to save (≤ full_resolution_window)
    spike_window_after=10,                  # steps after spike to save
    histogram_every_n_steps=50,             # weight histograms are expensive; sample them
    activation_metrics_every_n_steps=5,     # kurtosis sampling; always captured at spike
    activation_layer_filter=["attn", "mlp"],# None = all leaf modules
    stop_on_spike=False,                    # raise StopTraining on detection
    trace_every_n_steps=1,                  # subsample for very large models
    rank=None,                              # DDP rank → adds _rank{N} suffix to run dir
    device=None,                            # metric compute device; None = CPU
    track_memory=True,                      # record CPU/CUDA memory in global snapshot
    checkpoint_on_spike=None,               # True, a path template, or None/False
    rng_every_n_steps=0,                    # save RNG state every N steps (0 = only on spikes)
    resume=False,                           # append to existing Arrow files instead of overwriting
)
```

### New config options

- **`device`** — Device used for metric computation. `None` computes metrics on CPU to avoid GPU synchronization; set to `"cuda"` or `torch.device(...)` to force a specific device.
- **`track_memory`** — When `True`, the global snapshot includes `cpu_memory_mb` and `cuda_memory_mb`.
- **`checkpoint_on_spike`** — Save `model.state_dict()` (and optimizer state if available) when a spike is detected. `True` writes to `checkpoints/{step}.pt`; a string is treated as a path template with a single `{step}` placeholder.
- **`rng_every_n_steps`** — Save RNG state every N steps in addition to on spikes. `0` (default) only saves RNG state on spike steps.
- **`resume`** — If `True` and the run directory already contains Arrow files, new rows are appended; otherwise existing files are overwritten.

## HTTP API

The UI is backed by a FastAPI server. Endpoints:

- `GET /api/meta` — run metadata (model + config)
- `GET /api/manifest` — manifest of persisted files and latest step
- `GET /api/global` — list of global row dicts
- `GET /api/layers` — list of layer name strings
- `GET /api/layers/{layer_name}` — layer row dicts
- `GET /api/layers/ranked?top_n=` — top layers by gradient-variance
- `GET /api/spikes` — list of `{step, file}` spike records
- `GET /api/spikes/{step}` — global rows for the spike window
- `GET /api/spikes/{step}/layers` — layer names for the spike window
- `GET /api/spikes/{step}/layers/{layer_name}` — layer rows for the spike window
- `GET /api/diff?step_a=&step_b=` — KL divergence of weight histograms between two steps
- `GET /api/health` — health check
- `/` — built React UI (or fallback HTML if the build is missing)

## Demo

```bash
python examples/gpt2_spike_demo.py
```

Trains a 2-layer mini-GPT, injects a ×50 loss spike at step 50, and shows trainscope detecting it. Run `trainscope ui` on the output directory to explore the event.

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

Estimated storage: ~10 MB/step at full resolution. Rolling 500-step window → ~5 GB max for a 1B-param model. Spike windows are small.

## Development & contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, coding style, and how to open a pull request.

Quick developer commands:

```bash
make install   # pip install -e ".[dev]"
make test      # pytest tests/ -q
make lint      # ruff check + format check + mypy
make format    # ruff format
make frontend-build
```

Install the pre-commit hooks to run linting checks automatically on every commit:

```bash
pre-commit install
```

## Publishing

CI runs on every push to `main` and every PR (`pytest` + `ruff` + `mypy`, Python 3.11/3.12/3.13, frontend lint/build).

To publish a release to PyPI:
1. Set up [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) on PyPI for this repo (environment name: `pypi`).
2. Tag and push: `git tag v0.2.1 && git push origin v0.2.1`

The publish workflow builds the React frontend, bundles it into the wheel, and uploads via OIDC — no API token needed.
