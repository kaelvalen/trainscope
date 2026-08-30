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
- [Stability scope](#stability-scope)
- [Overhead](#overhead)
- [Development](#development)
- [Publishing](#publishing)
- [License](#license)

## Why trainscope?

Loss spikes in large language model training are expensive. Existing tools log aggregates; trainscope logs the *mechanism*:

- **CUSUM Change-Point Detection**: Catches subtle, persistent loss drifts ($0.10\sigma - 0.25\sigma$ per step). Verified on real training: in an organic mini-GPT-2/wikitext-2 loss explosion, CUSUM fired 9–11 steps (mean 9.7) before the loss diverged (see `scripts/verify_cusum_early_warning.py`).
- **Expert-utilization drift detection (MoE)**: For Mixtral-style models (any module named `router`), trainscope records per-expert routing shares and can detect routing *concentration* — one expert dominating token routing — 4–12 steps before loss divergence (see `scripts/verify_expert_collapse_signal.py` and the `expert_utilization_drift` detector). The Routing & addressing view plots per-expert shares over time.
- **Addressor-concentration drift (memory-augmented)**: For models with an `addressor` module (softmax addressing over a memory bank), the `addressor_concentration_drift` detector flags slot-share concentration — the addressor locking onto one slot — 7–11 steps before loss divergence (see `scripts/verify_addressor_collapse_signal.py`). Same view renders per-slot shares.
- **Activation kurtosis**: Excess kurtosis of per-block activations rises before divergence. Verified on the same organic mini-GPT-2/wikitext-2 scenario as the CUSUM claim: kurtosis crossed its robust baseline margin 14–18 steps (mean 16.7) before loss divergence — *earlier* than CUSUM's 9–11 step detection (see `scripts/verify_kurtosis_early_warning.py`). Note this supersedes the earlier "1–5 steps" estimate, which was not reproduced; kurtosis leads by more than CUSUM, not less. *(Crossing thresholds were calibrated to true 3σ = 1.4826·MAD in 1.8.1; the 14–18-step lead was measured with the earlier raw-MAD threshold and is pending re-measurement.)*
- **Chronological Spike Story Cascade**: Traces failure cascades chronologically (`Loss Shift` → `Gradient Explosion` → `NaN Collapse`) to isolate root causes instead of terminal symptoms.
- **Gradient L2 norms**: Per-layer breakdowns show exactly which transformer block initiated the update instability.
- **Run behavior clustering**: Runs are grouped by their early-warning signal signature; each cluster reports its typical early-warning lead, the config traits its members share that stable runs do not, the common-fate loss band, and a nearest-stable-run counterexample — "why did THESE runs blow up and not the others".
- **Post-mortem reports**: `trainscope report` turns a run (or a whole runs root) into a markdown/JSON case file with the spike story, fired signals, and cluster summary.
- **Replay planning**: `trainscope replay` writes a `replay_config.json`; the Replay view shows exactly which training steps its skip list maps to.
- **Remote storage**: Run trees on `s3://`/`gs://` are readable by the UI and CLI via fsspec materialization.
- **WandB Zero-Config Integration**: Auto-detects active `wandb.run` sessions for passive logging, with opt-in alerting (`integrations={"wandb": {"alerts": True}}`).
- **Weight histograms + KL divergence**: Compare parameter distributions before and after the spike.
- **RNG state + optional checkpoint**: At the spike step for exact replay.

All data is written to Arrow files (locally or to a remote store via fsspec); the UI is a lightweight standalone FastAPI server with lazy-loaded views and Plotly (initial shell ~60KB gzipped; the 4.9MB Plotly bundle is fetched only when the first chart renders) and incremental WebSocket live streaming for loss, spikes, layers, and MoE routing shares.

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
        print(f"Spike at step {spike['step']}, score={spike['spike_score']:.2f}")

scope.detach()
```

Open the run in the browser UI:

```bash
trainscope ui --run ./trainscope_runs/<run-name>
```

For a self-contained example with an injected drift and spike:

```bash
python examples/gpt2_spike_demo.py
trainscope ui --run ./trainscope_runs/<run-name>
```

## What gets recorded

### Per step (global)

- Train loss, global grad norm, learning rate
  (`grad_norm_after_clip` currently mirrors `grad_norm_before_clip`: TrainScope
  no longer clips gradients itself — clip externally with
  `torch.nn.utils.clip_grad_norm_()` before calling `step()` — so there is no
  separate post-clip reading to record)
- Anomaly score (`spike_score`) from the configured detector — CUSUM
  change-point by default, or Z-score/percentile if configured via
  `detector=`. Only the active detector's score is recorded per step.
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
- Chronological Failure Cascade diagnosis (`Loss Shift` → `Grad Explosion` → `NaN`)
- RNG state at the spike step for exact replay
- Optional model checkpoint when `checkpoint_on_spike` is enabled

## Browser UI

Seven views, one command:

| View | What it shows |
|------|---------------|
| **Runs** | Every run under a root side by side — model, detector, last loss, spike count — plus run behavior clusters, cluster discriminant config traits, the common-fate loss band, and nearest-stable-run counterexample analysis |
| **Timeline** | Loss + grad norm, top-8 layers by gradient variance, live WebSocket streaming |
| **Layer Drill-down** | Kurtosis / grad norm / weight norm per layer with histogram scrubber |
| **Routing & addressing** | Per-expert and per-slot share series over time, live-streamed over WebSocket |
| **Diff View** | KL divergence of weight distributions between any two steps, with per-layer gradient-norm change |
| **Spike Inspector** | **Spike Story Flow**: Chronological root cause cascade diagnosis & layer breakdown |
| **Replay** | The run's generated `replay_config.json` with the training steps its skip list maps to |

The React UI is served by default after `pip install trainscope` (pre-compiled assets included). If developing from source:

```bash
cd frontend && npm install && npm run build
```

## Command-line interface

```bash
# Open UI for a completed or in-progress run
trainscope ui --run ./trainscope_runs/run_20250516_143022 \
    [--host 127.0.0.1] [--port 7007] [--log-level INFO]

# Open UI for every run under a root directory (multi-run mode):
# the Runs view lists all runs side by side with last loss and spike
# count; selecting one switches every other view to it. Check two or
# more runs to compare loss curves (with an automatic divergence
# point), config differences, and shared causes among spiked runs.
trainscope ui --runs ./trainscope_runs

# Local paths and fsspec URIs both work; remote trees (s3://, gs://) are
# materialized to a local cache before the UI starts.
trainscope ui --runs s3://bucket/trainscope_runs

# Print version
trainscope --version

# Generate replay_config.json for exact batch skipping
trainscope replay --checkpoint ./checkpoints/step_4400.pt \
    --skip-batches 4521,4522,4523 [--resume]

# Read skip batches from a file (one index or comma-separated list per line)
trainscope replay --checkpoint ./checkpoints/step_4400.pt \
    --skip-batches @batches.txt

# Generate a post-mortem report for one run (spike story, fired signals, lead)
trainscope report --run ./trainscope_runs/run_20250516_143022 [--format markdown|json]

# Report for every run under a root: cluster by signal signature
trainscope report --runs ./trainscope_runs
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
    storage_uri=None,                       # s3:///gs:// URI for remote storage
)
```

### Notable options

- **`device`** — `None` computes metrics on CPU to avoid GPU synchronization; set to `"cuda"` to force GPU.
- **`detector`** — Selects the anomaly detector: `detector="changepoint"` (default, CUSUM) or `detector={"name": "z_score", "threshold": 3.5}` for the rolling z-score. Detector thresholds live inside this dict — there is no top-level `spike_threshold` since 1.0, because each detector's threshold is on a different scale (CUSUM's cumulative-sum decision threshold vs. a raw z-score cutoff). Architecture-aware detectors: `expert_utilization_drift` (default threshold 0.85, routing concentration in MoE models) and `addressor_concentration_drift` (default threshold 0.6, slot concentration in memory-augmented models).
- **`checkpoint_on_spike`** — Save `model.state_dict()` (and optimizer state if available) on spike. `True` writes `checkpoints/{step}.pt`; a string is a `{step}` path template.
- **`rng_every_n_steps`** — Save RNG state periodically in addition to spike steps.
- **`resume`** — Append to existing Arrow files instead of overwriting.
- **`storage_uri`** — Write to a remote store (`s3://`, `gs://`, `az://`, `file://`) via fsspec instead of the local `DiskWriter`. Remote objects are rewritten on the compaction cadence (no native append), so remote artifacts lag by up to `compaction_every_n_steps` steps.

## Storage layout

```
trainscope_runs/<run-name>/
    meta.json                          model config + trainscope config
    manifest.json                      summary of files and latest step
    global.arrow                       step-level scalars (Arrow IPC)
    layers/<param-name>.arrow          per-layer metrics (percent-encoded filenames)
    moe.arrow                          per-block routing/addressing shares (MoE & addressor)
    plugin_metrics.arrow               plugin metric rows (step, plugin, metric, value)
    spikes/spike_step_<N>.arrow        global window around spike N
    spikes/spike_step_<N>_layers/      per-layer data for that window
    rng_states/step_<N>.pkl            RNG state for replay
    checkpoints/<N>.pt                 model checkpoint on spike (optional)
```

Estimated storage: ~10 MB/step at full resolution for a 1B-parameter model. The default 500-step rolling window caps typical retention at ~5 GB. Spike windows are small.

## Stability scope

Starting with 1.0.0, trainscope follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) with a *defined surface*:

- **Python API (stable contract)** — `TrainScope`, `TrainScopeConfig`, `load_config`, `StopTraining`, and the `trainscope.*` import paths. Breaking changes to these (renames, removed parameters, changed semantics) only land in major releases. `StopTraining.spike_score` is the canonical attribute; `z_score` remains as a deprecated alias.
- **Config surface (stable contract)** — All `TrainScopeConfig` fields, their defaults, and the `TRAINSCOPE_*` / YAML / JSON loading conventions. Detector thresholds are configured per-detector (e.g. `detector={"name": "z_score", "threshold": 3.5}`); there is no top-level `spike_threshold` since 1.0.
- **Arrow file format (additive only within a major version)** — `global.arrow`, layer and spike-window files, and `meta.json`/`manifest.json`. Adding a new nullable field is a minor release; removing a field or changing an existing field's type or semantics requires a major release. Writers may add columns; readers must tolerate columns they do not know about. Plugins already get their own table (`PLUGIN_METRICS_SCHEMA`), so new metric surfaces should extend that rather than reshuffle the core schema.
- **HTTP/WebSocket API (not a public contract)** — The `/api/*` endpoints and `/ws` WebSocket are implementation details of the bundled UI. They are versioned implicitly by the trainscope release and may change shape in minor releases; do not build external clients against them. The [browser UI](#browser-ui) is the only supported consumer.
- **Plugins** — Detector plugins must subclass `AnomalyDetector` and implement `update(loss) -> float | None` plus `warmup`; metric plugins subclass `MetricPlugin` with a `name` ClassVar and `compute(model, optimizer, step) -> dict[str, float]`, writing rows to the plugin-metrics table (`PLUGIN_METRICS_SCHEMA`: `step`, `plugin`, `metric`, `value`). This surface is frozen (contract-tested in `tests/test_plugin_contract.py`); adding a plugin hook is a minor release, changing the existing methods or table columns requires a major release.

Anything not listed here (integration helper details, CLI output formatting) is considered internal.

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
