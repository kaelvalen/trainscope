# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - Observability UI and Live Run UX

### Added
- Redesigned responsive React UI with a shared navigation, card, chart, metric, and control system.
- Layer metric tabs and searchable layer selection for faster drill-down workflows.
- Quick step presets and step swapping in Diff View.
- Selectable anomaly windows and live data freshness indicators in Timeline.
- Regression coverage for live WebSocket snapshots, deltas, and initially empty runs.

### Changed
- Live run data now reconciles periodically without reloading or resetting the page.
- Transient Arrow reads and stale REST snapshots no longer erase current dashboard data.
- Spike notifications are grouped by anomaly window and capped to prevent toast flooding.
- The real-life demo uses a reproducible noisy stress process instead of fixed drift/spike steps.

## [0.3.0] - CUSUM Detection, WandB Integration, Spike Story & Live Streaming

### Added
- **CUSUM Change-Point Anomaly Detection**: Implemented Page's CUSUM algorithm with robust Median/MAD baseline statistics to detect slow, persistent loss drifts ($0.10\sigma - 0.25\sigma$) 5–10 steps before loss explosions.
- **WandB Zero-Config Integration & Opt-In Alerting**: Automatic passive metric logging when `wandb.run` is active, with explicit opt-in for `wandb.alert()` notifications via `integrations={"wandb": {"alerts": True}}`.
- **Chronological Spike Story Cascade Analysis**: Automated root cause diagnosis banner and pre-spike window timeline tracing (`Loss Shift` → `Gradient Explosion` → `NaN Collapse`).
- **Dynamic Plotly Code-Splitting**: Split heavy Plotly charting engine into an isolated lazy chunk, reducing initial UI shell load size from 5.06 MB to <150 KB.
- **Incremental WebSocket Live Streaming**: High-performance `global_delta` WebSocket streaming appends live steps with zero UI freeze or memory bloat.
- **Empirical Benchmarks & Verification**: Added 100M–7B model overhead benchmark script and 34,000+ step held-out detector robustness test suite.

### Changed
- Default UI server now serves compiled React UI assets directly from package.
- PyPI wheel packaging (`hatch build`) embeds React static assets into PyPI distribution.

## [0.2.2] - Fix CI and type annotations

### Fixed
- Fixed mypy type annotations across core, UI, integrations, plugins, and scope modules.
- Removed hardcoded `python_version = "3.11"` from mypy configuration to align with running environment.
- Corrected code formatting to pass `ruff format --check`.

## [0.2.1] - Refactored core

### Added
- New `TrainScopeConfig` options: `device`, `track_memory`, `checkpoint_on_spike`, `rng_every_n_steps`, and `resume`.
- Layer filenames are now percent-encoded by `DiskWriter` for safe filesystem storage.
- `DiskWriter` writes a `manifest.json` summarizing persisted files and the latest step.
- Server: `/api/health` and `/api/manifest` endpoints, CORS, Pydantic query-parameter validation, bounded LRU/TTL caching, async file I/O, structured logging, and improved error handling.
- CLI: `--version`, `--log-level` for the UI command, replay `@file` support for skip batches, `weights_only=True` checkpoint loading, and richer `replay_config.json` output.
- Replay: robust `SkippingDataLoader` with per-epoch reset, non-negative skip validation, and `load_rng_state` / `replay_step` helpers.
- Integration tests now exercise `/api/health`, `/api/manifest`, `/api/layers/ranked`, `/api/diff`, and spike-window endpoints.

### Changed
- Quickstart semantics: `scope.step()` is now intended to run between `loss.backward()` and `optimizer.step()`.
- Frontend: shared `RunContext`, reusable components, error boundaries, request validation/retries in `api.js`, keyboard shortcuts, and responsive layout.
- Tooling: mypy, ruff format, pytest-asyncio, pre-commit hooks, Makefile targets, and integration tests in CI.

### Fixed
- Arrow reader compatibility with PyArrow versions that do not expose `RecordBatchFileReader.close()`.
- Integration test baseline loss now has non-zero variance so the detected spike reports a finite z-score.
- README parameter count (2-layer mini-GPT is ~430K parameters, not 144).

## [0.1.0] - Initial release

### Added
- Core `TrainScope` recorder with rolling buffers, online spike detection, and Arrow persistence.
- Per-layer gradient, weight, activation, and histogram metrics.
- FastAPI UI server with Timeline, Layer Drill-down, Diff View, and Spike Inspector.
- CLI commands `ui` and `replay`.
- Example GPT-2 spike demo.
