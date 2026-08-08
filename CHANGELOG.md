# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] - Signal-Analysis UI Redesign

### Changed
- Redesigned the UI as a signal-analysis instrument instead of a generic dark SaaS dashboard: cold near-black palette (neutral-cold `220 20% 4%` background), teal-green oscilloscope accent (`168 68% 58%`) replacing the pastel cyan, a more saturated alarm red, and border radii scaled from `0.75rem` to `0.25rem` (pill shapes untouched). UI text is now Inter with tabular numerals; all chart text and hover values render in JetBrains Mono. The app background swaps the soft radial glow for a faint accent-colored grid texture.
- Chart interaction is now consistent across all four views: the Plotly modebar is enabled everywhere with scroll-zoom and drag-to-zoom (previously Timeline-only), and Plotly's noisy lasso/select/spikelines/hover-toggle buttons are stripped from the toolbar.
- `RollingBuffer` getters keep their O(n) segment merge but restore a cheap safety net: an O(n) order check falls back to an explicit sort if the two segments are ever out of chronological order, so a future insertion path that violates the segment-ordering invariant cannot silently produce out-of-order rows (the defensive sort removed in 0.7.0 would otherwise have been lost).

## [0.7.1] - General Bugfix Release

### Changed
- `RemoteWriter` (S3/GCS/Azure) now rewrites its objects on the same `compaction_every_n_steps` cadence instead of every 100 rows, roughly a 10x reduction in full-rewrite frequency on large runs. This mitigates but does not eliminate the O(n²) rewrite cost — object stores offer no native append, so the `DiskWriter`'s append-only write path is not achievable remotely. It also reads both on-disk formats (legacy IPC files and the stream format) and writes stream-format objects for consistency.

### Fixed
- The MLflow integration logged `top_grad_layer` and `spike_step` with `log_param`, which MLflow rejects on the second call with a different value (`Changing param values is not allowed`); the error was swallowed by `except Exception`, so MLflow users silently lost all logging after the first step/spike. The layer name is now logged as a text artifact (`top_grad_layer.txt`) and the spike step as a metric.
- `optimizer_v_norm` matched the optimizer's exact class name (`Adam`/`AdamW`), so fused/8-bit/wrapped Adam variants (bitsandbytes, apex, deepspeed) and user subclasses silently reported `0.0`. The metric now duck-types on the Adam-family state contract (`exp_avg_sq`).
- `RollingBuffer.get_global_steps`/`get_layer_steps`/`get_window` sorted and copied the full history on every call; they now linearly merge the two already-sorted, disjoint segments (O(n) instead of O(n log n)).

## [0.7.0] - Write-Path Performance

### Changed
- `DiskWriter` now writes global/layer/plugin-metrics streams in the Arrow IPC *stream* format with true appends: each flush persists only the new rows, and the full file is rewritten (compacted) only every `compaction_every_n_steps` steps (default 1000) instead of on every flush. This removes the O(n²) total write cost on long runs — e.g. a 100k-step run flushes in ~11s instead of tens of minutes — while the file stays readable by the UI server at every flush. Runs written by older versions (legacy IPC file format) remain readable, and resuming one rewrites it into the appendable format on the next flush.

### Added
- New `compaction_every_n_steps` config field (default 1000) controlling how often the Arrow streams are fully rewritten (local) or re-uploaded (remote).

### Fixed
- Prometheus telemetry labelled requests with the concrete request path, so path-parameterized endpoints like `/api/layers/{layer_name}` created one time series per unique layer name (unbounded cardinality). The `path` label now uses the route template, e.g. `/api/layers/{layer_name}`.

## [0.6.0] - PELT Score Semantics

### Changed
- PELT-triggered spikes no longer carry a clamped `|score| == threshold`: `ChangePointDetector` now returns the raw median/MAD-normalized deviation `(loss - median) / (1.4826 * MAD)` on the PELT path, preserving the true magnitude of the change point. CUSUM-triggered spikes (the default path) are unchanged and still satisfy `|score| >= threshold`. This is a behavior change for runs with the optional `ruptures` extra installed: subtle PELT-detected change points can now report scores below the threshold instead of a flat threshold value.

### Added
- `TrainScope.step()` accepts an optional `grad_norm_after_clip=` parameter. Callers that clip gradients externally (TrainScope itself no longer clips) can record the real post-clip norm instead of a mirror of `grad_norm_before_clip`.

## [0.5.1] - Display & Verification Fixes

### Fixed
- Unmeasured activation metrics (`act_mean`, `act_std`, `act_kurtosis`, etc.) were persisted as `0.0` placeholders on steps between `activation_metrics_every_n_steps` samples, so the UI could not distinguish "measured and zero" from "not measured" — producing misleading sawtooth charts. These fields now persist as `null`, and the UI renders them as gaps instead of a flat zero line.
- The detector's warmup window (`min_observations`, default 30) was invisible in the UI: during warmup the detector produces no scores, but the timeline showed an ordinary "no spikes" chart. The detector name and `min_observations` are now recorded in `meta.json` and the Timeline shades the warmup region with a "detector warming up — spikes not yet reported" band.

## [0.5.0] - Release Pipeline Hardening & Python 3.14 Support

### Added
- Python 3.14 support: added to the CI test matrix and package classifiers, verified against the full dependency set (`torch`, `pyarrow`, `numpy`, `scipy`) and test suite.
- `publish.yml` now verifies the pushed tag matches `pyproject.toml`'s version before building, and runs `twine check` on the built artifacts before upload.

### Changed
- `publish.yml` now runs the full CI test suite (via a reusable `ci.yml` workflow) as a required gate before publishing. Tag pushes don't trigger `ci.yml`'s own `push` trigger, so releases were previously built and uploaded to PyPI with no test or lint run at all.
- `ci.yml`'s frontend job now runs the Vitest unit test suite (`npm test`), which was previously never executed in CI.
- Removed a duplicate `pytest tests/test_integration.py` step in `ci.yml`; `make test` already runs the full `tests/` directory, including integration tests.

### Fixed
- CI had been failing on every push to `main` since the previous release: `trainscope/io/writer.py` wasn't `ruff format`-clean, so `make lint` (and thus CI) failed before tests ever ran. Reformatted the file; no behavior change.
- `mypy` failed with "Library stubs not installed for yaml" under mypy >=1.19 despite `ignore_missing_imports = true`; added `types-PyYAML` to the `dev` extra.
- `ci.yml`'s top-level `concurrency.group` used `${{ github.workflow }}-${{ github.ref }}`. When `ci.yml` runs as a reusable workflow called from `publish.yml`, `github.workflow` resolves to the *caller's* name ("Publish to PyPI"), so the group collided with `publish.yml`'s own concurrency group and GitHub cancelled the run before any job started. Switched to `github.workflow_ref`, which is unique per workflow file.
- **Security**: `AuthMiddleware` was defined but never registered with the FastAPI app, so `TRAINSCOPE_API_KEY` / `TRAINSCOPE_BASIC_AUTH` had no effect on any HTTP or WebSocket request. Auth is now wired up and independently enforced on `/ws` (Starlette's `BaseHTTPMiddleware` does not wrap WebSocket connections).
- **Security**: CORS combined `allow_origins=["*"]` with `allow_credentials=True`, a spec-invalid combination that could allow credentialed cross-origin requests. `allow_credentials` is now `False` (auth here is header-based, not cookie-based).
- Alerting (Slack/email) was completely non-functional: `TrainScope.step()` called `alerter.notify(...)`, a method no alerter implements (all define `.alert(...)`). The `AttributeError` was silently swallowed by a broad `except Exception` and logged only, so no alert was ever delivered. Alert tests only ever called `.alert()` directly, never through `step()`, so the mismatch went uncaught.
- `ChangePointDetector`'s optional `ruptures`/PELT code path did not reset the CUSUM accumulators on a detected change point, letting stale state leak into subsequent calls, and returned a raw z-score that was 2-3 orders of magnitude smaller than the CUSUM `threshold` convention — so spike sensitivity silently differed depending on whether `ruptures` was installed. The PELT path now resets CUSUM state and returns a score on the same scale as `threshold`.
- `TrainScope.step()` used `math.isnan()` to decide whether to skip feeding a loss value to the detector, missing `+inf`/`-inf`, which are equally capable of poisoning a detector's running baseline. Now uses `math.isfinite()`.
- `TrainScopeConfig()`'s default detector was `z_score`, not the CUSUM change-point detector documented as the flagship feature (and used throughout the quick-start example). The default is now `changepoint`; `make_detector()` no longer blindly injects the z_score-scaled `spike_threshold` into other detectors (which would raise `TypeError` for `percentile` and silently mis-scale `changepoint`).
- The Spike Inspector's failure-cascade diagnosis read a `row.z_score` field that was never written to the Arrow files, so it always fell back to an ad-hoc heuristic disconnected from the configured detector. The detector's real per-step anomaly score is now persisted as `spike_score` in `global.arrow` and read by the UI.
- The Spike Inspector's drift scan used `row.spike_score || fallback`, so the legitimate value `0.0` (written on every non-spike step) fell through to the old heuristic just like a missing field would, effectively bypassing the detector's real score for most of the scan. Changed to `??` so only a genuinely absent field (older runs) falls back.

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
