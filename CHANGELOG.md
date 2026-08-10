# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Cluster typical early-warning lead.** `/api/cluster` now reports `typical_lead_steps` per cluster — the median time between the first signal's crossing and the objective explosion step (loss > 10x baseline or non-finite) across the cluster's runs. The Runs view's cluster card shows it as a `~N-step lead` badge, answering "how much warning does this failure mode give" alongside "which runs share it".
- **VISION.md "what we will not do" reviewed.** The rejection list is now evaluated against the shipped product (v1.7.x) rather than the original draft: CV fleet monitoring and the W&B-style dashboard remain rejected (the forensics framing strengthened since 1.1/1.2), and two evidence-based rejections are added — promoting any single signal to "primary alarm" (the v1.6.0 cascade claim failed to reproduce with the correct activation-kurtosis metric, so signals stay independent evidence) and adding a third architecture class without a concrete failure mode it answers.

## [1.7.1] - Clustering Chronology & Signal-Ordering Correction

### Fixed
- **Run clustering now orders signals chronologically, not by code order.** `/api/cluster` previously chose a run's "first" signal by the order the code checked them (grad_norm → loss → kurtosis → concentration), not by actual crossing step — so a run where kurtosis fired at step 30 and grad norm at step 50 could be mislabeled "gradient-led". `_signal_crossing` now returns the crossing step (not a bool) and the first signal is the one with the minimum crossing step, mirroring the experiment's `analyze()`. Added a regression test with two fired signals in one run where chronological order differs from code order.
- **Signal-ordering claim corrected: kurtosis measurement used weights, not activations.** The v1.6.0 `verify_signal_ordering.py` measured kurtosis of the attention projection *weights*; production's `act_kurtosis` (and the original kurtosis experiment) measures *activation* kurtosis. Re-running with the correct activation metric, the "consistent mechanical cascade (kurtosis first)" result does NOT reproduce: the order varies across seeds (3 distinct orders in 3/3). The one stable finding is that loss CUSUM always fires last (lead 7–10). VISION.md updated: no signal is promoted to primary-alarm status; the Spike Inspector implication is withdrawn. Production `/api/cluster` already read the correct `act_kurtosis` source, so it was unaffected by the experiment's metric error.

## [1.7.0] - Run Behavior Clustering

### Added
- **Run behavior clustering (Phase 1 depth).** The Runs view now groups runs by their early-warning signal signature, using the v1.6.0 cascade ordering. `GET /api/cluster` computes, per run, which of the four signals (kurtosis, gradient norm, routing concentration, loss CUSUM/spike) fired under the same robust crossing rule used in the verification experiments, and groups runs with identical fired-signal sets and first signal. Clusters are labeled by cascade position (activation-led, gradient-led, routing-led, loss-led, no-signal); the UI renders each cluster with its runs as clickable buttons that switch the active run.

## [1.6.0] - Signal Ordering

### Added
- `scripts/verify_signal_ordering.py`: empirical check of signal ordering — do the four verified early-warning signals predict each other? All four (loss CUSUM, activation kurtosis, gradient norm, routing concentration) were measured in the same organic LR-ramp run on a hybrid MoE+memory transformer (wikitext-2). Result: the order is **consistent across 3/3 seeds** — kurtosis fires first (lead 29–36 steps), then gradient norm (22–24), then routing concentration (8–22), then loss CUSUM last (7–10). There is a mechanical cascade: activation distribution degrades, gradients grow, routing concentrates, loss CUSUM fires. This confirms the earlier single-run kurtosis-before-CUSUM observation was not a coincidence. VISION.md updated with the ordering and its UI implication (Spike Inspector should present signals in this order).

## [1.5.0] - Architecture-Aware Comparison

### Added
- **Architecture-aware comparison (Phase 1 × Phase 2).** `/api/compare` now carries each run's routing/addressing concentration series (`concentration_series`, max share per step from `moe.arrow`), and the common-cause analysis includes the *runtime signal*: if every spiked run concentrated (peak max-share above the configured detector threshold) and no stable run did, a "max routing concentration" common cause is reported — answering "which runs had expert collapse" alongside "which runs blew up". The Compare panel in the Runs view adds a concentration overlay chart (per-run lines with 0.6/0.85 threshold guides) and renders the concentration common cause with dedicated wording.
- **Addressor-concentration drift detector (Phase 2, memory-augmented).** The v1.4.1-verified signal is now production code: `{"name": "addressor_concentration_drift", ...}` joins the detector family (default threshold 0.6 — the experiment's "control max + margin" rule; fires after `run_steps` consecutive steps above it). The scope records per-slot addressing shares for any module named `addressor` (mean softmax weight per slot over tokens — matching the experiment's signal) into `moe.arrow`, and the Routing & addressing view renders per-slot share series for addressor blocks alongside per-expert series for routers. No schema change: the existing `MOE_SCHEMA` stream already carries per-block share vectors.

## [1.4.1] - Addressor-Collapse Verification

### Added
- `scripts/verify_addressor_collapse_signal.py`: empirical check of the addressor-collapse claim (memory-augmented architecture sibling of the MoE experiment), on a mini memory-augmented transformer with 16 soft-addressed slots trained on wikitext-2. Result is **positive**: mean max-slot addressing share exceeded 0.6 durably 7–11 steps (mean 9.3) before loss divergence in 3/3 LR-ramp seeds, with zero collapses in 3/3 stable-control seeds (max share 0.24–0.32). Methodology notes documented in the script: the 0.6 threshold is "control max + margin" (measured healthy ceiling 0.32; MoE's 0.85 sat above its control's 0.74 — both validated by running the control first), and the dead-slot signal was measured and rejected (a slot below 2% mean weight exists in every step of both conditions, mirroring MoE's dead-expert finding). VISION.md updated with the empirical status.

## [1.4.0] - Architecture-Aware Diagnostics

### Added
- **Architecture-aware diagnostics (Phase 2 body).** For Mixtral-style MoE models (any module named `router`), trainscope now records per-expert routing shares at every step and ships a detector for routing *concentration*:
  - New Arrow stream `moe.arrow` (`MOE_SCHEMA`: `step`, `block`, `shares`) — additive file, old readers ignore it. Written by both `DiskWriter` and `RemoteWriter`; resume, compaction, and manifest support included.
  - New detector `{"name": "expert_utilization_drift", ...}` joining CUSUM/z-score/percentile: consumes the step's max expert share (the scope feeds the routing signal instead of the loss when this detector is active) and reports when share ≥ `threshold` (default 0.85) for `run_steps` consecutive steps — the empirically verified v1.3.0 signal.
  - New "Expert utilization" UI view: per-block per-expert share time series, concentration badges per block, and a warning card citing the 4–12 step empirical lead.
  - `GET /api/moe` returns the active run's routing rows.

## [1.3.0] - Phase 2 Empirical Gate

### Added
- `scripts/verify_expert_collapse_signal.py`: empirical check of Phase 2's expert-collapse claim, on a mini Mixtral-style MoE (4 experts, top-2 routing) trained on wikitext-2. Result is **positive with a caveat**: routing *concentration* (max-expert share > 0.85 sustained) preceded loss divergence by 4–12 steps (mean 7.7) in 3/3 LR-ramp seeds, with zero collapses in 3/3 stable-control seeds — so the signal is specific. But a "dead expert" (share < 2%) also occurs in the stable control, so per-expert abandonment is normal MoE behavior, not a pathology; Phase 2 detectors must key on concentration. VISION.md updated with the empirical status and the corrected example.

## [1.2.0] - Run Comparison

### Added
- **Run comparison (Phase 1 body).** The Runs view now lets you check two or more runs and compare them: overlaid loss curves with an automatic **divergence point** (the first step where the curves durably separate — median warmup gap as baseline, requires 3 consecutive steps above 3x), a **config-diff** table (every `trainscope_config`/`model_config` field that differs across the selected runs), and a **common-cause** summary (numeric/boolean config traits shared by every spiked run but absent from every stable run, e.g. "all spiked runs have `detector.threshold` >= 6.0"). Backed by `GET /api/compare?runs=a,b,c`. Arrow schema unchanged.

## [1.1.0] - Multi-Run Reading

### Added
- **Multi-run reading (Phase 1 foundation).** `trainscope ui --runs <root>` opens a root directory containing many runs instead of a single run. The new Runs view lists every run side by side — model, detector, step count, last loss, spike count — so "which runs exploded overnight" is visible at a glance; selecting a run switches the Timeline/Layer/Diff/Spike views to it. Backed by `GET /api/runs` (summaries from each run's `meta.json`/`manifest.json`/`global.arrow`) and `POST /api/runs/select` (switches the active run; per-run server caches are invalidated). Arrow schema unchanged. No comparison logic yet — that is the next milestone.

## [1.0.0] - Stability Promise

### Added
- `VISION.md`: documents the product direction — the single-run promise today, the three-phase future (multi-run comparison, architecture-aware diagnostics, stability discipline), and the two directions deliberately rejected (inference-time fleet monitoring, general-purpose W&B/MLflow-style dashboards).
- `scripts/verify_kurtosis_early_warning.py`: empirical check of the README's activation-kurtosis claim, run on the same organic mini-GPT-2/wikitext-2 LR-ramp divergence as the CUSUM experiment. The earlier "kurtosis rises 1–5 steps before explosion" estimate was **not** reproduced: kurtosis crossed its robust baseline margin 14–18 steps (mean 16.7) before loss divergence, *ahead of* CUSUM's 9–11 step detection. README updated accordingly.

## [0.9.1] - Stability Audit & Bundle Slimming

### Added
- `scripts/verify_cusum_early_warning.py`: reproducible experiment that closes the long-standing empirical question about CUSUM's early-warning claim. On a real, organic mini-GPT-2/wikitext-2 loss explosion (LR ramp crossing the stability threshold — no scripted spike), the `ChangePointDetector` fired 9–11 steps (mean 9.7) before the loss diverged, across 3/3 seeds. This validates the README claim, which now cites the script instead of stating the early-warning window as design intent.
- `frontend/src/utils/diagnosis.js`: the SpikeInspector's cascade-diagnosis logic (spike_score reading, drift/gradient-explosion/NaN step detection, chronological event ordering) is extracted into a pure, unit-tested module. Previously that logic lived inline in the component and was only verified by eye.
- Documented the 1.0 stability scope in README ("Stability scope" section): the Python API and config surface are the SemVer contract; the Arrow file format is additive-only within a major (new nullable fields = minor, removal/type change = major); the HTTP/WebSocket API is declared an implementation detail of the bundled UI rather than a public contract.

### Changed
- **Breaking (1.0 prep):** `spike_threshold` was removed from `TrainScopeConfig`. Detector thresholds are configured per-detector: `detector={"name": "z_score", "threshold": 3.5}` (the `z_score` detector's default remains 3.5). `load_config` and `TRAINSCOPE_*` env loading now reject the old key with a migration hint instead of silently mis-scaling CUSUM or failing on `percentile`.
- **Breaking (1.0 prep):** `StopTraining.z_score` was renamed to `StopTraining.spike_score`, since the attribute carries the *active detector's* score (CUSUM's cumulative sum, not a z-score, by default). The old name remains as a `@property` emitting `DeprecationWarning`. The `spike_info` dict passed to callbacks/alerters and returned by `step()` now contains `spike_score` (canonical) and keeps `z_score` as a legacy alias with the same value.
- Frontend bundle: views and Plotly are now lazy-loaded (`React.lazy` + dynamic `import()`). The initial shell drops from ~52KB to ~29KB JS (gzip ~58KB total with vendor) and the 4.9MB Plotly bundle is only fetched when the first chart renders instead of on page load.
- Frontend dev dependencies upgraded to close all 8 `npm audit` findings (vite 5→6, vitest 1→3, @vitejs/plugin-react 4.3→4.7). All were build-time/dev tooling — no runtime dependency changed.

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
