# trainscope Production Roadmap

This document tracks the work required to move trainscope from a local debugging tool to a production-grade training observability platform.

## Guiding principles

- **Minimal training code changes.** A user should be able to add trainscope with two lines of code.
- **No lock-in.** All data is stored in open formats (Arrow, JSON) and can be read without the UI.
- **Scalability by default.** Histograms and activation metrics should not block the training loop.
- **Observability of the observer.** trainscope itself must expose health, metrics, and logs.

## Phases

### Phase 1 — Core productionization

- [x] Robust lifecycle management and writer consistency
- [x] Config validation and sensible defaults
- [ ] **YAML / env-var configuration** — `trainscope.yaml`, `TRAINSCOPE_*` overrides, profile presets (`minimal`, `debug`, `production`)
- [ ] **Structured logging** — `structlog` or stdlib logging with JSON output option
- [ ] **Remote storage** — S3/GCS/Azure writers via `fsspec`; local writer remains default
- [ ] **Plugin system** — custom global/layer metrics and anomaly detectors via entry points
- [ ] **Advanced anomaly detection** — change-point detection, isolation forest, percentile thresholds
- [ ] **WebSocket streaming** — live global/layer updates to the UI during training

### Phase 2 — Integrations & operations

- [ ] **Experiment tracker sync** — WandB, TensorBoard, MLflow callbacks
- [ ] **Alerting** — Slack / email / PagerDuty on spike detection
- [ ] **Authentication & multi-user** — API tokens or OIDC for the UI server
- [ ] **Telemetry** — OpenTelemetry traces/metrics for the server
- [ ] **Docker & deployment** — Dockerfile, docker-compose, Helm chart
- [ ] **Benchmark suite** — overhead measurements across model sizes and devices

### Phase 3 — Frontend platform

- [ ] **Design system** — Tailwind CSS, consistent spacing, typography, color palette
- [ ] **Real-time UI** — WebSocket-driven live charts and spike notifications
- [ ] **Multi-run comparison** — diff two runs side-by-side
- [ ] **Mobile-responsive layout** — usable on tablets and phones
- [ ] **Export & sharing** — PNG/SVG export, shareable view URLs

### Phase 4 — Enterprise features

- [ ] **Distributed training** — DDP/FSDP aware aggregation, per-rank dashboards
- [ ] **Long-term storage** — compaction, parquet conversion, archival policies
- [ ] **RBAC & audit logging** — user actions logged
- [ ] **Hosted cloud version** — managed service architecture

## Current priorities

The next development cycle focuses on Phase 1 and the Docker/integration groundwork in Phase 2. These changes have the highest impact on making the project feel production-ready while keeping the core API stable.
