"""Prometheus telemetry for the TrainScope UI server.

Exposes a ``/metrics`` endpoint with counters/gauges for HTTP requests,
WebSocket connections, and loaded runs.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        generate_latest,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Telemetry requires prometheus-client. Install it with: pip install trainscope[telemetry]"
    ) from exc

logger = logging.getLogger(__name__)


class Telemetry:
    """Container for TrainScope Prometheus metrics."""

    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()
        self.requests_total = Counter(
            "trainscope_requests_total",
            "Total HTTP requests received by the UI server",
            ["method", "path", "status_code"],
            registry=self.registry,
        )
        self.ws_connections = Gauge(
            "trainscope_ws_connections",
            "Number of active WebSocket connections",
            registry=self.registry,
        )
        self.runs_loaded = Gauge(
            "trainscope_runs_loaded",
            "Number of run directories currently loaded by the server",
            registry=self.registry,
        )

    def metrics_response(self) -> Response:
        return Response(
            content=generate_latest(self.registry),
            media_type=CONTENT_TYPE_LATEST,
        )


class TelemetryMiddleware(BaseHTTPMiddleware):
    """Record request counts and WebSocket connection gauges."""

    def __init__(self, app, telemetry: Telemetry):
        super().__init__(app)
        self._telemetry = telemetry

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        try:
            self._telemetry.requests_total.labels(
                method=request.method,
                path=request.url.path,
                status_code=str(response.status_code),
            ).inc()
        except Exception:
            logger.exception("Failed to record request metric")
        return response


def add_telemetry(app: FastAPI) -> Telemetry:
    """Attach telemetry middleware and ``/metrics`` endpoint to ``app``."""
    telemetry = Telemetry()
    telemetry.runs_loaded.set(1)

    app.add_middleware(TelemetryMiddleware, telemetry=telemetry)

    @app.get("/metrics")
    async def metrics() -> Response:
        return telemetry.metrics_response()

    return telemetry
