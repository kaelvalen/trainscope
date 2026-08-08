"""Tests for TrainScope UI telemetry."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainscope.ui.telemetry import Telemetry, add_telemetry


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    add_telemetry(app)

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/layers/{layer_name}")
    def layer(layer_name: str):
        return {"layer": layer_name}

    return app


def test_metrics_endpoint_returns_prometheus_format(app):
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "trainscope_runs_loaded 1.0" in response.text


def test_requests_total_is_incremented(app):
    client = TestClient(app)
    client.get("/api/health")

    response = client.get("/metrics")
    assert response.status_code == 200
    assert (
        'trainscope_requests_total{method="GET",path="/api/health",status_code="200"} 1.0'
        in response.text
    )


def test_path_label_uses_route_template_not_concrete_path(app):
    """Path-parameterized endpoints must be labelled with the route template
    (e.g. /api/layers/{layer_name}) so each unique layer name doesn't create
    a separate time series."""
    client = TestClient(app)
    client.get("/api/layers/transformer.h.0.attn")
    client.get("/api/layers/transformer.h.1.attn")

    response = client.get("/metrics")
    assert response.status_code == 200
    assert (
        'trainscope_requests_total{method="GET",path="/api/layers/{layer_name}",'
        'status_code="200"} 2.0' in response.text
    )
    assert "transformer.h.0.attn" not in response.text
    assert "transformer.h.1.attn" not in response.text


def test_telemetry_metrics_are_registered():
    telemetry = Telemetry()
    assert telemetry.requests_total is not None
    assert telemetry.ws_connections is not None
    assert telemetry.runs_loaded is not None


def test_ws_connections_gauge_can_be_set(app):
    client = TestClient(app)
    response = client.get("/metrics")
    # The gauge is present and starts at zero.
    assert "trainscope_ws_connections 0.0" in response.text
