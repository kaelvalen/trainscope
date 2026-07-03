"""Tests for TrainScope UI authentication."""

from __future__ import annotations

import base64
from unittest.mock import patch

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from trainscope.ui.auth import auth_enabled, auth_middleware_factory, verify_request


def _make_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(auth_middleware_factory())

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/manifest")
    def manifest():
        return {"manifest": True}

    return app


class TestAuthEnabled:
    def test_no_env_vars_means_auth_disabled(self):
        with patch.dict("os.environ", {}, clear=True):
            assert auth_enabled() is False

    def test_api_key_env_enables_auth(self):
        with patch.dict("os.environ", {"TRAINSCOPE_API_KEY": "secret"}, clear=True):
            assert auth_enabled() is True

    def test_basic_auth_env_enables_auth(self):
        with patch.dict("os.environ", {"TRAINSCOPE_BASIC_AUTH": "admin:pass"}, clear=True):
            assert auth_enabled() is True


class TestVerifyRequest:
    def test_verify_api_key_header(self):
        with patch.dict("os.environ", {"TRAINSCOPE_API_KEY": "secret"}, clear=True):
            app = FastAPI()

            @app.get("/")
            def root(request: Request):
                return {"ok": verify_request(request)}

            client = TestClient(app)
            response = client.get("/", headers={"X-API-Key": "secret"})
            assert response.json()["ok"] is True

    def test_verify_bearer_token(self):
        with patch.dict("os.environ", {"TRAINSCOPE_API_KEY": "secret"}, clear=True):
            app = FastAPI()

            @app.get("/")
            def root(request: Request):
                return {"ok": verify_request(request)}

            client = TestClient(app)
            response = client.get("/", headers={"Authorization": "Bearer secret"})
            assert response.json()["ok"] is True

    def test_verify_basic_auth(self):
        with patch.dict("os.environ", {"TRAINSCOPE_BASIC_AUTH": "admin:pass"}, clear=True):
            app = FastAPI()

            @app.get("/")
            def root(request: Request):
                return {"ok": verify_request(request)}

            client = TestClient(app)
            credentials = base64.b64encode(b"admin:pass").decode("ascii")
            response = client.get("/", headers={"Authorization": f"Basic {credentials}"})
            assert response.json()["ok"] is True

    def test_invalid_api_key_is_rejected(self):
        with patch.dict("os.environ", {"TRAINSCOPE_API_KEY": "secret"}, clear=True):
            app = FastAPI()

            @app.get("/")
            def root(request: Request):
                return {"ok": verify_request(request)}

            client = TestClient(app)
            response = client.get("/", headers={"X-API-Key": "wrong"})
            assert response.json()["ok"] is False


class TestAuthMiddleware:
    def test_public_health_endpoint_unprotected(self):
        with patch.dict("os.environ", {"TRAINSCOPE_API_KEY": "secret"}, clear=True):
            client = TestClient(_make_app())
            response = client.get("/api/health")
            assert response.status_code == 200
            assert response.json()["status"] == "ok"

    def test_private_endpoint_requires_auth(self):
        with patch.dict("os.environ", {"TRAINSCOPE_API_KEY": "secret"}, clear=True):
            client = TestClient(_make_app())
            response = client.get("/api/manifest")
            assert response.status_code == 401

    def test_private_endpoint_accepts_valid_key(self):
        with patch.dict("os.environ", {"TRAINSCOPE_API_KEY": "secret"}, clear=True):
            client = TestClient(_make_app())
            response = client.get("/api/manifest", headers={"X-API-Key": "secret"})
            assert response.status_code == 200
            assert response.json()["manifest"] is True

    def test_disabled_auth_allows_all_requests(self):
        with patch.dict("os.environ", {}, clear=True):
            client = TestClient(_make_app())
            response = client.get("/api/manifest")
            assert response.status_code == 200
