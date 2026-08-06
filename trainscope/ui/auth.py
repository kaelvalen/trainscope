"""Authentication helpers for the TrainScope UI server.

Auth is disabled by default. Set ``TRAINSCOPE_API_KEY`` to enable API-key
authentication, or set ``TRAINSCOPE_BASIC_AUTH`` to ``user:password`` to enable
HTTP Basic authentication. Both may be set at the same time.
"""

from __future__ import annotations

import base64
import os
import secrets
from typing import Callable

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import HTTPConnection

API_KEY_HEADER = "X-API-Key"
AUTH_BEARER_PREFIX = "Bearer "
AUTH_BASIC_PREFIX = "Basic "


def _get_api_key() -> str | None:
    return os.environ.get("TRAINSCOPE_API_KEY") or None


def _get_basic_auth() -> tuple[str, str] | None:
    value = os.environ.get("TRAINSCOPE_BASIC_AUTH")
    if not value:
        return None
    if ":" not in value:
        return None
    user, password = value.split(":", 1)
    return user, password


def _extract_bearer_token(request: HTTPConnection) -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.startswith(AUTH_BEARER_PREFIX):
        return auth[len(AUTH_BEARER_PREFIX) :].strip()
    return request.headers.get(API_KEY_HEADER)


def _extract_basic_credentials(request: HTTPConnection) -> tuple[str, str] | None:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith(AUTH_BASIC_PREFIX):
        return None
    try:
        decoded = base64.b64decode(auth[len(AUTH_BASIC_PREFIX) :]).decode("utf-8")
    except Exception:
        return None
    if ":" not in decoded:
        return None
    user, password = decoded.split(":", 1)
    return user, password


def auth_enabled() -> bool:
    """Return True if any authentication method is configured."""
    return _get_api_key() is not None or _get_basic_auth() is not None


def verify_request(request: HTTPConnection) -> bool:
    """Return True if the request or WebSocket connection satisfies the configured auth checks."""
    expected_api_key = _get_api_key()
    if expected_api_key is not None:
        provided = _extract_bearer_token(request)
        if not secrets.compare_digest(expected_api_key, provided or ""):
            return False

    basic_auth = _get_basic_auth()
    if basic_auth is not None:
        provided_basic = _extract_basic_credentials(request)
        if provided_basic is None:
            return False
        user, password = basic_auth
        if not secrets.compare_digest(user, provided_basic[0]):
            return False
        if not secrets.compare_digest(password, provided_basic[1]):
            return False

    return True


def auth_middleware_factory(
    public_paths: tuple[str, ...] = ("/api/health", "/metrics"),
) -> Callable:
    """Return a middleware class that protects endpoints when auth is enabled."""

    class AuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            if not auth_enabled():
                return await call_next(request)

            path = request.url.path
            if any(path == public or path.startswith(public + "/") for public in public_paths):
                return await call_next(request)

            if verify_request(request):
                return await call_next(request)

            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized"},
                headers={"WWW-Authenticate": "Bearer, Basic"},
            )

    return AuthMiddleware
