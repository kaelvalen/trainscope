"""Reusable TrainScope auth middleware."""

from trainscope.ui.auth import auth_middleware_factory

__all__ = ["AuthMiddleware"]

AuthMiddleware = auth_middleware_factory()
