"""Structured logging configuration for trainscope."""

import logging
import sys

try:
    import structlog
except Exception:  # pragma: no cover
    structlog = None  # type: ignore[assignment]


def configure_logging(level: str | int = "INFO", json_format: bool = False) -> None:
    """Configure trainscope logging.

    Parameters
    ----------
    level:
        Logging level (e.g. ``"INFO"`` or ``logging.INFO``).
    json_format:
        If True, emit JSON logs. When ``structlog`` is installed it will be
        used; otherwise a stdlib JSON formatter is used as a fallback.
    """
    if isinstance(level, str):
        level_value = getattr(logging, level.upper(), logging.INFO)
    else:
        level_value = level

    if structlog is not None and json_format:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.make_filtering_bound_logger(level_value),
            cache_logger_on_first_use=True,
        )
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=level_value,
            force=True,
        )
    else:
        if json_format:
            fmt = (
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
                '"logger": "%(name)s", "message": "%(message)s"}'
            )
        else:
            fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        logging.basicConfig(
            level=level_value,
            format=fmt,
            stream=sys.stdout,
            force=True,
        )


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger under the ``trainscope`` namespace."""
    if name is None:
        return logging.getLogger("trainscope")
    return logging.getLogger(f"trainscope.{name}")


__all__ = ["configure_logging", "get_logger"]
