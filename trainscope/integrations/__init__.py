"""TrainScope integrations with experiment trackers and alerting backends.

The submodules here are optional: they only import their third-party libraries
when instantiated. Install the relevant extras to use them::

    pip install trainscope[integrations]
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _safe_pop(cfg: dict[str, Any], key: str) -> Any:
    """Return a copy of ``cfg`` with ``key`` removed."""
    return {k: v for k, v in cfg.items() if k != key}


def build_callbacks(integrations_config: dict[str, Any] | None) -> list[Any]:
    """Build experiment-tracker callbacks from a config mapping.

    If WandB is imported and a run is active (`wandb.run`), it is automatically
    detected unless explicitly disabled via `integrations={"wandb": False}`.

    Example config::

        {
            "wandb": {"project": "my-project", "entity": "me"},
            "tensorboard": {"log_dir": "./runs"},
            "mlflow": {"experiment_name": "trainscope"},
        }
    """
    callbacks: list[Any] = []
    cfg_dict = integrations_config if isinstance(integrations_config, dict) else {}

    # Explicit integrations
    for name, cfg in cfg_dict.items():
        if cfg is None or cfg is False:
            continue
        kwargs = cfg if isinstance(cfg, dict) else {}

        if name == "wandb":
            from trainscope.integrations.wandb_ import WandbCallback

            callbacks.append(WandbCallback(**kwargs))
        elif name == "tensorboard":
            from trainscope.integrations.tensorboard_ import TensorBoardCallback

            callbacks.append(TensorBoardCallback(**kwargs))
        elif name == "mlflow":
            from trainscope.integrations.mlflow_ import MlflowCallback

            callbacks.append(MlflowCallback(**kwargs))
        else:
            logger.warning("Unknown integration '%s'; skipping", name)

    # Automatic WandB detection: if wandb is loaded & wandb.run is active,
    # and wandb was not explicitly disabled or already configured.
    if cfg_dict.get("wandb") is not False and "wandb" not in cfg_dict:
        import sys

        wandb_mod = sys.modules.get("wandb")
        if wandb_mod is not None and getattr(wandb_mod, "run", None) is not None:
            try:
                from trainscope.integrations.wandb_ import WandbCallback

                callbacks.append(WandbCallback())
                logger.info("Auto-detected active WandB run; attached WandbCallback")
            except Exception:
                logger.debug("WandB module found but WandbCallback initialization skipped")

    return callbacks


def build_alerters(alerts_config: list[dict[str, Any]] | None) -> list[Any]:
    """Build alerters from a list of alert config dicts.

    Example config::

        [
            {"type": "slack", "webhook_url": "https://hooks.slack.com/..."},
            {"type": "email", "to": "ops@example.com", "smtp_host": "smtp.example.com"},
        ]
    """
    alerters: list[Any] = []
    if not alerts_config:
        return alerters

    for cfg in alerts_config:
        alert_type = cfg.get("type")
        kwargs = _safe_pop(cfg, "type")

        if alert_type == "slack":
            from trainscope.integrations.alerts import SlackAlerter

            alerters.append(SlackAlerter(**kwargs))
        elif alert_type == "email":
            from trainscope.integrations.alerts import EmailAlerter

            alerters.append(EmailAlerter(**kwargs))
        elif alert_type is None or alert_type == "null":
            from trainscope.integrations.alerts import NullAlerter

            alerters.append(NullAlerter(**kwargs))
        else:
            logger.warning("Unknown alert type '%s'; skipping", alert_type)

    return alerters
