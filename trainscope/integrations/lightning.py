"""PyTorch Lightning integration.

One line to record a Lightning run::

    from trainscope.integrations.lightning import TrainScopeCallback

    trainer = L.Trainer(..., callbacks=[TrainScopeCallback()])
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch

try:
    from lightning.pytorch.callbacks import Callback
except ImportError as exc:
    raise ImportError(
        "TrainScopeCallback requires lightning>=2.0. Install it with: pip install lightning"
    ) from exc

from trainscope.core.config import TrainScopeConfig
from trainscope.scope import StopTraining, TrainScope


class TrainScopeCallback(Callback):
    """Record every optimizer step of a Lightning run with TrainScope.

    Metrics are taken in ``on_before_optimizer_step``: after gradient
    accumulation and AMP unscaling, before clipping and the optimizer step,
    so grad norms are the true pre-clip values. The recorded loss is the
    training loss of the accumulated batch. Each rank writes its own
    ``_rank{N}`` run under multi-process training. With
    ``stop_on_spike=True`` a spike stops training gracefully.
    """

    def __init__(self, config: TrainScopeConfig | None = None):
        self._config = config or TrainScopeConfig()
        self._scope: TrainScope | None = None
        self._optimizer: Any = None
        self._losses: list[torch.Tensor] = []

    @property
    def scope(self) -> TrainScope | None:
        return self._scope

    def on_train_start(self, trainer, pl_module):
        config = self._config
        if trainer.world_size > 1 and config.rank is None:
            config = dataclasses.replace(config, rank=trainer.global_rank)
        self._optimizer = trainer.optimizers[0]
        self._scope = TrainScope(pl_module, self._optimizer, config).attach()

    def on_before_backward(self, trainer, pl_module, loss):
        # Lightning applies the AMP scaler before this hook and has already
        # divided the loss by accumulate_grad_batches, so summing the unscaled
        # micro-batch losses gives the accumulated batch loss.
        scaler = getattr(trainer.precision_plugin, "scaler", None)
        scale = float(scaler.get_scale()) if scaler is not None and scaler.is_enabled() else 1.0
        self._losses.append(loss.detach().float() / scale)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # Only the first optimizer drives the step count (multi-optimizer setups).
        if self._scope is None or optimizer is not self._optimizer or not self._losses:
            return
        loss = float(torch.stack(self._losses).sum())
        self._losses.clear()
        try:
            self._scope.step(loss)
        except StopTraining:
            trainer.should_stop = True

    def on_train_end(self, trainer, pl_module):
        if self._scope is not None:
            self._scope.detach()
            self._scope = None
