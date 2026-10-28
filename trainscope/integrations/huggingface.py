"""Hugging Face ``transformers.Trainer`` integration.

One line to record a ``Trainer`` run::

    from trainscope.integrations.huggingface import TrainScopeCallback

    trainer = Trainer(..., callbacks=[TrainScopeCallback()])
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import torch

try:
    from transformers import TrainerCallback
except ImportError as exc:
    raise ImportError(
        "TrainScopeCallback requires transformers>=4.41. Install it with: pip install transformers"
    ) from exc

from trainscope.core.config import TrainScopeConfig
from trainscope.scope import StopTraining, TrainScope

logger = logging.getLogger(__name__)


class TrainScopeCallback(TrainerCallback):
    """Record every optimizer step of a ``Trainer`` run with TrainScope.

    Metrics are taken in ``on_pre_optimizer_step``: after gradient
    accumulation, AMP unscaling and clipping, before the optimizer mutates
    the weights. The recorded loss is the mean of the training-mode forward
    losses (``outputs.loss``) since the previous optimizer step. Under
    multi-process training each rank writes its own ``_rank{N}`` run. With
    ``stop_on_spike=True`` a spike stops training gracefully.
    """

    def __init__(self, config: TrainScopeConfig | None = None):
        self._config = config or TrainScopeConfig()
        self._scope: TrainScope | None = None
        self._losses: list[torch.Tensor] = []
        self._loss_hook: Any = None
        self._losses_are_partial = False
        self._loss_multiplier = 1
        self._warned_no_loss = False

    @property
    def scope(self) -> TrainScope | None:
        return self._scope

    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        config = self._config
        if args.world_size > 1 and config.rank is None:
            config = dataclasses.replace(config, rank=args.process_index)
        self._scope = TrainScope(model, optimizer, config).attach()
        self._loss_hook = model.register_forward_hook(self._capture_loss, with_kwargs=True)
        # Token-averaged across ranks, each rank's loss is local_sum / global_items.
        self._loss_multiplier = (
            args.world_size if getattr(args, "average_tokens_across_devices", False) else 1
        )

    def _capture_loss(self, module, _args, kwargs, output) -> None:
        # Evaluation forwards run in eval mode between optimizer steps.
        if not module.training:
            return
        loss = output.get("loss") if isinstance(output, dict) else None
        if isinstance(loss, torch.Tensor):
            # With num_items_in_batch the Trainer has the model normalize each
            # micro-batch loss by the whole accumulated batch: the step loss is
            # their sum. Without it, each is a per-micro-batch mean.
            self._losses_are_partial = kwargs.get("num_items_in_batch") is not None
            self._losses.append(loss.detach().float())

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if self._scope is None:
            return
        if not self._losses:
            if not self._warned_no_loss:
                logger.warning(
                    "TrainScopeCallback: the model returned no `loss` in its outputs; "
                    "steps are not recorded. Return a ModelOutput/dict with `loss`."
                )
                self._warned_no_loss = True
            return
        stacked = torch.stack(self._losses)
        if self._losses_are_partial:
            loss = float(stacked.sum()) * self._loss_multiplier
        else:
            loss = float(stacked.mean())
        self._losses.clear()
        try:
            self._scope.step(loss)
        except StopTraining:
            control.should_training_stop = True

    def on_train_end(self, args, state, control, **kwargs):
        if self._loss_hook is not None:
            self._loss_hook.remove()
            self._loss_hook = None
        if self._scope is not None:
            self._scope.detach()
            self._scope = None
