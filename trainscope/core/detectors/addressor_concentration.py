"""Addressor-concentration drift detector for memory-augmented models.

Detects *addressing* concentration (the addressor locking onto a single
memory slot) before it becomes a loss divergence. The v1.4.1 experiment
(``scripts/verify_addressor_collapse_signal.py``) established the empirical
claim: on a mini memory-augmented transformer (16 soft-addressed slots)
trained on wikitext-2, mean max-slot addressing share exceeding ``0.6`` for
several consecutive steps preceded the loss explosion by 7-11 steps (mean
9.3) in 3/3 seeds, with zero false positives in the stable control.

Like the MoE experiment, the same run showed that a *dead* slot (mean
addressing weight below a small threshold) is NOT a signal: with 16 slots
one slot lingers near zero in every step of healthy runs too. This detector
therefore keys on concentration, not on slot abandonment.
"""

from trainscope.core.detectors.expert_utilization import ExpertUtilizationDriftDetector


class AddressorConcentrationDriftDetector(ExpertUtilizationDriftDetector):
    """Report spikes when per-step max addressing share durably exceeds a threshold.

    The signal consumed is the *max slot share* of the step (mean over
    tokens of the softmax addressing weights, max over slots), not the
    loss. The scope feeds it via ``update()`` when this detector is active
    (see ``TrainScope.step``).

    Threshold default 0.6 follows the "control max + margin" rule from the
    experiment: the healthy control's max slot share peaks at 0.24-0.32
    (16 slots), so 0.6 is a durable ~2x margin above it — the same rule
    that set MoE's 0.85 above its control's 0.74.

    Score semantics: triggered steps return the max slot share at trigger
    time (``>= threshold``), preserving the magnitude of the concentration.
    """

    def __init__(
        self,
        threshold: float = 0.6,
        min_observations: int = 30,
        run_steps: int = 3,
        window: int = 200,
    ):
        super().__init__(
            threshold=threshold,
            min_observations=min_observations,
            run_steps=run_steps,
            window=window,
        )


__all__ = ["AddressorConcentrationDriftDetector"]
