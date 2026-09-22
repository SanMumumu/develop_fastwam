"""Trainer that supplies optimizer-step progress to the router and distiller.

Deliberately as small as :class:`~fastwam.threshold_trainer.ThresholdWan22Trainer`:
the router warmup, the distillation ramp and the EMA teacher update are all
driven from the progress callback consumed inside ``RoutedWAM.training_loss``,
so the base training loop needs no modification.
"""

from __future__ import annotations

from .trainer import Wan22Trainer


class RoutedWan22Trainer(Wan22Trainer):
    """Wan22Trainer wired to a RoutedWAM's progress-dependent machinery."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model = self.accelerator.unwrap_model(self.model)
        if not hasattr(model, "set_training_progress_provider"):
            raise TypeError("RoutedWan22Trainer requires a RoutedWAM.")
        model.set_training_progress_provider(
            lambda: (
                self.global_step,
                self.max_steps,
                bool(self.log_every > 0 and (self.global_step + 1) % self.log_every == 0),
            )
        )
