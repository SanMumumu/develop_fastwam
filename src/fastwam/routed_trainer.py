"""Trainer that supplies optimizer-step progress to the router and distiller.

Deliberately as small as :class:`~fastwam.threshold_trainer.ThresholdWan22Trainer`:
the router warmup, the distillation ramp and the EMA teacher update are all
driven from the progress callback consumed inside ``RoutedWAM.training_loss``,
so the base training loop needs no modification.
"""

from __future__ import annotations

from pathlib import Path
import math

import torch

from .trainer import Wan22Trainer
from .utils.logging_config import get_logger


logger = get_logger(__name__)


class RoutedWan22Trainer(Wan22Trainer):
    """Wan22Trainer wired to a RoutedWAM's progress-dependent machinery."""

    def _load_weight_checkpoint_before_optimizer(self):
        super()._load_weight_checkpoint_before_optimizer()
        # Only a weight-file warm start resets the router. Full-state resumes
        # and ordinary inference checkpoint loading preserve learned gates.
        if (self.cfg.get("reset_router_on_weight_load", False)
                and self.resume and Path(str(self.resume)).is_file()):
            router = self.model.mot.feature_router
            if router is None or router.full:
                raise ValueError("Router reset requires static or dynamic routing.")
            router.reset_zero()
            logger.info("Reset semantic QK Router after weight load: all initial gates=0 (tanh).")

    def _configure_trainable_parameters(self, model):
        parameters = super()._configure_trainable_parameters(model)
        router_lr = self.cfg.get("router_learning_rate")
        if router_lr is not None and (not math.isfinite(float(router_lr)) or float(router_lr) <= 0):
            raise ValueError("router_learning_rate must be finite and positive, or null.")
        router = getattr(model.mot, "feature_router", None)
        router_ids = {id(p) for p in router.parameters()} if router is not None else set()
        # ZeRO flattens each optimizer group. Separate dtypes so FP32 router
        # parameters cannot be packed into the BF16 expert parameter buffer.
        groups = {}
        for parameter in parameters:
            role = "router" if id(parameter) in router_ids else "experts"
            groups.setdefault((role, parameter.dtype), []).append(parameter)
        result = []
        for (role, dtype), values in groups.items():
            lr = float(router_lr) if role == "router" and router_lr is not None else self.learning_rate
            result.append({"params": values, "lr": lr, "name": role})
            logger.info("Optimizer group %s: dtype=%s params=%d peak_lr=%.3e",
                        role, dtype, sum(p.numel() for p in values), lr)
        return result

    def _build_scheduler(self, scheduler_type, total_train_steps, warmup_steps=0):
        if self.cfg.get("router_learning_rate") is None:
            return super()._build_scheduler(scheduler_type, total_train_steps, warmup_steps)
        kind = str(scheduler_type).strip().lower()
        if kind not in {"cosine", "constant"}:
            raise ValueError(f"Unsupported lr_scheduler_type: {scheduler_type}.")
        total = max(int(total_train_steps), 1)
        warmup = min(max(int(warmup_steps), 0), total - 1)

        def factor(step):
            if warmup and step < warmup:
                return 1 / warmup + (1 - 1 / warmup) * step / warmup
            if kind == "constant":
                return 1.0
            progress = min(max((step - warmup) / (total - warmup), 0.0), 1.0)
            # Each group's minimum is 1% of its own peak, preserving LR ratios.
            return 0.01 + 0.99 * (1 + math.cos(math.pi * progress)) / 2

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=factor)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model = self.accelerator.unwrap_model(self.model)
        router = getattr(model.mot, "feature_router", None)
        if router is not None and any(p.dtype != torch.float32 for p in router.parameters()):
            raise RuntimeError("Semantic Router must remain FP32 after accelerator.prepare().")
        if not hasattr(model, "set_training_progress_provider"):
            raise TypeError("RoutedWan22Trainer requires a RoutedWAM.")
        model.set_training_progress_provider(
            lambda: (
                self.global_step,
                self.max_steps,
                bool(self.log_every > 0 and (self.global_step + 1) % self.log_every == 0),
            )
        )
