"""Interface distillation: compress the world rollout into the tensor that is read.

The action expert never sees the imagined depth map, DINO field or segmentation.
Across the whole MoT it only ever reads the **per-layer keys and values** of the
Video and Dream tokens -- the object that ``prefill_video_dream_cache`` stores.
Distilling in output space therefore spends capacity on detail that no consumer
can observe.

``InterfaceDistiller`` supervises that interface directly: a one-step student
Dream pass is asked to reproduce the per-layer Dream K/V that an ``N``-step EMA
teacher converges to.  Because the Video expert is frozen and sees one clean
frame, its K/V are identical for teacher and student, so the whole comparison
reduces to the Dream half and only the (small) Dream expert has to be duplicated.

When ``route_aware`` is set, the loss is restricted to the tokens the router
actually keeps: there is no reason to match an interface slot the policy has
already decided to ignore.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn

from fastwam.utils.logging_config import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class InterfaceDistillConfig:
    enabled: bool = False
    teacher_steps: int = 8
    student_steps: int = 1
    ema_decay: float = 0.995
    layers: Optional[list[int]] = field(default=None)
    lambda_k: float = 1.0
    lambda_v: float = 1.0
    lambda_total: float = 1.0
    route_aware: bool = True
    warmup_ratio: float = 0.0
    detach_teacher: bool = True

    @classmethod
    def from_dict(cls, value: Optional[dict[str, Any]]) -> "InterfaceDistillConfig":
        cfg = cls(**({} if value is None else dict(value)))
        if cfg.teacher_steps < 1:
            raise ValueError(f"interface_distill.teacher_steps must be >= 1, got {cfg.teacher_steps}.")
        if cfg.student_steps < 1:
            raise ValueError(f"interface_distill.student_steps must be >= 1, got {cfg.student_steps}.")
        if cfg.student_steps > cfg.teacher_steps:
            raise ValueError(
                "interface_distill.student_steps must not exceed teacher_steps "
                f"({cfg.student_steps} > {cfg.teacher_steps}); the student is the cheap one."
            )
        if not 0.0 <= cfg.ema_decay < 1.0:
            raise ValueError(f"interface_distill.ema_decay must be in [0,1), got {cfg.ema_decay}.")
        for name in ("lambda_k", "lambda_v", "lambda_total"):
            if float(getattr(cfg, name)) < 0:
                raise ValueError(f"interface_distill.{name} must be >= 0.")
        if not 0.0 <= cfg.warmup_ratio <= 1.0:
            raise ValueError("interface_distill.warmup_ratio must be in [0,1].")
        if not cfg.detach_teacher:
            raise ValueError(
                "interface_distill.detach_teacher=false would let the student move the "
                "teacher, which is the collapse mode this asymmetry exists to prevent."
            )
        return cfg


class InterfaceDistiller(nn.Module):
    """Owns the EMA teacher Dream expert and the per-layer K/V loss."""

    def __init__(self, *, config: InterfaceDistillConfig, dream_expert: nn.Module, num_layers: int):
        super().__init__()
        self.config = config
        self.num_layers = int(num_layers)
        self.active_layers = (
            set(range(self.num_layers))
            if config.layers is None
            else {int(x) for x in config.layers}
        )
        unknown = {x for x in self.active_layers if not 0 <= x < self.num_layers}
        if unknown:
            raise ValueError(
                f"interface_distill.layers contains out-of-range entries: {sorted(unknown)}"
            )

        self.teacher_dream = copy.deepcopy(dream_expert)
        self.teacher_dream.requires_grad_(False)
        self.teacher_dream.eval()
        self._progress: float = 1.0
        logger.info(
            "InterfaceDistiller: teacher_steps=%d student_steps=%d ema_decay=%.4f "
            "layers=%s route_aware=%s (teacher params %.1fM)",
            config.teacher_steps,
            config.student_steps,
            config.ema_decay,
            "all" if config.layers is None else sorted(self.active_layers),
            config.route_aware,
            sum(p.numel() for p in self.teacher_dream.parameters()) / 1e6,
        )

    # ------------------------------------------------------------------ state
    def set_progress(self, fraction: float) -> None:
        self._progress = float(min(max(fraction, 0.0), 1.0))

    def current_weight(self) -> float:
        if self.config.warmup_ratio <= 0.0:
            return float(self.config.lambda_total)
        return float(self.config.lambda_total) * self._progress

    @torch.no_grad()
    def update_ema(self, student_dream: nn.Module) -> None:
        decay = float(self.config.ema_decay)
        teacher_params = dict(self.teacher_dream.named_parameters())
        for name, param in student_dream.named_parameters():
            target = teacher_params.get(name)
            if target is None:
                continue
            target.mul_(decay).add_(param.detach().to(target.dtype), alpha=1.0 - decay)
        teacher_buffers = dict(self.teacher_dream.named_buffers())
        for name, buffer in student_dream.named_buffers():
            target = teacher_buffers.get(name)
            if target is None or not torch.is_floating_point(target):
                continue
            target.copy_(buffer.detach().to(target.dtype))

    @contextmanager
    def use_teacher_dream(self, mot: nn.Module):
        """Temporarily run `mot` with the EMA teacher as its Dream expert.

        Swapping the module rather than building a second MoT keeps the 5B Video
        expert shared; only the Dream expert is duplicated.
        """
        student = mot.mixtures["dream"]
        mot.mixtures["dream"] = self.teacher_dream
        try:
            yield
        finally:
            mot.mixtures["dream"] = student

    # ------------------------------------------------------------------- loss
    def kv_loss(
        self,
        student_kv: list[dict[str, torch.Tensor]],
        teacher_kv: list[dict[str, torch.Tensor]],
        keep_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Cosine distance between student and teacher Dream K/V, per layer.

        Args:
            student_kv / teacher_kv: length-`num_layers` lists of ``{"k","v"}``
                tensors shaped ``[B, Sd, H*Dh]``.
            keep_mask: optional ``[B, Sd]`` boolean; when given, only the kept
                interface slots are supervised.
        """
        if len(student_kv) != len(teacher_kv):
            raise ValueError(
                f"K/V cache length mismatch: student={len(student_kv)} teacher={len(teacher_kv)}."
            )
        device = student_kv[0]["k"].device
        total = torch.zeros((), device=device, dtype=torch.float32)
        parts: dict[str, torch.Tensor] = {}
        counted = 0

        for layer_idx, (student, teacher) in enumerate(zip(student_kv, teacher_kv)):
            if layer_idx not in self.active_layers:
                continue
            layer_loss = torch.zeros((), device=device, dtype=torch.float32)
            for key, weight in (("k", self.config.lambda_k), ("v", self.config.lambda_v)):
                if float(weight) <= 0.0:
                    continue
                s = student[key].float()
                t = teacher[key].float()
                if self.config.detach_teacher:
                    t = t.detach()
                if s.shape != t.shape:
                    raise ValueError(
                        f"Layer {layer_idx} '{key}' shape mismatch: {tuple(s.shape)} vs {tuple(t.shape)}."
                    )
                cos = torch.nn.functional.cosine_similarity(s, t, dim=-1, eps=1.0e-6)
                per_token = 1.0 - cos
                if keep_mask is not None:
                    mask = keep_mask.to(device=per_token.device, dtype=per_token.dtype)
                    denominator = mask.sum().clamp(min=1.0)
                    value = (per_token * mask).sum() / denominator
                else:
                    value = per_token.mean()
                layer_loss = layer_loss + float(weight) * value
                parts[f"iface_l{layer_idx}_{key}"] = value.detach()
            total = total + layer_loss
            counted += 1

        if counted == 0:
            return total, parts
        return total / float(counted), parts
