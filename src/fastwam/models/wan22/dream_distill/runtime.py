"""Hydra factory for :class:`DreamDistillFastWAM`.

The model config's ``_target_`` points straight at this module, so
``src/fastwam/runtime.py`` needs no edit and this whole experiment stays
deletable. It deliberately reuses ``create_dream_fastwam`` rather than
duplicating its ~120 lines of config validation: everything about building the
underlying DreamFastWAM must stay identical between the R0 baseline and the DID
rungs, or the ablation is not measuring what it claims to.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from fastwam.runtime import create_dream_fastwam
from fastwam.utils.logging_config import get_logger

from .model import DreamDistillFastWAM

logger = get_logger(__name__)


def create_dream_distill_fastwam(
    *,
    dream_distill: Optional[Any] = None,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    **kwargs,
):
    """Build a DreamFastWAM and attach the distillation head to it.

    Every argument other than ``dream_distill`` is forwarded verbatim to
    ``create_dream_fastwam``, so the two rungs share one construction path.
    """
    if isinstance(dream_distill, DictConfig):
        dream_distill = OmegaConf.to_container(dream_distill, resolve=True)
    if dream_distill is None:
        dream_distill = {}
    if not isinstance(dream_distill, dict):
        raise ValueError(f"`dream_distill` must resolve to a dict, got {type(dream_distill)}")

    model = create_dream_fastwam(model_dtype=model_dtype, device=device, **kwargs)

    # Promote the instance in place. The distillation adds only the per-layer
    # projectors, so an existing dense DreamFastWAM checkpoint still loads (the
    # projectors are simply reported as freshly initialized).
    model.__class__ = DreamDistillFastWAM
    model._install_distillation(dream_distill)
    return model
