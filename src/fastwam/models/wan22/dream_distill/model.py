"""``DreamFastWAM`` plus the Dream Imagination Distillation objective.

See ``__init__.py`` for the motivation. This module only adds a training-time
loss and a small per-layer projector; the inference path is untouched, so a
DID-trained checkpoint runs at exactly the same test-time cost as a stock
``DreamFastWAM`` one.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

from ..dream_fastwam.model import DreamFastWAM
from .mot import DistillMoT

logger = get_logger(__name__)

TEACHER_MODES = ("future", "current")


class DreamDistillFastWAM(DreamFastWAM):
    """DreamFastWAM with a train-time distillation from video future frames.

    The extra loss term is::

        L_did = mean over (layer, offset) of [ 1 - cos( P_l(student), sg(teacher) ) ]

    where

    * ``student``  = mean of the Dream tokens belonging to one future offset,
      taken at layer ``l`` (shape ``[B, dream_hidden]``)
    * ``teacher``  = mean of the video expert's tokens for the corresponding
      latent frame, at the same layer (shape ``[B, video_hidden]``)
    * ``P_l``      = a per-layer ``Linear(dream_hidden -> video_hidden)``
    * ``sg``       = stop-gradient

    The stop-gradient follows the standard BYOL/SimSiam asymmetry: without it the
    cheapest way to reduce ``L_did`` would be for the video expert to collapse its
    future representation into something trivially predictable. Note, however, that
    every shipped rung config sets ``freeze_video_expert: true``, so the teacher has
    no gradient path regardless and this ``detach()`` is defensive rather than
    load-bearing -- it only starts to matter if the video expert is unfrozen.

    This is a recombination of standard parts, not a new mechanism. The loss is
    SimSiam's ``D(p, stopgrad(z))`` with a linear predictor, applied at an
    intermediate layer in the manner of FitNets hint layers / REPA. See the
    prior-art table in ``README_dream_distill.md`` before describing it as a method.
    """

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def _install_distillation(self, cfg: dict[str, Any]) -> None:
        self.did_enabled = bool(cfg.get("enabled", True))
        self.did_lambda = float(cfg.get("lambda_did", 1.0))
        self.did_teacher = str(cfg.get("teacher", "future"))
        if self.did_teacher not in TEACHER_MODES:
            raise ValueError(
                f"dream_distill.teacher must be one of {TEACHER_MODES}, got {self.did_teacher!r}"
            )
        layers = cfg.get("layers", None)
        if layers is None:
            layers = [self.mot.num_layers - 1]
        self.did_layers = sorted({int(x) for x in layers})

        # Swap in the capture-capable MoT. Adds no parameters, so dense
        # DreamFastWAM checkpoints stay loadable -- same class-swap trick the
        # action_dream_threshold variant uses.
        if not isinstance(self.mot, DistillMoT):
            self.mot.__class__ = DistillMoT
            # __init__ is not re-run by a class swap, so seed the capture state.
            self.mot._capture_layers = set()
            self.mot._capture_experts = set()
            self.mot._capture_enabled = False
            self.mot._snapshots = {}
            self.mot._block_index = None
        self.dit = self.mot

        # R4 (oracle rung): let Dream attend to the whole video sequence,
        # including the noisy future frames, at train AND test time. This is a
        # genuine architecture change, not a loss-weight tweak: it breaks the
        # Video+Dream prefill (Dream would depend on the denoising video) and
        # costs a full video forward at inference. Its only purpose is to bound
        # how much a perfect version of DID could possibly buy.
        self.did_dream_video_access = str(cfg.get("dream_video_access", "current"))
        if self.did_dream_video_access not in ("current", "all"):
            raise ValueError(
                "dream_distill.dream_video_access must be 'current' or 'all', "
                f"got {self.did_dream_video_access!r}"
            )
        if self.did_dream_video_access == "all":
            logger.warning(
                "dream_video_access='all' (R4 oracle): Dream reads the noisy future video tokens. "
                "The Video+Dream KV prefill is NO LONGER VALID for this model and inference must "
                "run the full video branch. Do not compare its test-time cost with the other rungs."
            )

        if not self.did_enabled:
            self.did_projectors = None
            logger.info("Dream Imagination Distillation is DISABLED for this model.")
            return

        for layer in self.did_layers:
            if not 0 <= layer < self.mot.num_layers:
                raise ValueError(
                    f"dream_distill.layers contains {layer}, outside [0, {self.mot.num_layers})"
                )
        self.mot.configure_capture(layers=self.did_layers, experts=("video", "dream"))

        dream_hidden = int(self.dream_expert.hidden_dim)
        video_hidden = int(self.video_expert.hidden_dim)
        # Projector lives on the student side (standard for self-distillation:
        # a teacher-side projector could absorb the mismatch instead of forcing
        # the student's representation to move).
        self.did_projectors = nn.ModuleDict(
            {str(layer): nn.Linear(dream_hidden, video_hidden) for layer in self.did_layers}
        ).to(device=self.device, dtype=self.torch_dtype)

        self._did_validate_offsets()
        logger.info(
            "Dream Imagination Distillation enabled: layers=%s teacher=%s lambda=%.4g "
            "(%.3fM extra params, train-time only)",
            self.did_layers,
            self.did_teacher,
            self.did_lambda,
            sum(p.numel() for p in self.did_projectors.parameters()) / 1e6,
        )

    def _did_validate_offsets(self) -> None:
        """Check every dream offset maps onto an integer latent-frame index."""
        factor = int(self.vae.temporal_downsample_factor)
        if factor <= 0:
            raise ValueError(f"vae.temporal_downsample_factor must be positive, got {factor}")
        self._did_temporal_factor = factor

        offsets = list(self.dream_expert.future_offsets)
        frames = []
        for offset in offsets:
            if offset % factor != 0:
                raise ValueError(
                    f"dream_target.future_offsets contains {offset}, which is not a multiple of the "
                    f"VAE temporal downsample factor ({factor}). Distillation requires each offset to "
                    "land exactly on a latent frame; rounding would silently distil the wrong frame."
                )
            frames.append(offset // factor)
        self._did_latent_frames = frames

        if self.did_teacher == "future" and all(f == 0 for f in frames):
            raise ValueError(
                "dream_distill.teacher='future' but every future_offset maps to latent frame 0 "
                "(the current frame). That is the R3 control, not the treatment -- set "
                "teacher='current' if this is intentional."
            )
        if self.did_teacher == "future" and self.loss_lambda_video <= 0.0:
            raise ValueError(
                "dream_distill.teacher='future' requires lambda_video > 0. With lambda_video=0 the "
                "video branch is fed a single latent frame, so there are no future-frame tokens to "
                "distil from."
            )

    # ------------------------------------------------------------------
    # capture plumbing
    # ------------------------------------------------------------------
    def _build_mot_attention_mask(self, *args, **kwargs):
        # The mask builder is the one place that already knows the video token
        # geometry, so record it here instead of re-deriving it later.
        mask = super()._build_mot_attention_mask(*args, **kwargs)
        video_seq_len = int(kwargs["video_seq_len"])
        dream_seq_len = int(kwargs["dream_seq_len"])
        self._did_tokens_per_frame = int(kwargs["video_tokens_per_frame"])
        self._did_video_seq_len = video_seq_len

        if getattr(self, "did_dream_video_access", "current") == "all":
            # R4 oracle: widen the dream -> video block from the current frame
            # to the whole video sequence.
            dream_start = video_seq_len
            dream_end = video_seq_len + dream_seq_len
            mask[dream_start:dream_end, :video_seq_len] = True
        return mask

    def _prefill_video_dream_cache(self, *args, **kwargs):
        # The Video+Dream prefill is only valid because Dream reads a single
        # clean frame and is therefore independent of the video denoising state.
        # Under the R4 oracle that premise is gone, and silently reusing the
        # cache would evaluate a different model than the one that was trained.
        if getattr(self, "did_dream_video_access", "current") == "all":
            raise RuntimeError(
                "dream_video_access='all' (R4 oracle) makes the Dream branch depend on the noisy "
                "future video tokens, so the Video+Dream KV prefill is no longer valid. Evaluate "
                "this rung through the full joint path (infer_joint) instead of infer_action."
            )
        return super()._prefill_video_dream_cache(*args, **kwargs)

    # ------------------------------------------------------------------
    # loss
    # ------------------------------------------------------------------
    def _did_pool_video_frame(self, video_tokens: torch.Tensor, frame_idx: int) -> torch.Tensor:
        tpf = self._did_tokens_per_frame
        start, end = frame_idx * tpf, (frame_idx + 1) * tpf
        if end > video_tokens.shape[1]:
            raise ValueError(
                f"latent frame {frame_idx} needs video tokens [{start}:{end}] but the video "
                f"sequence is only {video_tokens.shape[1]} long. Check num_frames / "
                "action_video_freq_ratio against dream_target.future_offsets."
            )
        return video_tokens[:, start:end, :].mean(dim=1)

    def _did_pool_dream_offset(self, dream_tokens: torch.Tensor, offset_idx: int) -> torch.Tensor:
        """Mean-pool every dream token belonging to one future offset.

        Dream queries are stored as ``[num_future_offsets, n_m, hidden]`` and
        flattened offset-major inside each modality's block, so offset ``oi`` of
        modality ``m`` occupies ``[oi * n_m, (oi + 1) * n_m)`` within that block.
        """
        chunks = []
        for modality, mod_slice in self.dream_expert.modality_slices().items():
            n_m = int(getattr(self.dream_expert, f"n_{modality}"))
            base = mod_slice.start + offset_idx * n_m
            chunks.append(dream_tokens[:, base : base + n_m, :])
        return torch.cat(chunks, dim=1).mean(dim=1)

    def _compute_did_loss(self, snapshots) -> tuple[torch.Tensor, dict[str, float]]:
        parts: list[torch.Tensor] = []
        stats: dict[str, float] = {}

        for layer in self.did_layers:
            video_tokens = snapshots.get(("video", layer))
            dream_tokens = snapshots.get(("dream", layer))
            if video_tokens is None or dream_tokens is None:
                raise RuntimeError(
                    f"DID expected captured video+dream states at layer {layer}; "
                    f"got keys {sorted(snapshots)}"
                )
            projector = self.did_projectors[str(layer)]

            for offset_idx, latent_frame in enumerate(self._did_latent_frames):
                target_frame = 0 if self.did_teacher == "current" else latent_frame
                # Detaching here (not in the capture hook) is what keeps this a
                # one-way distillation: the video expert is never asked to make
                # itself easier to predict.
                teacher = self._did_pool_video_frame(video_tokens, target_frame).detach()
                student = projector(self._did_pool_dream_offset(dream_tokens, offset_idx))
                cos = F.cosine_similarity(student.float(), teacher.float(), dim=-1)
                loss = (1.0 - cos).mean()
                parts.append(loss)
                key = f"did_l{layer}_f{target_frame}"
                stats[key] = float(loss.detach().item())

        total = torch.stack(parts).mean()
        return total, stats

    def training_loss(self, sample, tiled: bool = False):
        if not getattr(self, "did_enabled", False) or self.did_lambda == 0.0:
            return super().training_loss(sample, tiled=tiled)

        self.mot.enable_capture(True)
        try:
            loss_total, loss_dict = super().training_loss(sample, tiled=tiled)
            snapshots = self.mot.pop_snapshots()
        finally:
            self.mot.enable_capture(False)

        loss_did, did_stats = self._compute_did_loss(snapshots)
        loss_total = loss_total + self.did_lambda * loss_did

        loss_dict = dict(loss_dict)
        loss_dict["loss_did"] = self.did_lambda * float(loss_did.detach().item())
        for key, value in did_stats.items():
            loss_dict[key] = value
        return loss_total, loss_dict

    # ------------------------------------------------------------------
    # trainable parameters / checkpointing
    # ------------------------------------------------------------------
    def configure_trainable_parameters(self, freeze_video_expert: bool = False):
        params = super().configure_trainable_parameters(freeze_video_expert=freeze_video_expert)
        if getattr(self, "did_projectors", None) is not None:
            self.did_projectors.train()
            self.did_projectors.requires_grad_(True)
            params = params + list(self.did_projectors.parameters())
            logger.info(
                "DID projectors trainable: %.3fM",
                sum(p.numel() for p in self.did_projectors.parameters()) / 1e6,
            )
        return params

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if getattr(self, "did_projectors", None) is not None:
            # Train-time only: inference never touches these, but keeping them
            # makes a run resumable and the ablation reproducible.
            payload["did_projectors"] = self.did_projectors.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None, *, strict_shapes: bool = False):
        payload = super().load_checkpoint(path, optimizer=optimizer, strict_shapes=strict_shapes)
        if getattr(self, "did_projectors", None) is not None:
            if "did_projectors" in payload:
                self.did_projectors.load_state_dict(payload["did_projectors"], strict=True)
                logger.info("Loaded DID projector weights.")
            else:
                logger.warning(
                    "Checkpoint has no `did_projectors`; keeping freshly initialized projectors "
                    "(expected when warm-starting DID from a stock DreamFastWAM checkpoint)."
                )
        return payload
