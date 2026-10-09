"""Generative multi-modal Dream expert.

The shipped :class:`DreamQueryExpert` is a *regressor*: a fixed set of learnable
queries is decoded straight to the future depth / DINO / SAM / dynamics targets,
so the prediction is the conditional mean and there is nothing to iterate on.

``GenerativeDreamExpert`` turns the same module into a *denoiser* over the same
targets.  The Dream tokens become "identity query + encoded noisy target", the
expert is conditioned on a Dream diffusion timestep, and the decoders now emit a
flow-matching velocity instead of a point estimate.  Two things follow:

* the action expert can be offered several denoising states of the future, which
  is what makes an action-side router meaningful rather than a fixed re-weighting
  of a single deterministic prediction;
* there is a multi-step computation to compress, which is what interface
  distillation compresses.

Setting ``generative.enabled=false`` restores the parent behaviour exactly: no
extra tensors are added to the forward and the extra parameters stay unused, so
a regression-trained checkpoint keeps loading and keeps producing bit-identical
outputs.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

from ..dream_fastwam.dream_query_expert import DenseDreamDecoder, DreamQueryExpert
from ..wan_video_dit import sinusoidal_embedding_1d


logger = get_logger(__name__)


class DreamTargetEncoder(nn.Module):
    """Encode a modality target into one latent vector per Dream token.

    The layout mirrors :meth:`DenseDreamDecoder.forward_two_view` exactly, so the
    encoder pools the same tokens that the decoder will later be asked to
    reconstruct: with ``camera_token_split=[9, 9]`` the left half of the target
    grid is pooled into the first nine Dream tokens and the right half into the
    last nine.  Getting this wrong would silently hand the wrist camera's
    evidence to the primary camera's queries.
    """

    def __init__(
        self,
        *,
        modality: str,
        decoder: DenseDreamDecoder,
        hidden_dim: int,
        num_tokens: int,
        camera_token_split: Optional[tuple[int, int]],
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.modality = modality
        self.hidden_dim = int(hidden_dim)
        self.num_tokens = int(num_tokens)
        self.camera_token_split = camera_token_split
        self.target_layout = decoder.target_layout
        self.feature_dim = int(decoder.feature_dim)
        self.output_grid_shape = decoder.output_grid_shape
        self.num_output_tokens = int(decoder.num_output_tokens)

        self.proj = nn.Linear(self.feature_dim, self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)
        # `init_scale=0` makes conversion from a pretrained regression model a
        # no-op, but from scratch it means the Dream tokens carry no information
        # about the noise realisation at all. Measured: after 416 steps
        # out_scale had only reached
        # 0.012 (depth) / 0.018 (dino), and loss_dream never left the
        # conditional-mean baseline. Default to 1 and set 0 only when resuming.
        self.out_scale = nn.Parameter(torch.full((1,), float(init_scale)))
        self._decoder_ref = [decoder]

    def _to_token_layout(self, target: torch.Tensor) -> torch.Tensor:
        """[B*O, ...] in decoder output layout -> [B*O, num_output_tokens, F]."""
        if self.target_layout == "grid_feature":
            if target.ndim != 4:
                raise ValueError(
                    f"{self.modality} grid_feature target must be [N,H,W,F], got {tuple(target.shape)}."
                )
            return target.reshape(target.shape[0], -1, target.shape[-1])
        if target.ndim != 3:
            raise ValueError(
                f"{self.modality} token_feature target must be [N,T,F], got {tuple(target.shape)}."
            )
        return target

    @staticmethod
    def _pool(tokens: torch.Tensor, num_out: int) -> torch.Tensor:
        """Average-pool along the token axis to exactly `num_out` tokens."""
        if tokens.shape[1] == num_out:
            return tokens
        pooled = F.adaptive_avg_pool1d(tokens.transpose(1, 2), num_out)
        return pooled.transpose(1, 2)

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        tokens = self._to_token_layout(target)
        if tokens.shape[-1] != self.feature_dim:
            raise ValueError(
                f"{self.modality} target feature dim {tokens.shape[-1]} != {self.feature_dim}."
            )
        if self.camera_token_split is None:
            pooled = self._pool(tokens, self.num_tokens)
        else:
            primary_tokens, wrist_tokens = self.camera_token_split
            decoder = self._decoder_ref[0]
            primary_idx, wrist_idx = decoder._two_view_query_indices(device=tokens.device)
            pooled = torch.cat(
                [
                    self._pool(tokens.index_select(1, primary_idx), primary_tokens),
                    self._pool(tokens.index_select(1, wrist_idx), wrist_tokens),
                ],
                dim=1,
            )
        latent = self.norm(self.proj(pooled.to(self.proj.weight.dtype)))
        return latent * self.out_scale


class GenerativeDreamExpert(DreamQueryExpert):
    """DreamQueryExpert that denoises its targets instead of regressing them."""

    def __init__(self, *args, generative: Optional[dict[str, Any]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_generative(generative)

    @classmethod
    def promote(
        cls, expert: DreamQueryExpert, generative: Optional[dict[str, Any]] = None
    ) -> "GenerativeDreamExpert":
        """Convert an existing DreamQueryExpert in place.

        The dense factory (`fastwam.runtime.create_dream_fastwam`) builds and
        validates the Dream expert; re-implementing that here would duplicate its
        config handling, so the object is promoted instead -- the same technique
        `ThresholdDreamFastWAM.from_dense_model` uses on the model itself.
        """
        if isinstance(expert, cls):
            return expert
        if not isinstance(expert, DreamQueryExpert):
            raise TypeError(f"Expected DreamQueryExpert, got {type(expert)}.")
        expert.__class__ = cls
        expert._setup_generative(generative)
        return expert

    def _setup_generative(self, generative: Optional[dict[str, Any]]) -> None:
        generative = dict(generative or {})
        self.generative_enabled = bool(generative.get("enabled", False))
        self.freq_dim_dream = int(generative.get("freq_dim", self.freq_dim))

        self.target_encoders = nn.ModuleDict()
        if self.generative_enabled:
            for modality in self.modalities:
                decoder = self.decoders[modality]
                if not bool(getattr(decoder, "enabled", True)):
                    continue
                self.target_encoders[modality] = DreamTargetEncoder(
                    modality=modality,
                    decoder=decoder,
                    hidden_dim=self.hidden_dim,
                    num_tokens=int(getattr(self, f"n_{modality}")),
                    camera_token_split=self.camera_token_split,
                    init_scale=float(generative.get("target_encoder_init_scale", 1.0)),
                )
            self.dream_time_embedding = nn.Sequential(
                nn.Linear(self.freq_dim_dream, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.dream_time_projection = nn.Sequential(
                nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim * 6)
            )
            # Zero-init so the timestep modulation starts as a no-op and the
            # module is numerically identical to the parent at conversion time.
            nn.init.zeros_(self.dream_time_projection[1].weight)
            nn.init.zeros_(self.dream_time_projection[1].bias)
            logger.info(
                "GenerativeDreamExpert enabled over modalities %s (%d dream tokens).",
                list(self.target_encoders.keys()),
                self.num_dream_tokens,
            )

    # ------------------------------------------------------------------ utils
    def encode_targets(self, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode per-modality targets into the Dream token latent layout.

        Args:
            targets: modality -> tensor shaped ``[B, O, ...]`` in decoder layout.

        Returns:
            ``[B, num_dream_tokens, hidden_dim]``, modality-major then offset-major,
            matching :meth:`DreamQueryExpert.modality_slices`.
        """
        if not self.generative_enabled:
            raise RuntimeError("encode_targets requires generative.enabled=true.")
        slices = self.modality_slices()
        batch_size = None
        chunks: list[tuple[slice, torch.Tensor]] = []
        for modality in self.modalities:
            if modality not in self.target_encoders:
                continue
            if modality not in targets:
                raise ValueError(f"Missing dream target for modality {modality!r}.")
            target = targets[modality]
            if target.ndim < 3:
                raise ValueError(
                    f"{modality} target must be [B,O,...], got {tuple(target.shape)}."
                )
            bsz, num_offsets = int(target.shape[0]), int(target.shape[1])
            if batch_size is None:
                batch_size = bsz
            elif batch_size != bsz:
                raise ValueError("Dream targets disagree on batch size.")
            if num_offsets != self.num_future_offsets:
                raise ValueError(
                    f"{modality} target has {num_offsets} offsets, "
                    f"expected {self.num_future_offsets}."
                )
            flat = target.reshape(bsz * num_offsets, *target.shape[2:])
            latent = self.target_encoders[modality](flat)
            per_offset = int(getattr(self, f"n_{modality}"))
            latent = latent.reshape(bsz, num_offsets * per_offset, self.hidden_dim)
            chunks.append((slices[modality], latent))

        if batch_size is None:
            raise ValueError("No dream targets were encoded.")
        out = torch.zeros(
            (batch_size, self.num_dream_tokens, self.hidden_dim),
            device=chunks[0][1].device,
            dtype=chunks[0][1].dtype,
        )
        for slc, latent in chunks:
            out[:, slc, :] = latent
        return out

    def split_prediction(self, flat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Inverse of :meth:`encode_targets` bookkeeping, for schedulers.

        Not used by the forward pass; provided so callers can move between the
        per-modality dict and a flat tensor without re-deriving the layout.
        """
        slices = self.modality_slices()
        return {name: flat[:, slc, :] for name, slc in slices.items() if name in self.modalities}

    # --------------------------------------------------------------- pre_dit
    def pre_dit(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        noisy_latent: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
    ) -> Dict[str, Any]:
        pre_state = super().pre_dit(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            context=context,
            context_mask=context_mask,
        )
        if not self.generative_enabled:
            if noisy_latent is not None or timestep is not None:
                raise ValueError(
                    "noisy_latent/timestep were given but generative.enabled=false."
                )
            return pre_state

        if noisy_latent is not None:
            if tuple(noisy_latent.shape) != (
                batch_size,
                self.num_dream_tokens,
                self.hidden_dim,
            ):
                raise ValueError(
                    "noisy_latent must be [B,num_dream_tokens,hidden_dim], got "
                    f"{tuple(noisy_latent.shape)}."
                )
            pre_state["tokens"] = pre_state["tokens"] + noisy_latent.to(
                device=pre_state["tokens"].device, dtype=pre_state["tokens"].dtype
            )

        if timestep is not None:
            if timestep.ndim != 1:
                raise ValueError(f"`timestep` must be 1D [B], got {tuple(timestep.shape)}.")
            if timestep.shape[0] == 1 and batch_size > 1:
                timestep = timestep.expand(batch_size)
            embed = self.dream_time_embedding(
                sinusoidal_embedding_1d(self.freq_dim_dream, timestep).to(
                    dtype=pre_state["tokens"].dtype
                )
            )
            delta = self.dream_time_projection(embed).unflatten(1, (6, self.hidden_dim))
            pre_state["t_mod"] = pre_state["t_mod"] + delta
        pre_state["meta"]["generative"] = True
        return pre_state
