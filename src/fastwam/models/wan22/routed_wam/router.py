"""Action-side routing over the imagination interface.

`DynamicFeatureRouter` implements full/static/dynamic semantic QK gating:
one [B,16] vector from stable control context, shared across layers and noise
steps. The remainder of this introduction describes the legacy router retained
for old configurations (`routing_mode=null`).

The action expert in a DreamFastWAM MoT reads the world branch only through the
per-layer keys/values of the Video and Dream tokens.  `ImaginationRouter` lets
the action expert decide, per layer and per sample, *which* of those Dream keys
it wants to read, instead of attending to all of them with a fixed mask.

Three modes are supported, all sharing one interface so they can be swapped from
config:

``none``
    No routing.  The caller is expected to take the dense fast path.

``threshold``
    The parameter-free policy already shipped in
    ``models/wan22/action_dream_threshold``: score a Dream key by its dense
    attention mass normalised by the uniform attention level, then keep it when
    the score clears ``alpha``.  ``alpha == 1`` therefore means "attended more
    than uniformly".  Reimplemented here (rather than imported) only so that the
    three modes share one return contract; the arithmetic is identical and
    ``tests/test_routed_wam.py`` asserts equality against the original.

``learned``
    A low-rank bilinear score between a summary of the action queries and each
    Dream key, plus a per-layer bias and a per-group bias.  The resulting gate is
    injected into the attention logits additively as ``log(gate)``, which is
    exactly equivalent to scaling the unnormalised attention weight by ``gate``:
    a gate of 1 leaves the dense distribution untouched, and a gate of 0
    reproduces a hard mask.  That equivalence is what makes the dense model a
    strict special case of the routed one.

The router is deliberately **deterministic**.  Mixed attention runs inside
``torch.utils.checkpoint``, which re-executes the wrapped function during
backward; anything stochastic (Gumbel noise, dropout) would draw different
values on the recompute and silently corrupt gradients.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn

from fastwam.utils.logging_config import get_logger


logger = get_logger(__name__)

ROUTER_MODES = ("none", "threshold", "learned")


@dataclass(frozen=True)
class RouterConfig:
    """Configuration for :class:`ImaginationRouter`."""

    mode: str = "none"
    # Semantic QK routing. None selects the legacy router above.
    routing_mode: Optional[str] = None
    router_enabled: bool = True
    router_warmup_steps: int = 0
    router_hidden_dim: int = 128
    router_gate_loss_weight: float = 0.0
    gate_init_probability: float = 0.5  # Deprecated; accepted for old configs. QK gates always start at zero.
    # --- threshold mode ---
    alpha: float = 1.0
    # --- learned mode ---
    rank: int = 64
    temperature: float = 1.0
    bias_init: float = 4.0
    group_granularity: str = "modality_horizon"
    # --- shared ---
    min_keep_tokens: int = 0
    target_keep_ratio: float = 0.25
    lambda_budget: float = 0.0
    gate_threshold: float = 1.0e-3
    hard_prune_at_inference: bool = True
    warmup_ratio: float = 0.1
    log_statistics: bool = True
    debug_force_gate: Optional[float] = None
    layers: Optional[list[int]] = field(default=None)

    @classmethod
    def from_dict(cls, value: Optional[dict[str, Any]]) -> "RouterConfig":
        cfg = cls(**({} if value is None else dict(value)))
        if cfg.routing_mode not in (None, "full", "static", "dynamic"):
            raise ValueError("routing_mode must be full, static, dynamic, or null (legacy).")
        if cfg.routing_mode is not None and cfg.mode != "none":
            raise ValueError("Semantic routing requires router.mode=none; legacy routing cannot also run.")
        if cfg.router_warmup_steps < 0 or cfg.router_hidden_dim <= 0 or cfg.router_gate_loss_weight < 0:
            raise ValueError("Invalid router warmup, hidden dimension, or gate loss weight.")
        if not 0 < cfg.gate_init_probability < 1:
            raise ValueError("gate_init_probability must be strictly between zero and one.")
        if cfg.mode not in ROUTER_MODES:
            raise ValueError(f"router.mode must be one of {ROUTER_MODES}, got {cfg.mode!r}.")
        if cfg.alpha < 0:
            raise ValueError(f"router.alpha must be >= 0, got {cfg.alpha}.")
        if cfg.rank <= 0:
            raise ValueError(f"router.rank must be > 0, got {cfg.rank}.")
        if cfg.temperature <= 0:
            raise ValueError(f"router.temperature must be > 0, got {cfg.temperature}.")
        if not 0.0 <= cfg.target_keep_ratio <= 1.0:
            raise ValueError(
                f"router.target_keep_ratio must be in [0,1], got {cfg.target_keep_ratio}."
            )
        if cfg.lambda_budget < 0:
            raise ValueError(f"router.lambda_budget must be >= 0, got {cfg.lambda_budget}.")
        if not 0.0 <= cfg.warmup_ratio <= 1.0:
            raise ValueError(f"router.warmup_ratio must be in [0,1], got {cfg.warmup_ratio}.")
        if cfg.min_keep_tokens < 0:
            raise ValueError(f"router.min_keep_tokens must be >= 0, got {cfg.min_keep_tokens}.")
        if cfg.group_granularity not in ("none", "modality", "modality_horizon"):
            raise ValueError(
                "router.group_granularity must be one of "
                f"('none', 'modality', 'modality_horizon'), got {cfg.group_granularity!r}."
            )
        if cfg.debug_force_gate is not None and not 0.0 <= float(cfg.debug_force_gate) <= 1.0:
            raise ValueError("router.debug_force_gate must be in [0,1] when set.")
        return cfg

    @property
    def enabled(self) -> bool:
        return self.mode != "none"


def build_group_ids(
    *,
    modalities: list[str],
    num_future_offsets: int,
    tokens_per_modality: dict[str, int],
    granularity: str,
) -> tuple[torch.Tensor, list[str]]:
    """Map each Dream token to an interpretable group id.

    Dream tokens are laid out modality-major and, within a modality, offset-major
    (see ``DreamQueryExpert.pre_dit``), so the grouping is a pure function of the
    layout and needs no runtime information.

    Returns:
        ids: LongTensor [num_dream_tokens] of group indices.
        names: human-readable name per group index, used for logging and figures.
    """
    if granularity == "none":
        names = ["all"]
    elif granularity == "modality":
        names = list(modalities)
    else:
        names = [
            f"{modality}@t{offset_index}"
            for modality in modalities
            for offset_index in range(num_future_offsets)
        ]

    ids: list[int] = []
    for modality_index, modality in enumerate(modalities):
        per_offset = int(tokens_per_modality[modality])
        for offset_index in range(num_future_offsets):
            if granularity == "none":
                group_index = 0
            elif granularity == "modality":
                group_index = modality_index
            else:
                group_index = modality_index * num_future_offsets + offset_index
            ids.extend([group_index] * per_offset)
    return torch.tensor(ids, dtype=torch.long), names


class ImaginationRouter(nn.Module):
    """Per-layer gate over the Action -> Dream attention edges."""

    def __init__(
        self,
        *,
        config: RouterConfig,
        num_layers: int,
        inner_dim: int,
        num_dream_tokens: int,
        group_ids: torch.Tensor,
        group_names: list[str],
    ):
        super().__init__()
        self.config = config
        self.num_layers = int(num_layers)
        self.inner_dim = int(inner_dim)
        self.num_dream_tokens = int(num_dream_tokens)
        self.group_names = list(group_names)

        if group_ids.numel() != self.num_dream_tokens:
            raise ValueError(
                f"group_ids has {group_ids.numel()} entries but there are "
                f"{self.num_dream_tokens} dream tokens."
            )
        self.register_buffer("group_ids", group_ids.clone(), persistent=False)

        self.active_layers = (
            set(range(self.num_layers))
            if config.layers is None
            else {int(x) for x in config.layers}
        )
        unknown = {x for x in self.active_layers if not 0 <= x < self.num_layers}
        if unknown:
            raise ValueError(f"router.layers contains out-of-range entries: {sorted(unknown)}")

        if config.mode == "learned":
            rank = int(config.rank)
            self.query_proj = nn.ModuleList(
                [nn.Linear(self.inner_dim, rank, bias=False) for _ in range(self.num_layers)]
            )
            self.key_proj = nn.ModuleList(
                [nn.Linear(self.inner_dim, rank, bias=False) for _ in range(self.num_layers)]
            )
            self.layer_bias = nn.Parameter(torch.full((self.num_layers,), float(config.bias_init)))
            self.group_bias = nn.Parameter(torch.zeros(self.num_layers, len(self.group_names)))
            # Small init keeps the bilinear term near zero, so the gate starts at
            # sigmoid(bias_init) ~ 1 and the routed model begins life as the dense
            # model.  Fine-tuning then learns what to switch off.
            for module in list(self.query_proj) + list(self.key_proj):
                nn.init.normal_(module.weight, std=1.0e-3)
        else:
            self.query_proj = None
            self.key_proj = None
            self.layer_bias = None
            self.group_bias = None

        self._progress: float = 1.0
        self.last_statistics: list[dict[str, Any]] = []
        logger.info(
            "ImaginationRouter(mode=%s) over %d dream tokens, %d groups, layers=%s",
            config.mode,
            self.num_dream_tokens,
            len(self.group_names),
            "all" if config.layers is None else sorted(self.active_layers),
        )

    # ------------------------------------------------------------------ utils
    def set_progress(self, fraction: float) -> None:
        """Warmup fraction in [0, 1]; 0 means "behave densely"."""
        self._progress = float(min(max(fraction, 0.0), 1.0))

    def current_strength(self) -> float:
        """How strongly routing is applied right now.

        Warmup only applies while training: evaluation always uses the full
        policy, otherwise a checkpoint would evaluate differently depending on
        how far through training it was saved.
        """
        if not self.training or self.config.warmup_ratio <= 0.0:
            return 1.0
        return self._progress

    def reset_statistics(self) -> None:
        self.last_statistics = []

    # ------------------------------------------------------------------ gates
    def _learned_gate(
        self,
        *,
        layer_idx: int,
        q_action: torch.Tensor,
        k_dream: torch.Tensor,
    ) -> torch.Tensor:
        """Return a gate in (0, 1] of shape [B, Sd]."""
        rank = int(self.config.rank)
        # Mean over action queries: the routing decision is per (sample, layer),
        # not per action step -- the whole chunk is denoised jointly and a
        # per-step decision cannot be realised by a single key mask anyway.
        query_summary = q_action.mean(dim=1)
        q = self.query_proj[layer_idx](query_summary.to(self.query_proj[layer_idx].weight.dtype))
        k = self.key_proj[layer_idx](k_dream.to(self.key_proj[layer_idx].weight.dtype))
        logits = torch.einsum("br,bsr->bs", q, k) * (rank ** -0.5)
        logits = logits + self.layer_bias[layer_idx]
        group_bias = self.group_bias[layer_idx].index_select(
            0, self.group_ids.to(device=logits.device)
        )
        logits = logits + group_bias.unsqueeze(0)
        return torch.sigmoid(logits / float(self.config.temperature))

    @staticmethod
    def _uniform_normalised_dream_mass(
        *,
        dense_probs: torch.Tensor,
        dream_slice: slice,
        n_valid: torch.Tensor,
    ) -> torch.Tensor:
        """The shipped threshold score: mean over heads, max over action queries.

        ``dense_probs`` is [B, H, Sa, S]; multiplying by the number of valid keys
        turns "probability" into "multiples of the uniform attention level", so
        the threshold is comparable across layers and sequence lengths.
        """
        dense_dream = dense_probs[..., dream_slice]
        normalised = dense_dream * n_valid[:, None, :, None]
        return normalised.mean(dim=1).amax(dim=1)

    def _apply_min_keep(self, keep: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        minimum = min(int(self.config.min_keep_tokens), int(score.shape[-1]))
        if minimum <= 0:
            return keep
        needed = (minimum - keep.sum(dim=-1)).clamp(min=0)
        if not bool((needed > 0).any()):
            return keep
        order = torch.argsort(score, dim=-1, descending=True, stable=True)
        ranks = torch.arange(score.shape[-1], device=score.device).view(1, -1)
        additions_in_rank_order = ranks < needed.unsqueeze(-1)
        additions = torch.zeros_like(keep).scatter(1, order, additions_in_rank_order)
        return keep | additions

    # ---------------------------------------------------------------- forward
    def gate_for_layer(
        self,
        *,
        layer_idx: int,
        q_action: torch.Tensor,
        k_dream: torch.Tensor,
        dense_probs: Optional[torch.Tensor] = None,
        dream_slice: Optional[slice] = None,
        n_valid: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the routing gate for one layer.

        Returns a dict with:
            gate: [B, Sd] multiplicative gate in [0, 1] (differentiable in
                ``learned`` mode, a hard 0/1 tensor in ``threshold`` mode).
            keep: [B, Sd] boolean mask used for hard pruning / statistics.
            score: [B, Sd] the quantity the decision was based on.
        """
        batch_size = int(q_action.shape[0])
        num_dream = int(k_dream.shape[1])
        device = q_action.device

        if layer_idx not in self.active_layers or self.config.mode == "none":
            gate = torch.ones((batch_size, num_dream), device=device, dtype=torch.float32)
            return {"gate": gate, "keep": gate.bool(), "score": gate}

        if self.config.debug_force_gate is not None:
            value = float(self.config.debug_force_gate)
            gate = torch.full((batch_size, num_dream), value, device=device, dtype=torch.float32)
            return {"gate": gate, "keep": gate > self.config.gate_threshold, "score": gate}

        if self.config.mode == "threshold":
            if dense_probs is None or dream_slice is None or n_valid is None:
                raise ValueError(
                    "router.mode='threshold' needs dense_probs, dream_slice and n_valid."
                )
            score = self._uniform_normalised_dream_mass(
                dense_probs=dense_probs, dream_slice=dream_slice, n_valid=n_valid
            ).detach()
            keep = score >= float(self.config.alpha)
            keep = self._apply_min_keep(keep, score)
            gate = keep.to(dtype=torch.float32)
        else:
            score = self._learned_gate(
                layer_idx=layer_idx, q_action=q_action, k_dream=k_dream
            ).float()
            strength = self.current_strength()
            if strength < 1.0:
                # Blend towards the dense gate of 1 during warmup so the model
                # does not have to survive a discontinuity at step 0.
                score = score * strength + (1.0 - strength)
            keep = score > float(self.config.gate_threshold)
            keep = self._apply_min_keep(keep, score.detach())
            gate = score
            if self.config.hard_prune_at_inference and not self.training:
                gate = gate * keep.to(dtype=gate.dtype)

        return {"gate": gate, "keep": keep, "score": score}

    def budget_loss(self, gates: list[torch.Tensor]) -> torch.Tensor:
        """Push the average gate towards the configured keep ratio."""
        if not gates or self.config.lambda_budget <= 0.0 or self.config.mode != "learned":
            device = gates[0].device if gates else torch.device("cpu")
            return torch.zeros((), device=device)
        mean_gate = torch.stack([g.mean() for g in gates]).mean()
        target = torch.as_tensor(
            float(self.config.target_keep_ratio), device=mean_gate.device, dtype=mean_gate.dtype
        )
        return float(self.config.lambda_budget) * (mean_gate - target).pow(2)

    def record(
        self,
        *,
        layer_idx: int,
        gate: torch.Tensor,
        keep: torch.Tensor,
        score: torch.Tensor,
    ) -> None:
        if not self.config.log_statistics:
            return
        with torch.no_grad():
            group_ids = self.group_ids.to(device=keep.device)
            per_group = {}
            for index, name in enumerate(self.group_names):
                selector = group_ids == index
                if not bool(selector.any()):
                    continue
                per_group[name] = float(keep[:, selector].float().mean().item())
            self.last_statistics.append(
                {
                    "layer": int(layer_idx),
                    "keep_ratio": float(keep.float().mean().item()),
                    "gate_mean": float(gate.float().mean().item()),
                    "score_mean": float(score.float().mean().item()),
                    "k_mean": float(keep.sum(dim=-1).float().mean().item()),
                    "per_group_keep_ratio": per_group,
                }
            )

    def scalar_metrics(self) -> dict[str, float]:
        if not self.last_statistics:
            return {}
        count = float(len(self.last_statistics))
        metrics = {
            "router_keep_ratio": sum(r["keep_ratio"] for r in self.last_statistics) / count,
            "router_gate_mean": sum(r["gate_mean"] for r in self.last_statistics) / count,
            "router_k_mean": sum(r["k_mean"] for r in self.last_statistics) / count,
        }
        for name in self.group_names:
            values = [
                r["per_group_keep_ratio"][name]
                for r in self.last_statistics
                if name in r["per_group_keep_ratio"]
            ]
            if values:
                metrics[f"router_keep_{name}"] = sum(values) / len(values)
        return metrics


class DynamicFeatureRouter(nn.Module):
    """One semantic gate per modality/view/horizon, shared by the entire action solve.

    Legacy ImaginationRouter remains available for old experiments. This module
    never reads action noise, prunes tokens, or normalizes across groups.
    """

    modality_order = ("dino", "dyn", "sam", "depth")
    activation_names = ("dino_activation", "tracker_activation", "sam_activation", "depth_activation")

    def __init__(self, *, config: RouterConfig, input_dim: int, dream_expert):
        super().__init__()
        self.config = config
        if config.routing_mode is None:
            raise ValueError("DynamicFeatureRouter requires an explicit routing_mode.")
        if set(dream_expert.modalities) != set(self.modality_order):
            raise ValueError("16-group routing requires dyn, depth, dino, sam.")
        split = dream_expert.camera_token_split
        if split is None or len(split) != 2 or len(set(dream_expert.future_offsets)) != 2:
            raise ValueError("16-group routing requires two camera groups and two future offsets.")
        self.group_mapping = []
        token_groups = []
        for modality in dream_expert.modalities:
            if sum(split) != getattr(dream_expert, f"n_{modality}"):
                raise ValueError("Camera token split does not match the Dream layout.")
            for offset in dream_expert.future_offsets:
                for view, count in zip(("primary", "wrist"), split):
                    group_id = len(self.group_mapping)
                    start = len(token_groups)
                    token_groups.extend([group_id] * count)
                    self.group_mapping.append(dict(
                        group_id=group_id, modality=modality, view=view,
                        future_offset=int(offset), token_start=start, token_stop=len(token_groups),
                    ))
        if len(self.group_mapping) != 16:
            raise ValueError("Expected exactly 16 semantic groups.")
        self.register_buffer("group_ids", torch.tensor(token_groups, dtype=torch.long), persistent=False)
        self.register_buffer("modality_indices", torch.tensor([
            [g["group_id"] for g in self.group_mapping if g["modality"] == m]
            for m in self.modality_order
        ]), persistent=False)
        self.gate_prior = nn.Parameter(torch.zeros(16))
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, config.router_hidden_dim),
            nn.GELU(), nn.Linear(config.router_hidden_dim, 16),
        ) if config.routing_mode == "dynamic" else None
        if self.network is not None:
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)
        self.global_step = 0
        if self.full:
            self.requires_grad_(False)

    def _apply(self, fn, recurse=True):
        # DeepSpeed calls model.bfloat16() before building ZeRO partitions.
        # Keep the small semantic router in FP32, including through parent .to().
        # Use the original tensor when converting back to avoid BF16 rounding.
        def preserve_precision(tensor):
            converted = fn(tensor)
            if tensor.is_floating_point():
                return tensor.to(device=converted.device, dtype=torch.float32)
            return converted
        return super()._apply(preserve_precision, recurse=recurse)

    @torch.no_grad()
    def reset_zero(self):
        """Fresh stage-two router: every input starts with zero Dream QK scale."""
        self.gate_prior.zero_()
        if self.network is not None:
            for module in self.network:
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)

    @property
    def full(self):
        return not self.config.router_enabled or self.config.routing_mode == "full"

    def warming_up(self, training: bool) -> bool:
        return training and self.global_step < self.config.router_warmup_steps

    def forward(self, context: torch.Tensor, *, training: bool = False) -> torch.Tensor:
        if context.ndim != 2:
            raise ValueError("Router context must be [B,D].")
        if self.full:
            return context.new_ones((context.shape[0], 16), dtype=torch.float32)
        # A zero dependency keeps DDP's parameter participation stable across
        # warmup, without allowing either action or gate loss to train gates yet.
        logits = self.gate_prior.expand(context.shape[0], -1)
        if self.network is not None:
            with torch.autocast(device_type=context.device.type, enabled=False):
                logits = logits + self.network(context.float())
        if self.warming_up(training):
            return logits.float() * 0.0
        return torch.tanh(logits.float())

    def gate_loss(self, gates: torch.Tensor, *, training: bool) -> torch.Tensor:
        if self.full or self.warming_up(training):
            return gates.sum() * 0.0
        return gates.abs().mean()

    def activations(self, gates: torch.Tensor) -> torch.Tensor:
        if gates.ndim != 2 or gates.shape[1] != 16:
            raise ValueError("Group gates must be [B,16].")
        return gates[:, self.modality_indices].mean(dim=-1)

    def gate_cache(self, cache: dict, gates: torch.Tensor) -> dict:
        """Attach QK scales for Action; preserve all raw K/V tensors."""
        activations = self.activations(gates)  # also validate the public gate shape
        nv, nd = cache["video_seq_len"], cache["dream_seq_len"]
        if nd != self.group_ids.numel():
            raise ValueError("Dream cache token count does not match the group mapping.")
        weights = gates[:, self.group_ids]
        bypass = self.full and bool(torch.all(gates == 1))
        layers = []
        for layer in cache["kv_cache"]:
            value = layer["v"]
            if value.shape[1] != nv + nd or value.shape[0] != gates.shape[0]:
                raise ValueError("Action context cache shape does not match gates/layout.")
            layers.append(dict(layer) if bypass else {**layer, "dream_token_gates": weights})
        return {**cache, "kv_cache": layers, "group_gates": gates,
                "modality_activations": activations,
                "group_mapping": self.group_mapping}
