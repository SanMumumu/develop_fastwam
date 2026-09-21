"""RoutedWAM: a DreamFastWAM whose imagination is generated, routed and distilled.

Three changes on top of :class:`DreamFastWAM`, each independently switchable so
that ablations are config-only:

``generative``
    The Dream expert denoises its multi-modal future targets instead of
    regressing them (:mod:`generative_dream`).

``router``
    The action expert gates which Dream keys it reads, per layer and per sample
    (:mod:`router`).

``interface_distill``
    A one-step Dream pass is trained to reproduce the per-layer Dream K/V of an
    N-step EMA teacher, so the multi-step imagination collapses into a single
    forward whose output is cached and reused across every action denoising step
    (:mod:`interface_distill`).

With all three disabled the class is behaviourally identical to its parent, and
``tests/test_routed_wam.py`` asserts that element-wise.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

from ..dream_fastwam.model import DreamFastWAM
from ..schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .generative_dream import GenerativeDreamExpert
from .interface_distill import InterfaceDistillConfig, InterfaceDistiller
from .mot import RoutedMoT
from .router import ImaginationRouter, RouterConfig, build_group_ids


logger = get_logger(__name__)

# State-dict prefixes introduced by this module. They are legitimately absent
# from any DreamFastWAM checkpoint, so `load_checkpoint` must not treat them as
# a structural incompatibility.
NEW_PARAMETER_PREFIXES = (
    "router.",
    "mixtures.dream.target_encoders.",
    "mixtures.dream.dream_time_embedding.",
    "mixtures.dream.dream_time_projection.",
)


def _is_new_parameter(key: str) -> bool:
    return any(key.startswith(prefix) or f".{prefix}" in key for prefix in NEW_PARAMETER_PREFIXES)


class RoutedWAM(DreamFastWAM):
    """DreamFastWAM with a generative, routed and distillable imagination."""

    # ------------------------------------------------------------ construction
    @classmethod
    def from_wan22_pretrained(
        cls,
        *args,
        router: Optional[dict[str, Any]] = None,
        interface_distill: Optional[dict[str, Any]] = None,
        generative_dream: Optional[dict[str, Any]] = None,
        dream_scheduler: Optional[dict[str, Any]] = None,
        finetune_action_only: bool = False,
        **kwargs,
    ) -> "RoutedWAM":
        model = DreamFastWAM.from_wan22_pretrained(*args, **kwargs)
        return cls.from_dense_model(
            model,
            router=router,
            interface_distill=interface_distill,
            generative_dream=generative_dream,
            dream_scheduler=dream_scheduler,
            finetune_action_only=finetune_action_only,
        )

    @classmethod
    def from_dense_model(
        cls,
        model: DreamFastWAM,
        *,
        router: Optional[dict[str, Any]] = None,
        interface_distill: Optional[dict[str, Any]] = None,
        generative_dream: Optional[dict[str, Any]] = None,
        dream_scheduler: Optional[dict[str, Any]] = None,
        finetune_action_only: bool = False,
    ) -> "RoutedWAM":
        if not isinstance(model, DreamFastWAM):
            raise TypeError(f"Expected DreamFastWAM, got {type(model)}.")

        router_config = RouterConfig.from_dict(router)
        distill_config = InterfaceDistillConfig.from_dict(interface_distill)
        generative_config = dict(generative_dream or {})

        model.__class__ = cls
        model.router_config = router_config
        model.distill_config = distill_config
        model.finetune_action_only = bool(finetune_action_only)
        model._training_progress_provider = None
        model._last_ema_step = -1

        # 1. Promote the Dream expert in place so the dense factory is reused.
        GenerativeDreamExpert.promote(model.dream_expert, generative_config)
        model.dream_expert.to(device=model.device, dtype=model.torch_dtype)

        # 2. Build the router over the Dream token layout.
        dense_mot = model.mot
        dream_expert = model.dream_expert
        group_ids, group_names = build_group_ids(
            modalities=list(dream_expert.modalities),
            num_future_offsets=int(dream_expert.num_future_offsets),
            tokens_per_modality={
                name: int(getattr(dream_expert, f"n_{name}")) for name in dream_expert.modalities
            },
            granularity=router_config.group_granularity,
        )
        imagination_router = (
            ImaginationRouter(
                config=router_config,
                num_layers=dense_mot.num_layers,
                inner_dim=dense_mot.num_heads * dense_mot.attn_head_dim,
                num_dream_tokens=int(dream_expert.num_dream_tokens),
                group_ids=group_ids,
                group_names=group_names,
            )
            if router_config.enabled
            else None
        )

        # 3. Swap in the routed MoT, keeping the very same expert modules.
        model.mot = RoutedMoT(
            mixtures={name: dense_mot.mixtures[name] for name in dense_mot.expert_order},
            mot_checkpoint_mixed_attn=dense_mot.mot_checkpoint_mixed_attn,
            router=imagination_router,
        )
        model.dit = model.mot
        if imagination_router is not None:
            model.mot.router.to(device=model.device, dtype=model.torch_dtype)

        # 4. Dream diffusion scheduler (only consulted in generative mode).
        scheduler_kwargs = dict(dream_scheduler or {})
        model.train_dream_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(scheduler_kwargs.get("num_train_timesteps", 1000)),
            shift=float(scheduler_kwargs.get("train_shift", 5.0)),
        )
        model.infer_dream_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(scheduler_kwargs.get("num_train_timesteps", 1000)),
            shift=float(scheduler_kwargs.get("infer_shift", 5.0)),
        )
        model.dream_inference_steps = int(scheduler_kwargs.get("inference_steps", 1))
        if model.dream_inference_steps < 1:
            raise ValueError("dream_scheduler.inference_steps must be >= 1.")

        # 5. Interface distillation teacher.
        model.distiller = (
            InterfaceDistiller(
                config=distill_config,
                dream_expert=model.dream_expert,
                num_layers=model.mot.num_layers,
            )
            if distill_config.enabled
            else None
        )
        if model.distiller is not None:
            model.distiller.to(device=model.device, dtype=model.torch_dtype)
            if not model.dream_expert.generative_enabled:
                raise ValueError(
                    "interface_distill.enabled=true requires generative_dream.enabled=true: "
                    "with a regression Dream there is only one step and nothing to distil."
                )

        logger.info(
            "Installed RoutedWAM: router=%s generative_dream=%s interface_distill=%s "
            "dream_inference_steps=%d action_only=%s",
            router_config.mode,
            model.dream_expert.generative_enabled,
            distill_config.enabled,
            model.dream_inference_steps,
            model.finetune_action_only,
        )
        return model

    # -------------------------------------------------------------- bookkeeping
    @property
    def uses_split_path(self) -> bool:
        """Whether training should run Video-once / Dream-per-step / Action-cached.

        This is the deployment computation, so training through it removes a
        train/test mismatch; it is mandatory for interface distillation, which
        needs the per-layer Dream K/V that only this path materialises.
        """
        return bool(self.distill_config.enabled) or bool(self.dream_expert.generative_enabled)

    def set_training_progress_provider(self, provider) -> None:
        self._training_progress_provider = provider

    def _refresh_progress(self) -> None:
        if self._training_progress_provider is None:
            return
        progress = tuple(self._training_progress_provider())
        if len(progress) == 2:
            global_step, total_steps = progress
        elif len(progress) == 3:
            global_step, total_steps, _ = progress
        else:
            raise ValueError("Training progress provider must return 2 or 3 values.")
        total_steps = max(int(total_steps), 0)
        global_step = max(int(global_step), 0)

        if self.mot.router is not None:
            warmup = self.router_config.warmup_ratio
            if warmup <= 0.0 or total_steps <= 0:
                self.mot.router.set_progress(1.0)
            else:
                warmup_steps = max(int(round(total_steps * warmup)), 1)
                self.mot.router.set_progress(min(global_step / warmup_steps, 1.0))
        if self.distiller is not None:
            warmup = self.distill_config.warmup_ratio
            if warmup <= 0.0 or total_steps <= 0:
                self.distiller.set_progress(1.0)
            else:
                warmup_steps = max(int(round(total_steps * warmup)), 1)
                self.distiller.set_progress(min(global_step / warmup_steps, 1.0))
            # Advance the EMA teacher once per optimizer step. `training_loss`
            # is called once per micro-batch, so keying on the global step keeps
            # the decay schedule independent of gradient accumulation -- and it
            # avoids having to override the trainer's inner loop.
            if self.training and global_step != self._last_ema_step:
                self._last_ema_step = global_step
                self.distiller.update_ema(self.dream_expert)

    def configure_trainable_parameters(self, freeze_video_expert: bool = False):
        if self.finetune_action_only:
            self.eval()
            self.requires_grad_(False)
            self.mot.train()
            self.action_expert.train()
            self.action_expert.requires_grad_(True)
            if self.mot.router is not None:
                self.mot.router.train()
                self.mot.router.requires_grad_(True)
            params = [p for p in self.parameters() if p.requires_grad]
            logger.info(
                "RoutedWAM action-only fine-tuning: %.3fM trainable parameters.",
                sum(p.numel() for p in params) / 1e6,
            )
            return params

        params = super().configure_trainable_parameters(freeze_video_expert=freeze_video_expert)
        if self.mot.router is not None:
            # The parent freezes everything and then re-enables the experts it
            # knows about; the router is new, so it must be re-enabled here or it
            # would silently never receive gradients.
            self.mot.router.train()
            self.mot.router.requires_grad_(True)
            params = [p for p in self.parameters() if p.requires_grad]
        if self.distiller is not None:
            # The EMA teacher is never optimised; it follows the student.
            self.distiller.teacher_dream.requires_grad_(False)
            self.distiller.teacher_dream.eval()
            params = [p for p in params if p.requires_grad]
        if self.distill_config.enabled and not freeze_video_expert:
            raise ValueError(
                "interface_distill requires freeze_video_expert=true. The split "
                "Video-once prefill is only valid while the Video expert's K/V are "
                "independent of the diffusion step, which a trainable Video expert "
                "would break."
            )
        logger.info(
            "RoutedWAM trainable parameters: %.3fM (router=%s, distill=%s).",
            sum(p.numel() for p in params) / 1e6,
            self.router_config.mode,
            self.distill_config.enabled,
        )
        return params

    def load_checkpoint(self, path, optimizer=None, *, strict_shapes: bool = False):
        """Load a DreamFastWAM checkpoint, tolerating this module's new keys.

        The parent raises under ``strict_shapes`` on *any* missing key, and
        LIBERO evaluation passes ``strict_shapes=True``.  Router and generative
        Dream parameters are legitimately absent from a dense checkpoint, so the
        strictness check is re-run here with those keys excluded -- everything
        pretrained stays strict.
        """
        if strict_shapes:
            report = self.verify_checkpoint_compatibility(path)
            if report["missing_new_parameters"]:
                logger.info(
                    "Checkpoint predates RoutedWAM; %d new parameters keep their "
                    "initialisation. First keys: %s",
                    len(report["missing_new_parameters"]),
                    report["missing_new_parameters"][:20],
                )
        return super().load_checkpoint(path, optimizer=optimizer, strict_shapes=False)

    def verify_checkpoint_compatibility(self, path) -> dict[str, list[str]]:
        """Report which keys a checkpoint is missing, split into old and new.

        Used by evaluation in place of the parent's ``strict_shapes`` flag: new
        parameters may be missing, pretrained ones may not.
        """
        payload = torch.load(path, map_location="cpu")
        if "mot" not in payload:
            raise ValueError(f"Checkpoint has no `mot` state: {path}")
        current = self.mot.state_dict()
        missing = [k for k in current if k not in payload["mot"]]
        unexpected = [k for k in payload["mot"] if k not in current]
        shape_mismatch = [
            k
            for k, v in payload["mot"].items()
            if k in current and tuple(current[k].shape) != tuple(v.shape)
        ]
        pretrained_missing = [k for k in missing if not _is_new_parameter(k)]
        if pretrained_missing or unexpected or shape_mismatch:
            raise RuntimeError(
                "Checkpoint is not compatible with this RoutedWAM. "
                f"missing_pretrained={pretrained_missing[:20]} "
                f"unexpected={unexpected[:20]} shape_mismatch={shape_mismatch[:20]}"
            )
        return {
            "missing_new_parameters": [k for k in missing if _is_new_parameter(k)],
            "unexpected": unexpected,
        }

    # --------------------------------------------------------------- dream I/O
    def _dream_target_reference(self, targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        expected = set(self.dream_expert.modalities)
        got = set(targets.keys())
        if not expected.issubset(got):
            raise ValueError(
                f"Dream targets are missing modalities {sorted(expected - got)}; got {sorted(got)}."
            )
        return {name: targets[name] for name in self.dream_expert.modalities}

    def _sample_dream_noise(self, targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: torch.randn_like(value.float()).to(value.dtype) for name, value in targets.items()}

    def _dream_noise_like_targets(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
    ) -> dict[str, torch.Tensor]:
        """Pure noise in target space, for inference where no target exists."""
        noise: dict[str, torch.Tensor] = {}
        offsets = int(self.dream_expert.num_future_offsets)
        for name in self.dream_expert.modalities:
            decoder = self.dream_expert.decoders[name]
            if not bool(getattr(decoder, "enabled", True)):
                continue
            shape = (batch_size, offsets, *decoder.target_shape)
            noise[name] = torch.randn(shape, device=device, dtype=torch.float32, generator=generator).to(dtype)
        return noise

    def _dream_step(
        self,
        *,
        noisy_targets: dict[str, torch.Tensor],
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        context_attention_mask: torch.Tensor,
        video_seq_len: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        """One Dream denoising step against a cached Video K/V.

        Returns the predicted per-modality velocity, the advanced Dream tokens and
        the per-layer Dream K/V -- the interface the action expert will read.
        """
        latent = self.dream_expert.encode_targets(noisy_targets)
        dream_pre = self.dream_expert.pre_dit(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            context=context,
            context_mask=context_mask,
            noisy_latent=latent,
            timestep=timestep,
        )
        out = self.mot.forward_dream_with_video_cache(
            dream_tokens=dream_pre["tokens"],
            dream_freqs=dream_pre["freqs"],
            dream_t_mod=dream_pre["t_mod"],
            dream_context_payload={
                "context": dream_pre["context"],
                "mask": dream_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            context_attention_mask=context_attention_mask,
            video_seq_len=video_seq_len,
        )
        prediction = self.dream_expert.post_dit(out["tokens"], dream_pre)
        return {
            "prediction": prediction,
            "tokens": out["tokens"],
            "dream_kv": out["dream_kv"],
            "pre_state": dream_pre,
        }

    def _run_dream_rollout(
        self,
        *,
        num_steps: int,
        scheduler: WanContinuousFlowMatchScheduler,
        initial_targets: dict[str, torch.Tensor],
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        context_attention_mask: torch.Tensor,
        video_seq_len: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Iterate `_dream_step`; the K/V of the final step is the interface."""
        timesteps, deltas = scheduler.build_inference_schedule(
            num_inference_steps=int(num_steps), device=device, dtype=dtype
        )
        current = dict(initial_targets)
        result: dict[str, Any] = {}
        for index in range(int(num_steps)):
            timestep = timesteps[index].reshape(1).expand(batch_size)
            result = self._dream_step(
                noisy_targets=current,
                timestep=timestep,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                context_attention_mask=context_attention_mask,
                video_seq_len=video_seq_len,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            current = {
                name: scheduler.step(
                    model_output=result["prediction"][name].to(value.dtype),
                    delta=deltas[index],
                    sample=value,
                )
                for name, value in current.items()
                if name in result["prediction"]
            }
        result["targets"] = current
        return result

    # ------------------------------------------------------------ split prefill
    def _video_prefill(
        self,
        *,
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        action_seq_len: int,
    ) -> dict[str, Any]:
        """Run the frozen Video expert once and cache its per-layer K/V."""
        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        dream_seq_len = int(self.dream_expert.num_dream_tokens)
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            dream_seq_len=dream_seq_len,
            action_seq_len=int(action_seq_len),
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )
        return {
            "video_kv": video_kv,
            "attention_mask": attention_mask,
            "video_seq_len": video_seq_len,
            "dream_seq_len": dream_seq_len,
        }

    @torch.no_grad()
    def _prefill_video_dream_cache(
        self,
        first_frame_latents: torch.Tensor,
        action_seq_len: int,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        return_dream: bool = False,
    ) -> dict[str, Any]:
        """Inference-time prefill.

        Keeps the parent's return contract so ``DreamFastWAM.infer_action`` and
        every evaluation script work unchanged; only the way the cache is
        produced differs.  In generative mode the Video expert runs once and the
        Dream expert denoises for ``dream_inference_steps`` steps against that
        cache -- which, after interface distillation, is a single step.
        """
        if not self.dream_expert.generative_enabled:
            return super()._prefill_video_dream_cache(
                first_frame_latents=first_frame_latents,
                action_seq_len=action_seq_len,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
                return_dream=return_dream,
            )

        batch_size = int(first_frame_latents.shape[0])
        prefill = self._video_prefill(
            first_frame_latents=first_frame_latents,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            action_seq_len=action_seq_len,
        )
        context_seq_len = prefill["video_seq_len"] + prefill["dream_seq_len"]
        rollout = self._run_dream_rollout(
            num_steps=self.dream_inference_steps,
            scheduler=self.infer_dream_scheduler,
            initial_targets=self._dream_noise_like_targets(
                batch_size=batch_size,
                device=first_frame_latents.device,
                dtype=first_frame_latents.dtype,
            ),
            context=context,
            context_mask=context_mask,
            video_kv_cache=prefill["video_kv"],
            context_attention_mask=prefill["attention_mask"][:context_seq_len, :context_seq_len],
            video_seq_len=prefill["video_seq_len"],
            batch_size=batch_size,
            device=first_frame_latents.device,
            dtype=first_frame_latents.dtype,
        )
        return {
            "kv_cache": self.mot.merge_context_cache(prefill["video_kv"], rollout["dream_kv"]),
            "attention_mask": prefill["attention_mask"],
            "video_seq_len": prefill["video_seq_len"],
            "dream_seq_len": prefill["dream_seq_len"],
            "dream_predictions": rollout["targets"] if return_dream else None,
        }

    # ------------------------------------------------------------------- losses
    def _generative_dream_loss(
        self,
        prediction: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        *,
        future_valid_mask: Optional[torch.Tensor],
        modality_valid_masks: Optional[dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Flow-matching MSE per modality, masked exactly like the dense loss.

        The dense loss uses modality-specific objectives (BCE for the dynamic
        mask, smooth-L1 for depth, cosine for DINO/SAM).  Those are objectives on
        the *value*; here the network regresses a velocity, for which a single
        squared error is the right and only consistent choice.
        """
        modality_lambdas = {
            "dyn": self.loss_lambda_dyn,
            "depth": self.loss_lambda_depth,
            "dino": self.loss_lambda_dino,
            "sam": self.loss_lambda_sam,
        }
        device = next(iter(prediction.values())).device
        total = torch.zeros((), device=device, dtype=torch.float32)
        parts: dict[str, torch.Tensor] = {}
        for name in self.dream_expert.modalities:
            if name not in prediction:
                continue
            pred = prediction[name].float()
            goal = target[name].float()
            if pred.shape != goal.shape:
                raise ValueError(
                    f"{name} velocity shape mismatch: {tuple(pred.shape)} vs {tuple(goal.shape)}."
                )
            per_offset = F.mse_loss(pred, goal, reduction="none")
            per_offset = per_offset.flatten(2).mean(dim=2)  # [B, O]
            mask = None
            if modality_valid_masks is not None and name in modality_valid_masks:
                mask = modality_valid_masks[name].to(device=per_offset.device, dtype=per_offset.dtype)
            elif future_valid_mask is not None:
                mask = future_valid_mask.to(device=per_offset.device, dtype=per_offset.dtype)
            if mask is not None:
                if mask.ndim == 1:
                    mask = mask.unsqueeze(0).expand_as(per_offset)
                value = (per_offset * mask).sum() / mask.sum().clamp(min=1.0)
            else:
                value = per_offset.mean()
            parts[f"loss_{name}"] = value.detach()
            total = total + float(modality_lambdas.get(name, 1.0)) * value
        return total, parts

    def training_loss(self, sample, tiled: bool = False):
        self._refresh_progress()
        if not self.uses_split_path:
            loss, metrics = super().training_loss(sample, tiled=tiled)
            loss, metrics = self._add_router_terms(loss, metrics)
            return loss, metrics
        return self._split_training_loss(sample, tiled=tiled)

    def _add_router_terms(self, loss, metrics: dict[str, float]):
        metrics = dict(metrics)
        router = self.mot.router
        if router is None:
            return loss, metrics
        gates = getattr(self.mot, "last_gates", [])
        budget = router.budget_loss(gates)
        if budget.requires_grad or float(budget.detach().item()) != 0.0:
            loss = loss + budget
        metrics["loss_router_budget"] = float(budget.detach().item())
        metrics.update(router.scalar_metrics())
        return loss, metrics

    def _split_training_loss(self, sample, tiled: bool = False):
        """Train through the deployment computation: Video once, Dream, Action cached."""
        inputs = self.build_inputs(sample, tiled=tiled)
        if self.loss_lambda_video > 0.0:
            raise ValueError(
                "The split path does not denoise future video frames; set "
                "model.loss.lambda_video=0.0 for generative/interface-distilled runs."
            )
        input_latents = inputs["input_latents"]
        batch_size = int(input_latents.shape[0])
        device = input_latents.device
        dtype = input_latents.dtype
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        dream_targets = self._dream_target_reference(inputs["dream_targets"])
        future_valid_mask = inputs.get("future_valid_mask", None)
        modality_valid_masks = inputs.get("modality_valid_masks", None)

        first_frame_latents = inputs["first_frame_latents"]
        if first_frame_latents is None:
            first_frame_latents = input_latents[:, :, 0:1]

        # --- Video prefill (frozen; no gradient path, no optimizer state) ---
        with torch.no_grad():
            prefill = self._video_prefill(
                first_frame_latents=first_frame_latents,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
                action_seq_len=int(action.shape[1]),
            )
        context_seq_len = prefill["video_seq_len"] + prefill["dream_seq_len"]
        context_attention_mask = prefill["attention_mask"][:context_seq_len, :context_seq_len]

        # --- Dream: one noised step, supervised in velocity space ---
        clean_targets = {name: value.to(dtype) for name, value in dream_targets.items()}
        dream_noise = self._sample_dream_noise(clean_targets)
        timestep_dream = self.train_dream_scheduler.sample_training_t(
            batch_size=batch_size, device=device, dtype=dtype
        )
        noisy_targets = {
            name: self.train_dream_scheduler.add_noise(value, dream_noise[name], timestep_dream)
            for name, value in clean_targets.items()
        }
        velocity_targets = {
            name: self.train_dream_scheduler.training_target(
                value, dream_noise[name], timestep_dream
            )
            for name, value in clean_targets.items()
        }
        dream_out = self._dream_step(
            noisy_targets=noisy_targets,
            timestep=timestep_dream,
            context=context,
            context_mask=context_mask,
            video_kv_cache=prefill["video_kv"],
            context_attention_mask=context_attention_mask,
            video_seq_len=prefill["video_seq_len"],
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        loss_dream, dream_parts = self._generative_dream_loss(
            dream_out["prediction"],
            velocity_targets,
            future_valid_mask=future_valid_mask,
            modality_valid_masks=modality_valid_masks,
        )
        dream_weight = self.train_dream_scheduler.training_weight(timestep_dream).mean()
        loss_dream = loss_dream * dream_weight.to(loss_dream.dtype)

        # --- Action: denoise against the merged cache ---
        noise_action = self._sample_action_noise(
            action, use_correlated_noise=self.use_correlated_noise_train
        )
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size, device=device, dtype=action.dtype
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(
            action, noise_action, timestep_action
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_context_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            context_kv_cache=self.mot.merge_context_cache(
                prefill["video_kv"], dream_out["dream_kv"]
            ),
            attention_mask=prefill["attention_mask"],
            video_seq_len=prefill["video_seq_len"],
            dream_seq_len=prefill["dream_seq_len"],
        )
        pred_action = self.action_expert.post_dit(action_tokens, action_pre)

        token_loss = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        if action_is_pad is None:
            per_sample = token_loss.mean(dim=1)
        else:
            valid = (~action_is_pad).to(device=token_loss.device, dtype=token_loss.dtype)
            per_sample = (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            device=per_sample.device, dtype=per_sample.dtype
        )
        loss_action = (per_sample * action_weight).mean()

        loss_total = self.loss_lambda_action * loss_action + self.loss_lambda_dream * loss_dream
        metrics = {
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
            "loss_dream": self.loss_lambda_dream * float(loss_dream.detach().item()),
            "dream_timestep_mean": float(timestep_dream.detach().float().mean().item()),
        }
        for name, value in dream_parts.items():
            metrics[name] = self.loss_lambda_dream * float(value.item())

        # --- Interface distillation ---
        if self.distiller is not None:
            loss_iface, iface_metrics = self._interface_distillation_loss(
                student_kv=dream_out["dream_kv"],
                context=context,
                context_mask=context_mask,
                video_kv_cache=prefill["video_kv"],
                context_attention_mask=context_attention_mask,
                video_seq_len=prefill["video_seq_len"],
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            loss_total = loss_total + self.distiller.current_weight() * loss_iface
            metrics.update(iface_metrics)

        loss_total, metrics = self._add_router_terms(loss_total, metrics)
        return loss_total, metrics

    def _interface_distillation_loss(
        self,
        *,
        student_kv: list[dict[str, torch.Tensor]],
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        context_attention_mask: torch.Tensor,
        video_seq_len: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        assert self.distiller is not None
        with torch.no_grad():
            with self.distiller.use_teacher_dream(self.mot):
                teacher = self._run_dream_rollout(
                    num_steps=self.distill_config.teacher_steps,
                    scheduler=self.infer_dream_scheduler,
                    initial_targets=self._dream_noise_like_targets(
                        batch_size=batch_size, device=device, dtype=dtype
                    ),
                    context=context,
                    context_mask=context_mask,
                    video_kv_cache=video_kv_cache,
                    context_attention_mask=context_attention_mask,
                    video_seq_len=video_seq_len,
                    batch_size=batch_size,
                    device=device,
                    dtype=dtype,
                )

        keep_mask = None
        if self.distill_config.route_aware and self.mot.router is not None:
            gates = getattr(self.mot, "last_gates", [])
            if gates:
                keep_mask = (torch.stack(gates, dim=0).mean(dim=0) > self.router_config.gate_threshold)

        loss, parts = self.distiller.kv_loss(
            student_kv=student_kv,
            teacher_kv=teacher["dream_kv"],
            keep_mask=keep_mask,
        )
        metrics = {
            "loss_interface": float(loss.detach().item()),
            "interface_weight": float(self.distiller.current_weight()),
        }
        for key, value in parts.items():
            metrics[key] = float(value.item())
        return loss, metrics
