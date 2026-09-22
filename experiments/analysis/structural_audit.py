"""Structural audit of the MoT imagination interface.

Answers three questions with no training, no checkpoint and no dataset:

  Q1  Which tokens can the action expert actually read?
  Q2  What fraction of the compute produces tokens that nothing in the action
      path can read?
  Q3  What does the imagination interface cost, per modality and horizon?

Everything is derived from the *shipped* mask constructors
(``WanVideoDiT.build_video_to_video_mask``, ``DreamFastWAM._build_mot_attention_mask``)
and the composed Hydra config, so the numbers cannot drift from the code they
describe.

    python experiments/analysis/structural_audit.py --task dream_fastwam_libero_goal
    python experiments/analysis/structural_audit.py --task routed_wam_libero_goal --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from fastwam.models.wan22.dream_fastwam.dream_query_expert import DreamQueryExpert  # noqa: E402
from fastwam.models.wan22.dream_fastwam.model import DreamFastWAM  # noqa: E402
from fastwam.models.wan22.wan_video_dit import WanVideoDiT  # noqa: E402


def compose_task(task: str) -> dict:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from fastwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    with initialize_config_dir(config_dir=str(_REPO_ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"task={task}"])
    return OmegaConf.to_container(cfg, resolve=True)


def build_dream_expert(model_cfg: dict) -> DreamQueryExpert:
    dream_cfg = dict(model_cfg["dream_query_config"])
    return DreamQueryExpert(
        text_dim=dream_cfg["text_dim"],
        freq_dim=dream_cfg["freq_dim"],
        eps=dream_cfg["eps"],
        num_heads=dream_cfg["num_heads"],
        attn_head_dim=dream_cfg["attn_head_dim"],
        dream_query=dream_cfg["dream_query"],
        dream_expert=dream_cfg["dream_expert"],
        dream_decoder=dream_cfg["dream_decoder"],
    )


def build_real_mask(*, num_latent_frames, tokens_per_frame, dream_expert, action_seq_len):
    """Call the shipped mask builders rather than re-deriving them."""
    video_stub = object.__new__(WanVideoDiT)
    torch.nn.Module.__init__(video_stub)
    video_stub.video_attention_mask_mode = "first_frame_causal"

    model_stub = object.__new__(DreamFastWAM)
    torch.nn.Module.__init__(model_stub)
    model_stub.video_expert = video_stub
    model_stub.dream_expert = dream_expert

    video_seq_len = num_latent_frames * tokens_per_frame
    mask = model_stub._build_mot_attention_mask(
        video_seq_len=video_seq_len,
        dream_seq_len=dream_expert.num_dream_tokens,
        action_seq_len=action_seq_len,
        video_tokens_per_frame=tokens_per_frame,
        device=torch.device("cpu"),
    )
    return mask, video_seq_len


def block_flops(cfg: dict, seq_q: int, seq_kv: int, ctx_len: int) -> int:
    """Forward FLOPs (2 x MACs) of one DiTBlock.

    `seq_q` is the expert's own token count and `seq_kv` the length of the mixed
    MoT sequence it attends over: each expert projects only its own tokens into
    the shared inner space (`mot.py::_build_expert_attention_io`) and the
    concatenated keys are then attended jointly. Context is pre-embedded to
    `hidden` by the expert, so cross-attention k/v are `hidden -> inner`
    (`CrossAttention.__init__`).
    """
    d, f = cfg["hidden"], cfg["ffn"]
    inner = cfg["heads"] * cfg["head_dim"]
    self_qkv = 3 * 2 * seq_q * d * inner
    self_attn = 2 * 2 * seq_q * seq_kv * inner
    self_out = 2 * seq_q * inner * d
    cross_q = 2 * seq_q * d * inner
    cross_kv = 2 * 2 * ctx_len * d * inner
    cross_attn = 2 * 2 * seq_q * ctx_len * inner
    cross_out = 2 * seq_q * inner * d
    ffn = 2 * 2 * seq_q * d * f
    return self_qkv + self_attn + self_out + cross_q + cross_kv + cross_attn + cross_out + ffn


def expert_flops(cfg: dict, seq_q: int, seq_kv: int, ctx_len: int) -> int:
    return cfg["layers"] * block_flops(cfg, seq_q, seq_kv, ctx_len)


def expert_params(cfg: dict) -> int:
    d, f, inner = cfg["hidden"], cfg["ffn"], cfg["heads"] * cfg["head_dim"]
    per_layer = (3 * d * inner + inner * d) * 2 + 2 * d * f
    return cfg["layers"] * per_layer


def _expert_shape(block_cfg: dict) -> dict:
    return {
        "hidden": int(block_cfg["hidden_dim"]),
        "ffn": int(block_cfg["ffn_dim"]),
        "heads": int(block_cfg["num_heads"]),
        "head_dim": int(block_cfg["attn_head_dim"]),
        "layers": int(block_cfg["num_layers"]),
    }


def audit(task: str) -> dict:
    cfg = compose_task(task)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]["train"]

    video_size = list(data_cfg["video_size"])
    num_frames = int(data_cfg["num_frames"])
    action_seq_len = num_frames - 1
    vae_spatial, vae_temporal = 16, 4
    patch = list(model_cfg["video_dit_config"]["patch_size"])

    latent_t = (num_frames - 1) // vae_temporal + 1
    latent_h = video_size[0] // vae_spatial
    latent_w = video_size[1] // vae_spatial
    tokens_per_frame = (latent_h // patch[1]) * (latent_w // patch[2])

    dream_expert = build_dream_expert(model_cfg)
    mask, video_seq_len = build_real_mask(
        num_latent_frames=latent_t,
        tokens_per_frame=tokens_per_frame,
        dream_expert=dream_expert,
        action_seq_len=action_seq_len,
    )
    dream_seq_len = dream_expert.num_dream_tokens
    total = video_seq_len + dream_seq_len + action_seq_len
    video_slice = slice(0, video_seq_len)
    action_slice = slice(video_seq_len + dream_seq_len, total)

    action_rows = mask[action_slice]
    readable_video = int(action_rows[:, video_slice].any(dim=0).sum())

    # Transitive closure: which tokens can influence the action expert at all,
    # through any chain of attention edges?
    reachable = torch.zeros(total, dtype=torch.bool)
    reachable[action_slice] = True
    for _ in range(len(mask)):
        grown = reachable | mask[reachable].any(dim=0)
        if bool((grown == reachable).all()):
            break
        reachable = grown
    unreachable_video = int((~reachable[video_slice]).sum())

    video_shape = _expert_shape(
        {
            "hidden_dim": model_cfg["video_dit_config"]["hidden_dim"],
            "ffn_dim": model_cfg["video_dit_config"]["ffn_dim"],
            "num_heads": model_cfg["video_dit_config"]["num_heads"],
            "attn_head_dim": model_cfg["video_dit_config"]["attn_head_dim"],
            "num_layers": model_cfg["video_dit_config"]["num_layers"],
        }
    )
    dream_shape = _expert_shape(model_cfg["dream_query_config"]["dream_expert"])
    action_shape = _expert_shape(model_cfg["action_dit_config"])
    ctx_len = int(model_cfg.get("tokenizer_max_len", 128))

    f_video = expert_flops(video_shape, video_seq_len, total, ctx_len)
    f_dream = expert_flops(dream_shape, dream_seq_len, total, ctx_len)
    f_action = expert_flops(action_shape, action_seq_len, total, ctx_len)
    f_total = f_video + f_dream + f_action

    current = tokens_per_frame
    total_current = current + dream_seq_len + action_seq_len
    f_video_current = expert_flops(video_shape, current, total_current, ctx_len)
    f_dream_current = expert_flops(dream_shape, dream_seq_len, total_current, ctx_len)
    f_action_current = expert_flops(action_shape, action_seq_len, total_current, ctx_len)

    dream_breakdown = {
        name: dream_expert.num_future_offsets * int(getattr(dream_expert, f"n_{name}"))
        for name in dream_expert.modalities
    }

    return {
        "task": task,
        "latent_grid": {
            "T": latent_t,
            "H": latent_h,
            "W": latent_w,
            "tokens_per_frame": tokens_per_frame,
        },
        "sequence": {
            "video_tokens": video_seq_len,
            "dream_tokens": dream_seq_len,
            "action_tokens": action_seq_len,
            "total": total,
            "dream_breakdown": dream_breakdown,
            "future_offsets": list(dream_expert.future_offsets),
        },
        "action_readability": {
            "video_tokens_readable_by_action": readable_video,
            "fraction_of_video_readable": readable_video / video_seq_len,
            "dream_tokens_readable_by_action": int(
                action_rows[:, video_seq_len : video_seq_len + dream_seq_len].any(dim=0).sum()
            ),
            "video_tokens_unreachable_from_action": unreachable_video,
            "fraction_video_unreachable": unreachable_video / video_seq_len,
        },
        "mask_density": {
            "overall": float(mask.float().mean()),
            "action_rows": float(action_rows.float().mean()),
        },
        "params_M": {
            "video_expert": expert_params(video_shape) / 1e6,
            "dream_expert_dit": expert_params(dream_shape) / 1e6,
            "dream_expert_total": sum(p.numel() for p in dream_expert.parameters()) / 1e6,
            "dream_decoders": sum(p.numel() for p in dream_expert.decoders.parameters()) / 1e6,
            "action_expert": expert_params(action_shape) / 1e6,
        },
        "tflops": {
            "train_video_expert": f_video / 1e12,
            "train_dream_expert": f_dream / 1e12,
            "train_action_expert": f_action / 1e12,
            "train_total": f_total / 1e12,
            "video_share_of_train": f_video / f_total,
            "wasted_on_unreachable_video": f_video * (unreachable_video / video_seq_len) / 1e12,
            "wasted_fraction_of_train": (f_video * (unreachable_video / video_seq_len)) / f_total,
            "infer_video_prefill": f_video_current / 1e12,
            "infer_dream_per_step": f_dream_current / 1e12,
            "infer_action_per_step": f_action_current / 1e12,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="dream_fastwam_libero_goal")
    parser.add_argument("--json", default=None, help="Write the report to this path.")
    args = parser.parse_args()

    report = audit(args.task)
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text, encoding="utf-8")

    readable = report["action_readability"]
    tflops = report["tflops"]
    print(
        "\nheadline: the action expert can read "
        f"{readable['video_tokens_readable_by_action']}/"
        f"{report['sequence']['video_tokens']} video tokens "
        f"({100 * readable['fraction_of_video_readable']:.1f}%); "
        f"{100 * readable['fraction_video_unreachable']:.1f}% are unreachable from it, "
        f"consuming {100 * tflops['wasted_fraction_of_train']:.1f}% of training FLOPs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
