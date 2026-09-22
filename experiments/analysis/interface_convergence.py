"""Does the imagination *interface* converge faster than the imagination itself?

This is the measurement the interface-distillation claim rests on.

The action expert never reads the imagined depth map, DINO field or
segmentation. Across the whole MoT it only reads the per-layer keys and values
of the Dream tokens. So the question that decides whether a one-step student can
replace an N-step teacher is not "how many steps does the future prediction need
to converge" but "how many steps does the *K/V the policy reads* need to
converge". If the interface settles earlier than the output -- which is what we
expect, since K/V are a heavily pooled, 30-layer-smoothed view of the sample --
then one-step interface distillation is close to free, and distilling in output
space (what step-distillation methods do) is paying for detail nobody reads.

For every denoising step ``t`` this reports, relative to the final step ``N``:

    interface_drift(t) = mean over layers and tokens of  1 - cos(KV_t, KV_N)
    output_drift(t)    = mean over modalities of  ||y_t - y_N|| / ||y_N||

A crossing point where ``interface_drift`` is already near zero while
``output_drift`` is not is the quantitative form of the claim.

    python experiments/analysis/interface_convergence.py \
      --task routed_wam_libero_goal --ckpt runs/.../step_00X.pt --steps 16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _cosine_drift(a: list[dict[str, torch.Tensor]], b: list[dict[str, torch.Tensor]]) -> float:
    """Mean over layers, tokens and {k, v} of 1 - cosine similarity."""
    values = []
    for layer_a, layer_b in zip(a, b):
        for key in ("k", "v"):
            cos = torch.nn.functional.cosine_similarity(
                layer_a[key].float(), layer_b[key].float(), dim=-1, eps=1e-6
            )
            values.append(float((1.0 - cos).mean().item()))
    return sum(values) / max(len(values), 1)


def _relative_drift(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    values = []
    for name in a:
        if name not in b:
            continue
        x, y = a[name].float(), b[name].float()
        denominator = y.norm().clamp(min=1e-8)
        values.append(float(((x - y).norm() / denominator).item()))
    return sum(values) / max(len(values), 1)


@torch.no_grad()
def measure_convergence(
    model,
    *,
    num_steps: int,
    video_kv_cache: list[dict[str, torch.Tensor]],
    context_attention_mask: torch.Tensor,
    video_seq_len: int,
    context: Optional[torch.Tensor],
    context_mask: Optional[torch.Tensor],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    initial_targets: Optional[dict[str, torch.Tensor]] = None,
) -> dict[str, Any]:
    """Run one Dream rollout, recording the interface and the sample at each step.

    `model` only needs to expose the RoutedWAM Dream machinery
    (`_dream_step`, `_dream_noise_like_targets`, `infer_dream_scheduler`), so
    this is testable against a stub that does not require the 5B video expert.
    """
    if num_steps < 2:
        raise ValueError(f"--steps must be >= 2 to measure convergence, got {num_steps}.")

    scheduler = model.infer_dream_scheduler
    timesteps, deltas = scheduler.build_inference_schedule(
        num_inference_steps=num_steps, device=device, dtype=dtype
    )
    current = dict(
        initial_targets
        if initial_targets is not None
        else model._dream_noise_like_targets(batch_size=batch_size, device=device, dtype=dtype)
    )

    interfaces: list[list[dict[str, torch.Tensor]]] = []
    samples: list[dict[str, torch.Tensor]] = []
    for index in range(num_steps):
        timestep = timesteps[index].reshape(1).expand(batch_size)
        step = model._dream_step(
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
        interfaces.append(
            [{"k": entry["k"].detach().clone(), "v": entry["v"].detach().clone()} for entry in step["dream_kv"]]
        )
        current = {
            name: scheduler.step(
                model_output=step["prediction"][name].to(value.dtype),
                delta=deltas[index],
                sample=value,
            )
            for name, value in current.items()
            if name in step["prediction"]
        }
        samples.append({name: value.detach().clone() for name, value in current.items()})

    final_interface = interfaces[-1]
    final_sample = samples[-1]
    rows = []
    for index in range(num_steps):
        rows.append(
            {
                "step": index,
                "timestep": float(timesteps[index].item()),
                "interface_drift": _cosine_drift(interfaces[index], final_interface),
                "output_drift": _relative_drift(samples[index], final_sample),
            }
        )

    # How many steps until each quantity is within 5% of its starting drift?
    def steps_to_settle(key: str, tolerance: float = 0.05) -> Optional[int]:
        start = rows[0][key]
        if start <= 0:
            return 0
        for row in rows:
            if row[key] <= tolerance * start:
                return int(row["step"])
        return None

    return {
        "num_steps": num_steps,
        "per_step": rows,
        "interface_steps_to_settle": steps_to_settle("interface_drift"),
        "output_steps_to_settle": steps_to_settle("output_drift"),
        "interface_drift_at_step_0": rows[0]["interface_drift"],
        "output_drift_at_step_0": rows[0]["output_drift"],
    }


def format_report(result: dict[str, Any]) -> str:
    lines = [f"{'step':>5} {'timestep':>10} {'interface':>12} {'output':>12}"]
    for row in result["per_step"]:
        lines.append(
            f"{row['step']:>5} {row['timestep']:>10.1f} "
            f"{row['interface_drift']:>12.5f} {row['output_drift']:>12.5f}"
        )
    lines.append("")
    lines.append(
        f"steps until within 5% of the initial drift -- "
        f"interface: {result['interface_steps_to_settle']}, "
        f"output: {result['output_steps_to_settle']}"
    )
    lines.append(
        "If the interface settles in fewer steps than the output, one-step "
        "interface distillation is cheap and output-space step distillation is "
        "paying for detail the policy cannot read."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="routed_wam_libero_goal")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", default=None)
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from torch.utils.data import DataLoader

    from fastwam.runtime import build_datasets
    from fastwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    with initialize_config_dir(config_dir=str(_REPO_ROOT / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="train", overrides=[f"task={args.task}", *args.overrides]
        )

    device = torch.device(args.device)
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=str(device))
    model.load_checkpoint(args.ckpt)
    model.eval()
    if not getattr(model.dream_expert, "generative_enabled", False):
        raise SystemExit(
            "This measurement needs a generative Dream branch; the configured "
            "model regresses its targets in a single pass and has no rollout to "
            "measure. Use a routed_wam task config."
        )

    # The interface is conditioned on the current observation, so the numbers
    # only mean something on real frames. Reuse the training dataset rather than
    # fabricating latents.
    train_ds, _ = build_datasets(cfg.data)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    aggregated: list[dict[str, Any]] = []
    for batch_index, sample in enumerate(loader):
        if batch_index >= args.num_batches:
            break
        with torch.no_grad():
            inputs = model.build_inputs(sample)
            first_frame = inputs["first_frame_latents"]
            if first_frame is None:
                first_frame = inputs["input_latents"][:, :, 0:1]
            prefill = model._video_prefill(
                first_frame_latents=first_frame,
                context=inputs["context"],
                context_mask=inputs["context_mask"],
                fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
                action_seq_len=int(inputs["action"].shape[1]),
            )
            context_seq_len = prefill["video_seq_len"] + prefill["dream_seq_len"]
            result = measure_convergence(
                model,
                num_steps=args.steps,
                video_kv_cache=prefill["video_kv"],
                context_attention_mask=prefill["attention_mask"][
                    :context_seq_len, :context_seq_len
                ],
                video_seq_len=prefill["video_seq_len"],
                context=inputs["context"],
                context_mask=inputs["context_mask"],
                batch_size=int(first_frame.shape[0]),
                device=device,
                dtype=first_frame.dtype,
            )
        aggregated.append(result)

    if not aggregated:
        raise SystemExit("The dataset yielded no batches.")

    merged = {
        "task": args.task,
        "ckpt": args.ckpt,
        "num_steps": args.steps,
        "num_batches": len(aggregated),
        "per_step": [
            {
                "step": index,
                "timestep": aggregated[0]["per_step"][index]["timestep"],
                "interface_drift": sum(r["per_step"][index]["interface_drift"] for r in aggregated)
                / len(aggregated),
                "output_drift": sum(r["per_step"][index]["output_drift"] for r in aggregated)
                / len(aggregated),
            }
            for index in range(args.steps)
        ],
    }
    settled = [r["interface_steps_to_settle"] for r in aggregated if r["interface_steps_to_settle"] is not None]
    merged["interface_steps_to_settle"] = (sum(settled) / len(settled)) if settled else None
    settled = [r["output_steps_to_settle"] for r in aggregated if r["output_steps_to_settle"] is not None]
    merged["output_steps_to_settle"] = (sum(settled) / len(settled)) if settled else None

    print(format_report(merged))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(merged, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
