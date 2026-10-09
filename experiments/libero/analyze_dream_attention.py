"""Record Action attention along native, cached LIBERO rollouts.

Only dense/Full routing is supported. Attention is recomputed in float32 from
the actual post-RoPE Q/K and boolean mask; the original attention output is
returned unchanged. No full attention tensors are retained.
The main output is one row per replan with mean-per-token scores.
--attention-sources dream (default) retains dyn/depth/dino/sam;
--attention-sources all includes video and action (self-attention) as well.
By default, average action queries and modality tokens within each head,
select the strongest head per layer/denoising step/modality, then average
layers and denoising steps. --head-reduction mean restores the original mean.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

# Direct script execution puts only experiments/libero on sys.path. Resolve
# local packages from this file so the CLI does not depend on PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "experiments" / "libero"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import numpy as np
import torch

MODALITIES = ("dyn", "depth", "dino", "sam")
ALL_SOURCES = ("video", *MODALITIES, "action")


def attention_source_names(mode):
    if mode == "dream":
        return MODALITIES
    if mode == "all":
        return ALL_SOURCES
    raise ValueError("attention_sources must be dream or all.")


@torch.no_grad()
def modality_masses(q_action, k_all, mask, *, num_heads, modality_slices,
                    context_slices, action_slice, prefix_queries, head_reduction="max",
                    attention_sources="dream"):
    """Reduce queries and tokens per head, then take the max or mean head.

    Report both the whole predicted action horizon and the prefix scheduled
    for execution before the next replan. The denominator includes video,
    Dream and action keys, never just Dream keys. Different modalities may
    select different heads, so max-head scores do not form one distribution.
    Diagnostic video/self/Dream masses refer to the selected head for each
    modality (or the mean head in mean mode).
    """
    if head_reduction not in ("mean", "max"):
        raise ValueError("head_reduction must be mean or max.")
    names = attention_source_names(attention_sources)
    if q_action.shape[0] != 1 or k_all.shape[0] != 1:
        raise ValueError("Attention analysis requires batch size 1.")
    if mask.dtype != torch.bool or not bool(mask.any(-1).all()):
        raise ValueError("Expected a boolean mask with at least one key per query.")
    head_dim = q_action.shape[-1] // num_heads
    with torch.autocast(device_type=q_action.device.type, enabled=False):
        def heads(x):
            return x.float().reshape(1, -1, num_heads, head_dim).transpose(1, 2)
        scores = heads(q_action) @ heads(k_all).transpose(-2, -1)
        probs = (scores * head_dim ** -0.5).masked_fill(~mask, -torch.inf).softmax(-1)
        # [scope, head, key]: retain heads until after modality aggregation.
        means = torch.stack([probs[0].mean(1), probs[0, :, :prefix_queries].mean(1)]).cpu().numpy()
    dream = context_slices["dream"]
    groups = {name: slice(dream.start + sl.start, dream.start + sl.stop)
              for name, sl in modality_slices.items()}
    if sorted(i for sl in groups.values() for i in range(sl.start, sl.stop)) != list(range(dream.start, dream.stop)):
        raise ValueError("Modality slices must partition the actual Dream keys.")
    groups.update(video=context_slices["video"], action=action_slice)
    output = []
    for scope, per_head in zip(("all", "executed_prefix"), means):
        for name in names:
            sl = groups[name]
            if sl.stop <= sl.start:
                raise ValueError(f"Empty attention source token group: {name}.")
            if head_reduction == "max":
                selected_head = int(per_head[:, sl].mean(-1).argmax())
                mean = per_head[selected_head]
            else:
                selected_head = None
                mean = per_head.mean(0)
            dream_mass = float(mean[dream].sum())
            mass = float(mean[sl].sum())
            output.append(dict(query_scope=scope, modality=name,
                               head_reduction=head_reduction, selected_head=selected_head,
                               num_queries=q_action.shape[1] if scope == "all" else min(prefix_queries, q_action.shape[1]),
                               num_tokens=sl.stop - sl.start, mass=mass,
                               dream_share=mass / dream_mass if name in MODALITIES and dream_mass > 0 else None,
                               dream_mass=dream_mass,
                               video_mass=float(mean[context_slices["video"]].sum()),
                               self_mass=float(mean[action_slice].sum())))
    return output


class AttentionRecorder:
    """Temporary observers on the native inference path, restored on exit."""

    def __init__(self, model, episode_dir, *, episode_id, replan_steps, wait_steps,
                 layers, denoise_steps, num_inference_steps, save_detail=False, head_reduction="max",
                 attention_sources="dream"):
        self.model = model
        self.directory = Path(episode_dir)
        self.episode_id = episode_id
        self.replan_steps = replan_steps
        self.wait_steps = wait_steps
        self.layers = set(layers)
        self.steps = set(denoise_steps)
        self.num_inference_steps = num_inference_steps
        self.save_detail = save_detail
        if head_reduction not in ("mean", "max"):
            raise ValueError("head_reduction must be mean or max.")
        self.head_reduction = head_reduction
        attention_source_names(attention_sources)
        self.attention_sources = attention_sources
        self.rows = []
        self.replan = -1
        self.step = -1
        self.slices = model.dream_expert.modality_slices()
        if set(self.slices) != set(MODALITIES):
            raise ValueError("Expected dyn/depth/dino/sam Dream groups.")
        mot = model.mot
        feature_router = getattr(mot, "feature_router", None)
        if getattr(mot, "routing_enabled", False) or (feature_router is not None and not feature_router.full):
            raise ValueError("This observer requires Full routing (no modified attention logits).")

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stack = ExitStack()
        self.stack.__enter__()
        infer = self.model.infer_action
        predict = self.model._predict_action_noise_with_cache
        attend = self.model.mot._action_attention_with_context_cache

        def observed_infer(*args, **kwargs):
            self.replan += 1
            self.step = -1
            start = len(self.rows)
            self.seen = set()
            self.noise_seed = kwargs.get("seed")
            self.horizon = int(kwargs["action_horizon"])
            output = infer(*args, **kwargs)
            expected = {(step, layer) for step in self.steps for layer in self.layers}
            if self.step + 1 != self.num_inference_steps or self.seen != expected:
                raise RuntimeError("Incomplete attention capture: inference path/layer schedule changed.")
            if self.save_detail:
                append_csv(self.directory / "attention_detail.csv", self.rows[start:])
            # Persist the selected source scores after every completed replan.
            append_csv(self.directory / "attention_per_replan.csv",
                       summarize_replans(self.rows[start:], attention_sources=self.attention_sources))
            return output

        def observed_predict(*args, **kwargs):
            self.step += 1
            self.timestep = float(kwargs["timestep_action"].detach().float().item())
            return predict(*args, **kwargs)

        def observed_attend(**kwargs):
            layer = int(kwargs["layer_idx"])
            if layer in self.layers and self.step in self.steps:
                gates = kwargs.get("dream_token_gates")
                if gates is not None and not bool((gates == 1).all()):
                    raise ValueError("Non-unit Dream gates: dense attention analysis is not valid.")
                key = (self.step, layer)
                if key in self.seen:
                    raise RuntimeError("Duplicate layer/denoising step in cached inference.")
                self.seen.add(key)
                action_slice = kwargs["action_slice"]
                values = modality_masses(
                    kwargs["q_action"], kwargs["k_all"], kwargs["attention_mask"][action_slice, :],
                    num_heads=self.model.mot.num_heads, modality_slices=self.slices,
                    context_slices=kwargs["context_slices"], action_slice=action_slice,
                    prefix_queries=min(self.replan_steps, self.horizon), head_reduction=self.head_reduction,
                    attention_sources=self.attention_sources)
                for row in values:
                    self.rows.append(dict(episode=self.episode_id, replan=self.replan,
                                          env_step=self.wait_steps + self.replan * min(self.replan_steps, self.horizon),
                                          noise_seed=self.noise_seed, denoise_step=self.step,
                                          timestep=self.timestep, layer=layer, **row))
            return attend(**kwargs)

        self.stack.enter_context(patch.object(self.model, "infer_action", observed_infer))
        self.stack.enter_context(patch.object(self.model, "_predict_action_noise_with_cache", observed_predict))
        self.stack.enter_context(patch.object(self.model.mot, "_action_attention_with_context_cache", observed_attend))
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)


def append_csv(path, rows):
    if not rows:
        return
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def summarize_replans(rows, *, attention_sources="dream"):
    """One row per replan: mean over layers/steps/action queries/source keys.

    Each input mass already reduces heads (max or mean) and averages action
    queries but sums source tokens. Divide by that source's actual key count
    (36 per Dream modality, variable for video/action), including masked keys
    whose attention is zero, then average layers/steps. No renormalization.
    """
    names = attention_source_names(attention_sources)
    groups = defaultdict(list)
    for row in rows:
        if row["query_scope"] == "all":
            groups[(row["episode"], row["replan"])].append(row)
    summary = []
    for key, items in sorted(groups.items()):
        scores = {}
        for modality in names:
            selected = [x for x in items if x["modality"] == modality]
            if not selected or any(x["num_tokens"] <= 0 for x in selected):
                raise ValueError(f"Missing attention or empty tokens for {modality} at replan {key}.")
            scores[modality] = float(np.mean([x["mass"] / x["num_tokens"] for x in selected]))
        summary.append(dict(episode=key[0], replan=key[1], env_step=items[0]["env_step"], **scores))
    return summary


def save_plots(directory, summary, *, head_reduction="max", attention_sources="dream"):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/fastwam_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = attention_source_names(attention_sources)
    colors = dict(video="#9467bd", dyn="#1f77b4", depth="#ff7f0e",
                  dino="#2ca02c", sam="#d62728", action="#8c564b")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for modality in names:
        ax.plot([r["replan"] for r in summary], [r[modality] for r in summary],
                marker=".", label=modality, color=colors[modality])
    ax.set_xlabel("Replan index (0-based)")
    ax.set_ylabel("Mean attention per source token")
    target = "Dream" if attention_sources == "dream" else "Video / Dream / Action"
    ax.set_title(f"Action -> {target} attention | {directory.name} | heads: {head_reduction}")
    values = np.asarray([[r[m] for m in names] for r in summary], dtype=float)
    low, high = float(values.min()), float(values.max())
    margin = max((high - low) * 0.12, abs(high) * 0.005, 1e-8)
    ax.set_ylim(max(0.0, low - margin), high + margin)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    ax.legend()
    ax.grid(alpha=0.25)
    reduction_label = ("Mean queries/tokens per head -> max head -> mean layers/denoising steps."
                       if head_reduction == "max" else
                       "Mean over layers, denoising steps, heads, Action queries and source tokens.")
    fig.text(0.5, 0.02,
             reduction_label + " Y-axis zoomed.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    filename = "modality_attention.png" if attention_sources == "dream" else "source_attention.png"
    fig.savefig(directory / filename, dpi=160)
    plt.close(fig)


def selected_indices(raw, total):
    values = list(range(total)) if raw == "all" else [int(v) for v in raw.split(",")]
    if not values or len(set(values)) != len(values) or any(v < 0 or v >= total for v in values):
        raise ValueError(f"Invalid selection {raw!r}; expected unique indices in [0, {total}).")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config-name", default="dense_router_libero_10_full")
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--layers", default="all", help="all or comma-separated 0-based layer ids")
    parser.add_argument("--denoise-steps", default="all", help="all or comma-separated 0-based denoising step ids")
    parser.add_argument("--head-reduction", choices=("max", "mean"), default="max",
                        help="Reduce heads after averaging queries/modality tokens, before averaging layers/steps.")
    parser.add_argument("--attention-sources", choices=("dream", "all"), default="dream",
                        help="dream: four Dream groups (original); all: video,dyn,depth,dino,sam,action.")
    parser.add_argument("--save-detail", action="store_true", help="Also save per-layer/step diagnostic masses.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", required=True, help="New output directory; existing paths are refused.")
    args = parser.parse_args()
    if args.num_episodes < 1 or args.start_episode < 0 or args.num_inference_steps < 1:
        parser.error("Episode count/inference steps must be positive and start-episode nonnegative.")

    # Defer simulator/model imports so CPU metric tests and --help need no LIBERO.
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/fastwam_numba_cache")
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from PIL import Image
    from experiments.libero.eval_libero_single import (
        _apply_training_model_config, _load_model_checkpoint, _maybe_load_action_noise_stats,
        _mixed_precision_to_model_dtype, _resolve_dataset_stats_path, run_single_episode,
    )
    from experiments.libero.libero_utils import LIBERO_ENV_RESOLUTION, get_libero_env, save_rollout_video
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    from fastwam.utils.pytorch_utils import set_global_seed
    from libero.libero import benchmark

    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
        cfg = compose(config_name="sim_libero", overrides=[
            f"task={args.config_name}", f"ckpt={args.checkpoint}", f"seed={args.seed}",
            f"EVALUATION.task_suite_name={args.task_suite}", f"EVALUATION.task_id={args.task_id}",
            f"EVALUATION.num_inference_steps={args.num_inference_steps}",
            f"EVALUATION.device={args.device}", f"EVALUATION.output_dir={args.out_dir}",
            f"EVALUATION.num_trials={args.num_episodes}"])
    training_config = _apply_training_model_config(cfg)
    if training_config is None:
        raise ValueError("A saved training config.yaml is required next to the checkpoint run.")
    stats_path = _resolve_dataset_stats_path(cfg)
    stats = load_dataset_stats_from_json(str(stats_path))
    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    task = suite.get_task(args.task_id)
    states = suite.get_task_init_states(args.task_id)
    if args.start_episode + args.num_episodes > len(states):
        raise ValueError("Requested episodes exceed available initial states.")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    set_global_seed(args.seed, get_worker_init_fn=False)
    dtype = _mixed_precision_to_model_dtype(cfg.mixed_precision)
    model = instantiate(cfg.model, model_dtype=dtype, device=args.device)
    _load_model_checkpoint(model, args.checkpoint)
    model = model.to(args.device).eval()
    _maybe_load_action_noise_stats(model, cfg, stats_path)
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(stats)
    layers = selected_indices(args.layers, model.mot.num_layers)
    steps = selected_indices(args.denoise_steps, args.num_inference_steps)
    horizon = int(cfg.EVALUATION.action_horizon or (int(cfg.data.train.num_frames)-1))
    replan_steps = int(cfg.EVALUATION.replan_steps)
    height, width = map(int, cfg.data.train.video_size)
    OmegaConf.save(cfg, out / "resolved_attention_config.yaml")
    metadata = dict(vars(args), training_config=str(training_config), dataset_stats=str(stats_path),
                    layers=layers, denoise_steps=steps, action_horizon=horizon, replan_steps=replan_steps,
                    modality_slices={k: [v.start, v.stop] for k, v in model.dream_expert.modality_slices().items()},
                    metric="mean softmax attention per source token; softmax denominator includes all visible keys",
                    aggregation=f"mean over action queries and source tokens, then {args.head_reduction} over heads independently per layer/denoising step/source, then mean over selected layers and denoising steps",
                    source_token_averaging="divide by actual number of keys in each source, including masked zero-probability keys; action denotes all action self-attention keys, not just the diagonal",
                    output_columns=["episode", "replan", "env_step", *attention_source_names(args.attention_sources)],
                    detail_query_scopes="optional detail only: all and executed_prefix",
                    env_step="includes initialization wait; rollout_step excludes it",
                    note="Attention is an association measure, not a causal contribution. Max-head source scores may come from different heads and are per-token means; their sum is not a probability distribution.")
    (out / "run_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    env, description = get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    outcomes = []
    try:
        for episode in range(args.start_episode, args.start_episode + args.num_episodes):
            directory = out / f"episode_{episode:03d}"
            with AttentionRecorder(model, directory, episode_id=episode, replan_steps=replan_steps,
                                   wait_steps=int(cfg.EVALUATION.num_steps_wait), layers=layers,
                                   denoise_steps=steps, num_inference_steps=args.num_inference_steps,
                                   save_detail=args.save_detail, head_reduction=args.head_reduction,
                                   attention_sources=args.attention_sources) as recorder:
                success, replay, _, _ = run_single_episode(
                    env, states[episode], description, model, processor, cfg, episode,
                    action_horizon=horizon, input_w=width, input_h=height, model_device=args.device)
            summary = summarize_replans(recorder.rows, attention_sources=args.attention_sources)
            save_plots(directory, summary, head_reduction=args.head_reduction,
                       attention_sources=args.attention_sources)
            # Standard rollout video and labeled replan start images link scores to task stages.
            save_rollout_video(directory, replay, f"episode{episode}", success, description)
            timeline = []
            for replan in range(recorder.replan + 1):
                rollout_step = replan * min(replan_steps, horizon)
                images = replay[rollout_step]
                frame = np.concatenate([np.asarray(img) for img in images.values()], axis=1)
                filename = f"replan_{replan:03d}.png"
                Image.fromarray(frame).save(directory / filename)
                timeline.append(dict(replan=replan, rollout_step=rollout_step,
                                     env_step=rollout_step + int(cfg.EVALUATION.num_steps_wait), image=filename))
            (directory / "timeline.json").write_text(json.dumps(timeline, indent=2), encoding="utf-8")
            outcome = dict(episode=episode, initial_state_id=episode, success=bool(success),
                           executed_steps=len(replay), num_replans=recorder.replan+1, task_description=description)
            (directory / "episode_result.json").write_text(json.dumps(outcome, indent=2), encoding="utf-8")
            outcomes.append(outcome)
            (out / "episodes.json").write_text(json.dumps(outcomes, indent=2), encoding="utf-8")
            print(f"Saved episode {episode}: success={success}, replans={recorder.replan+1}, {directory}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
