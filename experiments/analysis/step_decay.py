"""Sweep the number of denoising steps at evaluation time.

Two different step counts are worth sweeping, and they answer different
questions:

``--axis action`` (``EVALUATION.num_inference_steps``)
    how many action denoising steps the policy needs.

``--axis dream`` (``model.dream_scheduler.inference_steps``)
    how many *imagination* steps the policy needs. This is the one that sizes
    the headroom for interface distillation: if success is already flat at one
    Dream step, distillation buys nothing and should not be claimed; if it
    collapses, the gap is exactly what distillation has to recover.

The sweep reuses an existing evaluation launcher (configured via the
``$LIBERO_EVAL_LAUNCHER`` / ``$LIBERO_PLUS_EVAL_LAUNCHER`` environment
variables) -- one output directory per setting, so every run stays
independently resumable -- and then tabulates the summaries.

    python experiments/analysis/step_decay.py \
      --benchmark libero_plus --axis dream --values 8 4 2 1 \
      --task-config routed_wam_libero_goal --ckpt <path> --suites libero_goal

Use ``--dry-run`` first: this launches full evaluation sweeps.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Evaluation sweep drivers. The cluster-specific launchers used to produce the
# numbers in `docs/EXPERIMENT_REPORT.md` are site-local and not part of this
# repository, so point these at your own via the environment. A launcher must
# honour the TASK_CONFIG / CKPT / DATASET_STATS / OUTPUT_DIR / NUM_GPUS /
# MAX_TASKS_PER_GPU / SUITES / EXTRA_ARGS contract assembled in `run_one`, and
# write one summary JSON per output directory.
LAUNCHER_ENV_VARS = {
    "libero": "LIBERO_EVAL_LAUNCHER",
    "libero_plus": "LIBERO_PLUS_EVAL_LAUNCHER",
}


def resolve_launcher(benchmark: str) -> Path:
    """Return the sweep launcher for ``benchmark`` or explain what is missing."""
    env_var = LAUNCHER_ENV_VARS[benchmark]
    configured = os.environ.get(env_var)
    if not configured:
        raise SystemExit(
            f"No evaluation launcher configured for `{benchmark}`. Set ${env_var} to a "
            "script accepting the TASK_CONFIG / CKPT / DATASET_STATS / OUTPUT_DIR / "
            "NUM_GPUS / MAX_TASKS_PER_GPU / SUITES / EXTRA_ARGS environment contract."
        )
    launcher = Path(configured).expanduser()
    if not launcher.is_file():
        raise SystemExit(f"${env_var} points at {launcher}, which is not a file.")
    return launcher

AXIS_OVERRIDE = {
    # Action denoising steps: an EVALUATION-level knob read by the worker.
    "action": "EVALUATION.num_inference_steps={value}",
    # Dream denoising steps: a model-level knob, so it must reach Hydra as a
    # model override rather than an EVALUATION one.
    "dream": "model.dream_scheduler.inference_steps={value}",
}


def run_one(
    *,
    benchmark: str,
    axis: str,
    value: int,
    task_config: str,
    ckpt: str,
    dataset_stats: str | None,
    suites: list[str],
    output_root: Path,
    num_gpus: int,
    tasks_per_gpu: int,
    num_trials: int | None,
    dry_run: bool,
) -> Path:
    output_dir = output_root / f"{axis}_{value}"
    env = dict(os.environ)
    env.update(
        {
            "TASK_CONFIG": task_config,
            "CKPT": ckpt,
            "OUTPUT_DIR": str(output_dir),
            "NUM_GPUS": str(num_gpus),
            "MAX_TASKS_PER_GPU": str(tasks_per_gpu),
            "SUITES": " ".join(suites),
            "EXTRA_ARGS": AXIS_OVERRIDE[axis].format(value=value),
        }
    )
    if dataset_stats:
        env["DATASET_STATS"] = dataset_stats
    if num_trials is not None and benchmark == "libero":
        # LIBERO-Plus pins one trial per task; only standard LIBERO takes this.
        env["NUM_TRIALS"] = str(num_trials)

    command = ["bash", str(resolve_launcher(benchmark))]
    print(f"[step-decay] {axis}={value} -> {output_dir}")
    print(f"             EXTRA_ARGS={env['EXTRA_ARGS']}")
    if dry_run:
        return output_dir
    subprocess.run(command, env=env, cwd=str(_REPO_ROOT), check=False)
    return output_dir


def collect(benchmark: str, output_dir: Path, libero_plus_root: str) -> dict[str, Any]:
    if benchmark == "libero_plus":
        from experiments.libero_plus.libero_plus_protocol import (
            load_task_classification,
            summarize,
        )

        summary = summarize(output_dir, load_task_classification(libero_plus_root))
        return {
            "headline": summary["perturbation_average_success_rate"],
            "weighted": summary["weighted_success_rate"],
            "evaluated_tasks": summary["evaluated_tasks"],
            "per_category": {
                name: bucket["success_rate"]
                for name, bucket in summary["category_results"].items()
            },
        }

    successes = episodes = 0
    for path in sorted(output_dir.rglob("*_task*_results.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        successes += int(payload["successes"])
        episodes += int(payload.get("total_episodes", 0))
    return {
        "headline": None if episodes == 0 else 100.0 * successes / episodes,
        "weighted": None if episodes == 0 else 100.0 * successes / episodes,
        "evaluated_tasks": len(list(output_dir.rglob("*_task*_results.json"))),
        "per_category": {},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=sorted(LAUNCHER_ENV_VARS), default="libero_plus")
    parser.add_argument("--axis", choices=sorted(AXIS_OVERRIDE), default="dream")
    parser.add_argument("--values", type=int, nargs="+", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset-stats", default=None)
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--tasks-per-gpu", type=int, default=2)
    parser.add_argument("--num-trials", type=int, default=None)
    parser.add_argument(
        "--libero-plus-root", default=os.environ.get("LIBERO_PLUS_ROOT", "/opt/LIBERO-plus")
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    output_root = Path(
        args.output_root
        or _REPO_ROOT / "evaluate_results" / f"step_decay_{args.benchmark}" / args.task_config
    )
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for value in args.values:
        output_dir = run_one(
            benchmark=args.benchmark,
            axis=args.axis,
            value=value,
            task_config=args.task_config,
            ckpt=args.ckpt,
            dataset_stats=args.dataset_stats,
            suites=args.suites,
            output_root=output_root,
            num_gpus=args.num_gpus,
            tasks_per_gpu=args.tasks_per_gpu,
            num_trials=args.num_trials,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            continue
        rows.append({"value": value, "output_dir": str(output_dir),
                     **collect(args.benchmark, output_dir, args.libero_plus_root)})

    if args.dry_run:
        print("\ndry run: nothing was launched.")
        return 0

    print(f"\n{args.axis} steps | success rate | tasks")
    for row in rows:
        headline = "--" if row["headline"] is None else f"{row['headline']:.2f}"
        print(f"{row['value']:>11} | {headline:>12} | {row['evaluated_tasks']}")

    payload = {"benchmark": args.benchmark, "axis": args.axis, "rows": rows}
    output = Path(args.json or output_root / "step_decay.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
