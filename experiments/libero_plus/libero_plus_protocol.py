"""LIBERO-Plus evaluation protocol: paths, task classification and aggregation.

LIBERO-Plus is the perturbation-robustness superset of LIBERO: the same four
suites, expanded to 10,030 tasks, each carrying one of seven perturbation
categories, evaluated with exactly one trial per task.

This module is deliberately free of any model or policy dependency so it can be
imported by the worker, by the summariser and by tests.  It does three things:

1. point the `libero` package at a LIBERO-Plus checkout (`configure_libero_plus`)
   -- the fork ships its own bddl files, init states and assets, and importing
   the stock LIBERO by accident would silently evaluate the unperturbed tasks;
2. load the task -> perturbation-category table;
3. aggregate per-task result JSONs into the two numbers the benchmark reports:
   the pooled success rate and the unweighted mean over the seven categories.

The protocol constants are pinned and cross-checked against the benchmark's own
files at run time, because a silent mismatch here produces a number that looks
plausible and is not comparable with anything published.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

LIBERO_PLUS_SUITES: tuple[str, ...] = (
    "libero_10",
    "libero_goal",
    "libero_spatial",
    "libero_object",
)

#: Task counts per suite in the official 10,030-task protocol.
LIBERO_PLUS_TASK_COUNTS: dict[str, int] = {
    "libero_10": 2519,
    "libero_goal": 2591,
    "libero_spatial": 2402,
    "libero_object": 2518,
}

#: (short name, category string as written in task_classification.json)
LIBERO_PLUS_PERTURBATIONS: tuple[tuple[str, str], ...] = (
    ("Camera", "Camera Viewpoints"),
    ("Robot", "Robot Initial States"),
    ("Language", "Language Instructions"),
    ("Light", "Light Conditions"),
    ("Background", "Background Textures"),
    ("Noise", "Sensor Noise"),
    ("Layout", "Objects Layout"),
)

LIBERO_PLUS_PERTURBATION_COUNTS: dict[str, int] = {
    "Camera": 1599,
    "Robot": 1550,
    "Language": 1537,
    "Light": 1142,
    "Background": 1076,
    "Noise": 1601,
    "Layout": 1525,
}

#: Episode budget per suite. Identical to standard LIBERO, so
#: `experiments/libero/eval_libero_single.py::_get_max_steps` needs no patch.
LIBERO_PLUS_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 400,
    "libero_object": 400,
    "libero_goal": 400,
    "libero_10": 700,
    "libero_90": 700,
}

TRIALS_PER_TASK = 1

_CATEGORY_BY_LONG_NAME = {long: short for short, long in LIBERO_PLUS_PERTURBATIONS}


# --------------------------------------------------------------------- setup
def configure_libero_plus(
    *,
    repository_root: str | Path,
    config_root: str | Path,
    dataset_root: Optional[str | Path] = None,
    validate_noise_operators: bool = True,
) -> Path:
    """Point the `libero` import at a LIBERO-Plus checkout.

    Must run **before** anything imports `libero.libero`, because the package
    reads `LIBERO_CONFIG_PATH` at import time.

    Returns the benchmark root (``<repo>/libero/libero``).
    """
    repository_root = Path(repository_root).resolve()
    package_root = repository_root / "libero"
    benchmark_root = package_root / "libero"
    required = {
        "package": package_root,
        "bddl_files": benchmark_root / "bddl_files",
        "init_states": benchmark_root / "init_files",
        "assets": benchmark_root / "assets",
    }
    classification_path = benchmark_root / "benchmark" / "task_classification.json"
    missing = {name: str(path) for name, path in required.items() if not path.is_dir()}
    if missing or not classification_path.is_file():
        raise FileNotFoundError(
            "Incomplete LIBERO-Plus installation at "
            f"{repository_root}: missing_directories={missing}, "
            f"task_classification={classification_path}"
        )

    _preload_noise_dependencies()

    config_root = Path(config_root).resolve()
    config_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(required["bddl_files"]),
        "init_states": str(required["init_states"]),
        "datasets": str(Path(dataset_root).resolve()) if dataset_root else str(benchmark_root / "datasets"),
        "assets": str(required["assets"]),
    }
    import yaml

    rendered = yaml.safe_dump(payload, sort_keys=False)
    config_path = config_root / "config.yaml"
    if config_path.exists() and config_path.read_text(encoding="utf-8") != rendered:
        config_path.write_text(rendered, encoding="utf-8")
    elif not config_path.exists():
        config_path.write_text(rendered, encoding="utf-8")

    os.environ["LIBERO_CONFIG_PATH"] = str(config_root)
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    if validate_noise_operators:
        _validate_noise_operators()
    return benchmark_root


def assert_libero_plus_is_active(repository_root: str | Path) -> None:
    """Fail loudly if the imported `libero` is not the LIBERO-Plus checkout."""
    import inspect

    import libero.libero as libero_core

    package_path = Path(inspect.getfile(libero_core)).resolve()
    repository_root = Path(repository_root).resolve()
    if not str(package_path).startswith(str(repository_root)):
        raise RuntimeError(
            "LIBERO-Plus evaluation imported the wrong package: "
            f"{package_path} is not inside {repository_root}. "
            "configure_libero_plus() must run before any libero import."
        )


def _preload_noise_dependencies() -> None:
    """The Sensor Noise axis needs scikit-image and a loadable MagickWand."""
    try:
        from skimage.filters import gaussian as _gaussian  # noqa: F401
        from wand.api import library as _wand_library  # noqa: F401
    except (ImportError, OSError) as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "LIBERO-Plus sensor-noise evaluation requires scikit-image, Wand and a "
            "loadable ImageMagick MagickWand runtime. See docs/LIBERO_PLUS_SETUP.md."
        ) from exc


def _validate_noise_operators() -> None:
    """Run the corruption operators once; a broken ImageMagick only fails here."""
    import numpy as np
    from PIL import Image

    from libero.libero.envs.env_wrapper import gaussian_blur, motion_blur

    image = Image.fromarray(np.full((224, 224, 3), 127, dtype=np.uint8))
    gaussian_blur(image, severity=1)
    motion_blur(image, severity=1)


# ------------------------------------------------------------- classification
def load_task_classification(repository_root: str | Path) -> dict[str, dict[int, dict[str, Any]]]:
    """task_classification.json -> ``{suite: {task_id: {name, category, level}}}``.

    Task ids are 0-based to match ``Benchmark.get_task(i)``; the JSON numbers
    them from 1.
    """
    path = (
        Path(repository_root).resolve()
        / "libero"
        / "libero"
        / "benchmark"
        / "task_classification.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing LIBERO-Plus task classification: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("LIBERO-Plus task classification must be a JSON object.")

    table: dict[str, dict[int, dict[str, Any]]] = {}
    for suite in LIBERO_PLUS_SUITES:
        if suite not in payload:
            raise KeyError(f"task_classification.json has no entry for suite {suite!r}.")
        entries = payload[suite]
        if len(entries) != LIBERO_PLUS_TASK_COUNTS[suite]:
            raise ValueError(
                f"{suite}: task_classification.json lists {len(entries)} tasks, "
                f"the official protocol has {LIBERO_PLUS_TASK_COUNTS[suite]}."
            )
        suite_table: dict[int, dict[str, Any]] = {}
        for entry in entries:
            category = str(entry["category"])
            if category not in _CATEGORY_BY_LONG_NAME:
                raise ValueError(f"Unknown perturbation category {category!r} in {suite}.")
            # `difficulty_level` is null for a subset of the real entries.
            level = entry.get("difficulty_level")
            suite_table[int(entry["id"]) - 1] = {
                "name": str(entry["name"]),
                "category": _CATEGORY_BY_LONG_NAME[category],
                "category_long": category,
                "difficulty_level": None if level is None else int(level),
            }
        table[suite] = suite_table

    observed = defaultdict(int)
    for suite_table in table.values():
        for record in suite_table.values():
            observed[record["category"]] += 1
    mismatched = {
        name: (observed[name], expected)
        for name, expected in LIBERO_PLUS_PERTURBATION_COUNTS.items()
        if observed[name] != expected
    }
    if mismatched:
        raise ValueError(
            "Perturbation-category counts do not match the official protocol "
            f"(observed, expected): {mismatched}"
        )
    return table


# --------------------------------------------------------------- init states
def resolve_init_states_path(task) -> Path:
    """Locate a task's init-state file, following the fork's naming fallbacks.

    Perturbed variants reuse their base task's init states, and the filename
    encodes the perturbation. The fork's own `Benchmark.get_task_init_states`
    handles most of these, but not all; this is the complete list, tried in
    order of specificity.
    """
    from libero.libero import get_libero_path

    root = Path(get_libero_path("init_states"))
    filename = str(task.init_states_file)
    suite_root = root / task.problem_folder
    suffix = Path(filename).suffix
    candidates = [suite_root / filename]
    if "_language_" in filename:
        candidates.append(suite_root / f"{filename.split('_language_')[0]}{suffix}")
    if "_view_" in filename:
        candidates.append(suite_root / f"{filename.split('_view_')[0]}{suffix}")
    if "_light_" in filename:
        candidates.append(suite_root / f"{filename.split('_light_')[0]}{suffix}")
    if "_table_" in filename:
        candidates.append(suite_root / re.sub(r"_table_\d+", "", filename))
    if "_tb_" in filename:
        candidates.append(suite_root / re.sub(r"_tb_\d+", "", filename))
    if "_add_" in filename or "_level" in filename:
        candidates.append(root / "libero_newobj" / task.problem_folder / filename)
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Cannot resolve init states for task {task.name!r}; tried "
        f"{[str(path) for path in candidates]}"
    )


def install_init_state_fallback() -> None:
    """Wrap `Benchmark.get_task_init_states` with the complete fallback list.

    The fork's own resolver covers most naming variants; wrapping rather than
    replacing keeps its behaviour wherever it already succeeds.
    """
    import inspect

    import torch
    from libero.libero.benchmark import Benchmark

    if getattr(Benchmark, "_fastwam_init_state_fallback", False):
        return
    original = Benchmark.get_task_init_states

    def patched(self, i):
        try:
            return original(self, i)
        except (FileNotFoundError, OSError):
            load_kwargs = {}
            if "weights_only" in inspect.signature(torch.load).parameters:
                load_kwargs["weights_only"] = False
            return torch.load(resolve_init_states_path(self.get_task(i)), **load_kwargs)

    Benchmark.get_task_init_states = patched
    Benchmark._fastwam_init_state_fallback = True


# ---------------------------------------------------------------- aggregation
def iter_result_files(results_dir: str | Path) -> Iterable[Path]:
    results_dir = Path(results_dir)
    for suite in LIBERO_PLUS_SUITES:
        suite_dir = results_dir / suite
        if not suite_dir.is_dir():
            continue
        yield from sorted(suite_dir.glob("*_task*_results.json"))


def summarize(
    results_dir: str | Path,
    classification: dict[str, dict[int, dict[str, Any]]],
    *,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Aggregate per-task result JSONs into the LIBERO-Plus report.

    Two headline numbers are produced, and they are not interchangeable:

    ``weighted_success_rate``
        successes / episodes pooled over all tasks.
    ``perturbation_average_success_rate``
        the unweighted mean over the seven categories, which is what the
        LIBERO-Plus tables report. Category sizes differ by 1.5x, so the two
        numbers can differ by several points.
    """
    per_suite: dict[str, dict[str, int]] = defaultdict(lambda: {"successes": 0, "episodes": 0, "tasks": 0})
    per_category: dict[str, dict[str, int]] = defaultdict(lambda: {"successes": 0, "episodes": 0, "tasks": 0})
    per_suite_category: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"successes": 0, "episodes": 0, "tasks": 0})
    )
    seen: dict[str, set[int]] = defaultdict(set)
    anomalies: list[str] = []

    for path in iter_result_files(results_dir):
        payload = json.loads(path.read_text(encoding="utf-8"))
        suite = str(payload["task_suite"])
        task_id = int(payload["task_id"])
        successes = int(payload["successes"])
        episodes = int(payload.get("total_episodes", TRIALS_PER_TASK))
        if suite not in classification:
            anomalies.append(f"{path}: unknown suite {suite!r}")
            continue
        if task_id in seen[suite]:
            anomalies.append(f"{suite} task {task_id}: duplicate result file {path}")
            continue
        record = classification[suite].get(task_id)
        if record is None:
            anomalies.append(f"{suite} task {task_id}: not present in task_classification.json")
            continue
        if episodes != TRIALS_PER_TASK:
            anomalies.append(
                f"{suite} task {task_id}: {episodes} episodes, the official protocol is "
                f"{TRIALS_PER_TASK}"
            )
        seen[suite].add(task_id)
        category = record["category"]
        for bucket in (per_suite[suite], per_category[category], per_suite_category[suite][category]):
            bucket["successes"] += successes
            bucket["episodes"] += episodes
            bucket["tasks"] += 1

    def rate(bucket: dict[str, int]) -> Optional[float]:
        return None if bucket["episodes"] == 0 else 100.0 * bucket["successes"] / bucket["episodes"]

    category_rates = {name: rate(per_category[name]) for name, _ in LIBERO_PLUS_PERTURBATIONS}
    present = [value for value in category_rates.values() if value is not None]
    total_successes = sum(bucket["successes"] for bucket in per_suite.values())
    total_episodes = sum(bucket["episodes"] for bucket in per_suite.values())
    evaluated = sum(len(ids) for ids in seen.values())
    expected_total = sum(LIBERO_PLUS_TASK_COUNTS.values())

    missing_per_suite = {
        suite: LIBERO_PLUS_TASK_COUNTS[suite] - len(seen[suite]) for suite in LIBERO_PLUS_SUITES
    }
    complete = evaluated == expected_total and not anomalies
    if require_complete and not complete:
        raise RuntimeError(
            f"Incomplete LIBERO-Plus sweep: evaluated {evaluated}/{expected_total}, "
            f"missing_per_suite={missing_per_suite}, anomalies={anomalies[:10]}"
        )

    return {
        "official_protocol": complete,
        "evaluated_tasks": evaluated,
        "expected_tasks": expected_total,
        "missing_per_suite": missing_per_suite,
        "weighted_success_rate": None if total_episodes == 0 else 100.0 * total_successes / total_episodes,
        "perturbation_average_success_rate": None if not present else sum(present) / len(present),
        "suite_average_success_rate": (
            None
            if not [r for r in (rate(per_suite[s]) for s in LIBERO_PLUS_SUITES) if r is not None]
            else sum(r for r in (rate(per_suite[s]) for s in LIBERO_PLUS_SUITES) if r is not None)
            / len([r for r in (rate(per_suite[s]) for s in LIBERO_PLUS_SUITES) if r is not None])
        ),
        "category_results": {
            name: {
                **per_category[name],
                "success_rate": category_rates[name],
                "expected_tasks": LIBERO_PLUS_PERTURBATION_COUNTS[name],
            }
            for name, _ in LIBERO_PLUS_PERTURBATIONS
        },
        "suite_results": {
            suite: {**per_suite[suite], "success_rate": rate(per_suite[suite])}
            for suite in LIBERO_PLUS_SUITES
        },
        "suite_category_results": {
            suite: {
                name: {**bucket, "success_rate": rate(bucket)}
                for name, bucket in per_suite_category[suite].items()
            }
            for suite in LIBERO_PLUS_SUITES
        },
        "anomalies": anomalies,
    }


def format_report(summary: dict[str, Any]) -> str:
    """Render the seven-axis table the LIBERO-Plus papers report."""
    lines = []
    header = " | ".join(f"{name:>10}" for name, _ in LIBERO_PLUS_PERTURBATIONS)
    lines.append(f"{'':>14}{header} | {'Avg':>10}")
    values = []
    for name, _ in LIBERO_PLUS_PERTURBATIONS:
        rate = summary["category_results"][name]["success_rate"]
        values.append(f"{'--':>10}" if rate is None else f"{rate:>10.2f}")
    average = summary["perturbation_average_success_rate"]
    average_text = "--" if average is None else f"{average:.2f}"
    lines.append(f"{'success rate':>14}{' | '.join(values)} | {average_text:>10}")
    lines.append("")
    lines.append(
        f"evaluated {summary['evaluated_tasks']}/{summary['expected_tasks']} tasks; "
        f"official_protocol={summary['official_protocol']}; "
        f"weighted={summary['weighted_success_rate']}"
    )
    if summary["anomalies"]:
        lines.append(f"anomalies ({len(summary['anomalies'])}): {summary['anomalies'][:5]}")
    return "\n".join(lines)
