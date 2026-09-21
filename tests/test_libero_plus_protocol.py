"""Tests for the LIBERO-Plus evaluation protocol.

No simulator and no LIBERO-Plus checkout are needed: the classification table is
synthesised with the official shape, and the aggregator is fed hand-written
result JSONs.

The properties that matter here are arithmetic ones. LIBERO-Plus reports the
*unweighted mean over the seven perturbation categories*, and the categories
differ in size by more than 1.5x (1076 vs 1601), so quietly reporting the pooled
rate instead would produce a number that is wrong in a way no reader could
detect.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.libero_plus.libero_plus_protocol import (  # noqa: E402
    LIBERO_PLUS_PERTURBATION_COUNTS,
    LIBERO_PLUS_PERTURBATIONS,
    LIBERO_PLUS_SUITES,
    LIBERO_PLUS_TASK_COUNTS,
    TRIALS_PER_TASK,
    format_report,
    load_task_classification,
    summarize,
)


def test_pinned_protocol_constants_are_self_consistent():
    assert sum(LIBERO_PLUS_TASK_COUNTS.values()) == 10030
    assert sum(LIBERO_PLUS_PERTURBATION_COUNTS.values()) == 10030
    assert set(LIBERO_PLUS_TASK_COUNTS) == set(LIBERO_PLUS_SUITES)
    assert {short for short, _ in LIBERO_PLUS_PERTURBATIONS} == set(
        LIBERO_PLUS_PERTURBATION_COUNTS
    )


def _write_classification(root: Path, *, break_count: bool = False) -> None:
    """Synthesise a task_classification.json with the official shape."""
    long_names = [long for _, long in LIBERO_PLUS_PERTURBATIONS]
    remaining = dict(LIBERO_PLUS_PERTURBATION_COUNTS)
    payload: dict[str, list[dict]] = {}
    for suite in LIBERO_PLUS_SUITES:
        entries = []
        for index in range(LIBERO_PLUS_TASK_COUNTS[suite]):
            for short, long in LIBERO_PLUS_PERTURBATIONS:
                if remaining[short] > 0:
                    remaining[short] -= 1
                    category = long
                    break
            else:  # pragma: no cover - the totals match by construction
                category = long_names[0]
            entries.append(
                {
                    "id": index + 1,
                    "name": f"{suite}_task_{index}",
                    "category": category,
                    "difficulty_level": None if index % 7 == 0 else index % 5,
                }
            )
        if break_count and suite == "libero_goal":
            entries.pop()
        payload[suite] = entries

    path = root / "libero" / "libero" / "benchmark"
    path.mkdir(parents=True, exist_ok=True)
    (path / "task_classification.json").write_text(json.dumps(payload), encoding="utf-8")


def test_load_task_classification_accepts_the_official_shape(tmp_path):
    _write_classification(tmp_path)
    table = load_task_classification(tmp_path)
    assert {suite: len(entries) for suite, entries in table.items()} == LIBERO_PLUS_TASK_COUNTS
    # Ids are rebased to 0 so they index Benchmark.get_task(i) directly.
    assert 0 in table["libero_goal"]
    assert LIBERO_PLUS_TASK_COUNTS["libero_goal"] - 1 in table["libero_goal"]
    assert table["libero_goal"][0]["difficulty_level"] is None


def test_load_task_classification_rejects_a_wrong_task_count(tmp_path):
    _write_classification(tmp_path, break_count=True)
    with pytest.raises(ValueError, match="task_classification.json lists"):
        load_task_classification(tmp_path)


def _make_table(spec: dict[str, list[str]]) -> dict[str, dict[int, dict]]:
    """spec: suite -> list of category short names, one per task id."""
    long_by_short = {short: long for short, long in LIBERO_PLUS_PERTURBATIONS}
    return {
        suite: {
            task_id: {
                "name": f"{suite}_{task_id}",
                "category": short,
                "category_long": long_by_short[short],
                "difficulty_level": 1,
            }
            for task_id, short in enumerate(categories)
        }
        for suite, categories in spec.items()
    }


def _write_results(root: Path, suite: str, outcomes: list[int], episodes: int = TRIALS_PER_TASK):
    suite_dir = root / suite
    suite_dir.mkdir(parents=True, exist_ok=True)
    for task_id, successes in enumerate(outcomes):
        (suite_dir / f"gpu0_task{task_id}_results.json").write_text(
            json.dumps(
                {
                    "task_suite": suite,
                    "task_id": task_id,
                    "successes": successes,
                    "total_episodes": episodes,
                }
            ),
            encoding="utf-8",
        )


def test_summarize_reports_pooled_and_per_category_rates_separately(tmp_path):
    # Camera has 3 tasks (1 success), Light has 1 task (1 success).
    table = _make_table({"libero_goal": ["Camera", "Camera", "Camera", "Light"]})
    _write_results(tmp_path, "libero_goal", [1, 0, 0, 1])

    summary = summarize(tmp_path, table)
    assert summary["category_results"]["Camera"]["success_rate"] == pytest.approx(100 / 3)
    assert summary["category_results"]["Light"]["success_rate"] == pytest.approx(100.0)
    # Pooled: 2/4 = 50%. Unweighted over the two present categories: 66.67%.
    assert summary["weighted_success_rate"] == pytest.approx(50.0)
    assert summary["perturbation_average_success_rate"] == pytest.approx(
        (100 / 3 + 100.0) / 2
    )
    assert summary["category_results"]["Noise"]["success_rate"] is None
    # An incomplete sweep must never claim to be the official protocol.
    assert summary["official_protocol"] is False
    assert summary["evaluated_tasks"] == 4


def test_summarize_flags_duplicates_and_unknown_tasks(tmp_path):
    table = _make_table({"libero_goal": ["Camera", "Light"]})
    _write_results(tmp_path, "libero_goal", [1, 1])
    # A second file for task 0, as a retry on another GPU would produce.
    (tmp_path / "libero_goal" / "gpu3_task0_results.json").write_text(
        json.dumps(
            {"task_suite": "libero_goal", "task_id": 0, "successes": 0, "total_episodes": 1}
        ),
        encoding="utf-8",
    )
    # A task id the classification table does not know about.
    (tmp_path / "libero_goal" / "gpu0_task99_results.json").write_text(
        json.dumps(
            {"task_suite": "libero_goal", "task_id": 99, "successes": 1, "total_episodes": 1}
        ),
        encoding="utf-8",
    )
    summary = summarize(tmp_path, table)
    assert summary["evaluated_tasks"] == 2
    assert any("duplicate" in note for note in summary["anomalies"])
    assert any("not present" in note for note in summary["anomalies"])
    # The duplicate must not be double-counted.
    assert summary["weighted_success_rate"] == pytest.approx(100.0)


def test_summarize_flags_a_non_official_trial_count(tmp_path):
    table = _make_table({"libero_goal": ["Camera"]})
    _write_results(tmp_path, "libero_goal", [3], episodes=5)
    summary = summarize(tmp_path, table)
    assert any("official protocol is" in note for note in summary["anomalies"])
    assert summary["weighted_success_rate"] == pytest.approx(60.0)


def test_summarize_can_require_completeness(tmp_path):
    table = _make_table({"libero_goal": ["Camera"]})
    _write_results(tmp_path, "libero_goal", [1])
    with pytest.raises(RuntimeError, match="Incomplete LIBERO-Plus sweep"):
        summarize(tmp_path, table, require_complete=True)


def test_format_report_renders_all_seven_axes(tmp_path):
    table = _make_table({"libero_goal": ["Camera", "Light"]})
    _write_results(tmp_path, "libero_goal", [1, 0])
    text = format_report(summarize(tmp_path, table))
    for short, _ in LIBERO_PLUS_PERTURBATIONS:
        assert short in text
    assert "evaluated 2/10030 tasks" in text
