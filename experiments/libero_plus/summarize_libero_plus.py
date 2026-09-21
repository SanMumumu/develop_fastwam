"""Aggregate a LIBERO-Plus sweep into the seven-axis report.

    python experiments/libero_plus/summarize_libero_plus.py \
      --results-dir evaluate_results/libero_plus/<run_id> \
      [--libero-plus-root /opt/LIBERO-plus] [--require-complete]

Writes ``libero_plus_summary.json`` next to the results and prints the table.
Safe to run on a partial sweep: ``official_protocol`` is false until all 10,030
tasks are present, and the missing counts are reported per suite.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.libero_plus.libero_plus_protocol import (  # noqa: E402
    format_report,
    load_task_classification,
    summarize,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument(
        "--libero-plus-root",
        default=os.environ.get("LIBERO_PLUS_ROOT", "/opt/LIBERO-plus"),
        help="LIBERO-Plus checkout holding benchmark/task_classification.json.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit non-zero unless all 10,030 tasks are present and consistent.",
    )
    args = parser.parse_args()

    classification = load_task_classification(args.libero_plus_root)
    summary = summarize(
        args.results_dir, classification, require_complete=args.require_complete
    )
    output = Path(args.output or Path(args.results_dir) / "libero_plus_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(format_report(summary))
    print(f"\nwrote {output}")
    return 0 if summary["official_protocol"] or not args.require_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
