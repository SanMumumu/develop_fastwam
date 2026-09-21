"""Single-task LIBERO-Plus worker.

Deliberately thin.  Everything that differs from standard LIBERO evaluation is
either (a) which `libero` package is on the path, or (b) protocol bookkeeping
that belongs to the summariser, so the model-side pipeline in
``experiments/libero/eval_libero_single.py`` is reused verbatim rather than
forked.

Two ordering constraints make this file look unusual, and both are load-bearing:

* ``configure_libero_plus`` must run **before** ``eval_libero_single`` is
  imported, because that module imports ``libero.libero`` at module scope and
  the package resolves ``LIBERO_CONFIG_PATH`` at import time.  Importing first
  would silently evaluate the *unperturbed* LIBERO tasks and produce numbers
  that look reasonable and mean nothing.
* the init-state fallback must be installed before the benchmark is constructed.

Usage (the sweep script sets these):

    LIBERO_PLUS_ROOT=/opt/LIBERO-plus \
    python experiments/libero_plus/eval_libero_plus_single.py \
      task=routed_wam_libero_4suite ckpt=... gpu_id=0 \
      EVALUATION.task_suite_name=libero_goal EVALUATION.task_id=0 \
      EVALUATION.num_trials=1 EVALUATION.output_dir=...
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from experiments.libero_plus.libero_plus_protocol import (  # noqa: E402
    TRIALS_PER_TASK,
    assert_libero_plus_is_active,
    configure_libero_plus,
    install_init_state_fallback,
)

LIBERO_PLUS_ROOT = os.environ.get("LIBERO_PLUS_ROOT", "/opt/LIBERO-plus")
LIBERO_PLUS_CONFIG_ROOT = os.environ.get(
    "LIBERO_PLUS_CONFIG_ROOT", str(Path.home() / ".libero_plus_fastwam")
)

# --- must precede the eval_libero_single import; see the module docstring ---
configure_libero_plus(
    repository_root=LIBERO_PLUS_ROOT,
    config_root=LIBERO_PLUS_CONFIG_ROOT,
    dataset_root=os.environ.get("LIBERO_PLUS_DATA_ROOT") or None,
    validate_noise_operators=os.environ.get("LIBERO_PLUS_SKIP_NOISE_CHECK", "0") != "1",
)
assert_libero_plus_is_active(LIBERO_PLUS_ROOT)

from experiments.libero import eval_libero_single as base_eval  # noqa: E402

install_init_state_fallback()


def main() -> None:
    if "EVALUATION.num_trials" not in " ".join(sys.argv):
        # The official protocol is one rollout per task; more would change the
        # meaning of every aggregate without changing its name.
        sys.argv.append(f"EVALUATION.num_trials={TRIALS_PER_TASK}")
    base_eval.eval_single_process()


if __name__ == "__main__":
    main()
