"""Trainer that supplies optimizer-step progress to the router and distiller.

Deliberately as small as :class:`~fastwam.threshold_trainer.ThresholdWan22Trainer`:
the router warmup, the distillation ramp and the EMA teacher update are all
driven from the progress callback consumed inside ``RoutedWAM.training_loss``,
so the base training loop needs no modification.

It also mirrors every logged scalar into TensorBoard event files.  This repository
only ever had wandb, which is run `offline` on the cluster -- so the router curves
were only readable by scraping the text log, and the text log wraps its metrics
across physical lines (see ``experiments/analysis/parse_train_log.py``).  Writing
events to the run directory, which lives on the shared bucket, makes the curves
openable with a plain ``tensorboard --logdir`` *while the job is still running*,
and keeps working after the job ages out of the cluster's own TensorBoard window.
"""

from __future__ import annotations

import os
from pathlib import Path

from .trainer import Wan22Trainer
from .utils.logging_config import get_logger


logger = get_logger(__name__)


class RoutedWan22Trainer(Wan22Trainer):
    """Wan22Trainer wired to a RoutedWAM's progress-dependent machinery."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model = self.accelerator.unwrap_model(self.model)
        if not hasattr(model, "set_training_progress_provider"):
            raise TypeError("RoutedWan22Trainer requires a RoutedWAM.")
        model.set_training_progress_provider(
            lambda: (
                self.global_step,
                self.max_steps,
                bool(self.log_every > 0 and (self.global_step + 1) % self.log_every == 0),
            )
        )
        self._tb_writer = None
        self._setup_tensorboard()

    # ------------------------------------------------------------ tensorboard
    def _setup_tensorboard(self) -> None:
        """Open a SummaryWriter on the main process, if torch ships one.

        TensorBoard is treated as strictly optional: a missing `tensorboard`
        package must not take a 26-hour training job down with it, so an import
        failure is logged and training continues with the text log as before.
        """
        if not self.accelerator.is_main_process:
            return
        if os.environ.get("FASTWAM_DISABLE_TENSORBOARD", "").lower() in {"1", "true", "yes"}:
            logger.info("TensorBoard logging disabled by FASTWAM_DISABLE_TENSORBOARD.")
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning(
                "TensorBoard unavailable (%s); router curves will only be in the text log. "
                "Install `tensorboard` to get event files.",
                exc,
            )
            return
        log_dir = Path(os.environ.get("FASTWAM_TB_DIR") or (Path(self.cfg.output_dir) / "tb"))
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=str(log_dir))
        except Exception as exc:  # pragma: no cover - bucket/permission dependent
            logger.warning("Could not open TensorBoard writer at %s: %s", log_dir, exc)
            return
        logger.info("TensorBoard events -> %s  (tensorboard --logdir %s)", log_dir, log_dir)

    def _wandb_log(self, payload: dict):
        """Mirror the metric payload into TensorBoard, then log to wandb as usual.

        `_wandb_log` is the single funnel every logged scalar already passes
        through, so hooking it keeps the two sinks in lockstep -- including the
        router statistics, which is the whole point.
        """
        if self._tb_writer is not None:
            for key, value in payload.items():
                # Router group statistics arrive as `router_keep_dino@t1/wrist`;
                # TensorBoard reads `/` as hierarchy, which is what we want, but
                # `@` is left alone so the tag still matches the log text.
                try:
                    self._tb_writer.add_scalar(key, float(value), self.global_step)
                except (TypeError, ValueError):
                    # Non-scalar entries (images, strings) are wandb's business.
                    continue
            self._tb_writer.flush()
        super()._wandb_log(payload)

    def _finish_wandb(self):
        if self._tb_writer is not None:
            self._tb_writer.close()
            self._tb_writer = None
        super()._finish_wandb()
