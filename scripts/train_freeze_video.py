"""Training entrypoint with the video expert frozen.

This is a thin wrapper around :mod:`scripts.train` that forces
``freeze_video_expert=true``. It exists only so that the historical command
line ``python scripts/train_freeze_video.py task=...`` keeps working.

Usage::

    bash scripts/train_freeze_video.sh 2 task=libero_uncond_2cam224_1e-4_300m \\
        resume=checkpoints/libero_uncond_2cam224_300m.pt

Equivalent, and preferred for new work::

    bash scripts/train_zero1.sh 2 task=... freeze_video_expert=true

Historical note -- why this file no longer monkey-patches the trainer:

    It used to replace ``Wan22Trainer._configure_trainable_parameters`` and
    ``Wan22Trainer._set_dit_only_train_mode``. Both patches were subtly broken.
    ``_configure_trainable_parameters`` began with
    ``if hasattr(model, "configure_trainable_parameters"): return
    model.configure_trainable_parameters(freeze_video_expert=self.freeze_video_expert)``
    -- and ``self.freeze_video_expert`` came from the config, defaulting to
    ``False``. So for any model that defined that method (``DreamFastWAM``), the
    entire patch body was dead code and nothing was frozen at optimizer-build
    time. The *other* patch then set ``requires_grad_(False)`` on the video
    expert only after ``accelerator.prepare``, by which point ZeRO had already
    allocated master weights plus Adam moments for ~5 B parameters that would
    never receive a gradient.

    ``FastWAM`` now implements ``configure_trainable_parameters`` itself (as
    ``DreamFastWAM`` already did), so the single ``freeze_video_expert`` config
    flag is honoured by every model and the patches are unnecessary.
"""

import hydra
from omegaconf import DictConfig, open_dict

from fastwam.runtime import run_training
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    # Force the freeze on, but tell the user if their override said otherwise so
    # a contradictory command line does not silently do the opposite.
    if cfg.get("freeze_video_expert", False) is False:
        with open_dict(cfg):
            cfg.freeze_video_expert = True
        logger.info("train_freeze_video.py: forcing freeze_video_expert=true.")
    run_training(cfg)


if __name__ == "__main__":
    main()
