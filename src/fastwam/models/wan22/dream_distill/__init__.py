"""Dream Imagination Distillation (DID).

Motivation
----------
FastWAM's joint attention mask (see
``fastwam/model.py::_build_mot_attention_mask`` and
``wan_video_dit.py::build_video_to_video_mask``) has a consequence that is easy
to miss: the video expert's **future-frame tokens have no downstream reader**.

* Video-frame-0 tokens attend only to themselves.
* Video frames 1..N attend to everything -- but nothing attends to *them*.
* Action attends to frame 0, the Dream tokens, and itself.
* Dream attends to frame 0 and its own modality block.

So the video branch computes an imagined future at every layer, and neither the
Dream expert nor the Action expert can read it. The video loss shapes the shared
weights; it never reaches the activations the policy consumes. That is the
structural reason a world-action model of this shape does not need test-time
future imagination -- it was never able to use it.

DID adds the missing channel **at training time only**: the Dream tokens are
supervised to predict the video expert's future-frame representation. At
inference nothing changes -- Dream still reads only the current frame, the
Video+Dream prefix is still prefillable, and the test-time cost is identical.

Alignment
---------
The correspondence is exact rather than heuristic. The VAE has a temporal
downsample factor of 4, and ``dream_target.future_offsets`` are expressed in raw
frames, so **latent frame ``f`` is exactly future offset ``4f``**. Offsets that
are not a multiple of the temporal factor are rejected rather than rounded.

Ablation ladder (all configs live in ``configs/task/dream_distill_*``):

======  ====================================================================
R0      stock ``DreamFastWAM`` -- structurally has no distillation head at
        all, rather than the same model with ``lambda_did=0``
R1/R2   DID at one / three layers
R3      teacher is the **current** frame instead of a future frame. The
        control that separates "future information helps" from "an extra
        supervision signal helps"
R4      Dream attends to the future video tokens directly, at train *and*
        test time. Breaks the prefill and costs a full video forward at test
        time, but measures the ceiling
======  ====================================================================

R3 is not a formality. ``ckpt_eval_report.md`` already shows the
``future_offsets=[0]`` runs (Dream supervised on the *current* frame) scoring
97.0 / 96.2 / 94.8 / 94.2 -- the best of the Dream group -- while the default
``[4]`` runs sit around 92-95. The existing evidence therefore leans towards
"Dream helps via multi-modal representation supervision, not via the future".
DID should only be considered to work if R1 beats R3.

Rollback: delete this package plus the ``dream_distill`` configs, scripts and
tests. Nothing outside it imports it -- the model configs reach it through
``_target_``, so ``fastwam/runtime.py`` is untouched.
"""

from .mot import DistillMoT
from .model import DreamDistillFastWAM

__all__ = ["DistillMoT", "DreamDistillFastWAM"]
