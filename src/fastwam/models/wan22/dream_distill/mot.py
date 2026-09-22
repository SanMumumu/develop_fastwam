"""A MoT that can snapshot per-layer expert token states.

The distillation objective needs the hidden state of the video and dream experts
at intermediate layers, but ``MoT.forward`` only returns the final one.

Rather than duplicate ``MoT.forward`` (which would silently drift from the
original), this subclass overrides the single small helper that the forward pass
calls exactly once per (expert, layer): ``_build_expert_attention_io``. That
helper receives the layer's *input* tokens, so the snapshot for layer ``L`` is
"the state entering layer L", i.e. the output of layer ``L-1``.

Two properties make this the right hook:

* it is called from the plain forward path, **not** from inside
  ``torch.utils.checkpoint``. Hooking the post-block instead would fire a second
  time during gradient-checkpoint recomputation and capture tensors belonging to
  the recomputation graph;
* it receives the ``block`` object, so the (expert, layer) identity is recovered
  by object identity without threading an index through the call chain.

Capture is off unless explicitly enabled, and when off this class is
behaviourally identical to ``MoT``.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch

from ..mot import MoT


class DistillMoT(MoT):
    """``MoT`` with opt-in per-layer activation capture."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._capture_layers: set[int] = set()
        self._capture_experts: set[str] = set()
        self._capture_enabled: bool = False
        self._snapshots: dict[tuple[str, int], torch.Tensor] = {}
        self._block_index: Optional[dict[int, tuple[str, int]]] = None

    # -- capture control ----------------------------------------------------
    def _build_block_index(self) -> dict[int, tuple[str, int]]:
        index: dict[int, tuple[str, int]] = {}
        for name in self.expert_order:
            for layer_idx, block in enumerate(self.mixtures[name].blocks):
                index[id(block)] = (name, layer_idx)
        return index

    def configure_capture(self, layers: Iterable[int], experts: Iterable[str]) -> None:
        """Declare which (expert, layer) states should be captured."""
        layers = {int(x) for x in layers}
        experts = {str(x) for x in experts}
        for layer in layers:
            if not 0 <= layer < self.num_layers:
                raise ValueError(
                    f"capture layer {layer} out of range for a {self.num_layers}-layer MoT"
                )
        missing = experts - set(self.expert_order)
        if missing:
            raise ValueError(f"capture requested for unknown expert(s): {sorted(missing)}")
        self._capture_layers = layers
        self._capture_experts = experts
        self._block_index = None  # rebuilt lazily

    def enable_capture(self, enabled: bool = True) -> None:
        self._capture_enabled = bool(enabled)
        if not enabled:
            self._snapshots = {}

    def pop_snapshots(self) -> dict[tuple[str, int], torch.Tensor]:
        """Return the captured states and clear the buffer.

        Always drain this after a forward pass: holding the dict keeps the
        activations (and their autograd graph) alive.
        """
        snapshots, self._snapshots = self._snapshots, {}
        return snapshots

    # -- hook ---------------------------------------------------------------
    def _build_expert_attention_io(self, expert, block, x, freqs, t_mod):
        if self._capture_enabled and self._capture_layers:
            if self._block_index is None:
                self._block_index = self._build_block_index()
            key = self._block_index.get(id(block))
            if (
                key is not None
                and key[1] in self._capture_layers
                and key[0] in self._capture_experts
            ):
                # Keep the graph: the dream-side snapshot is the student and must
                # be differentiable. The teacher is detached later, at the point
                # of use, so that this hook stays modality-agnostic.
                self._snapshots[key] = x
        return super()._build_expert_attention_io(expert, block, x, freqs, t_mod)
