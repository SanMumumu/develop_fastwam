"""Export unnormalized semantic gates at replan and executed-control-step rates."""

import json
from pathlib import Path

import numpy as np


class GateHistory:
    modality_order = ("dino", "tracker", "sam", "depth")

    def __init__(self):
        self.group_mapping = None
        self.replan_gates = []
        self.gates = []
        self.env_steps = []
        self.replan_ids = []
        self._current = None

    def begin_decision(self, prediction, *, batch_index=0):
        """Call once per infer_action result; supports selecting one batch member."""
        if "group_gates" not in prediction:
            self._current = None
            return
        gates = prediction["group_gates"]
        if hasattr(gates, "detach"):
            gates = gates.detach().float().cpu().numpy()
        gates = np.asarray(gates, dtype=np.float32)
        if gates.ndim != 2 or gates.shape[1] != 16:
            raise ValueError("Prediction group_gates must be [B,16].")
        if not np.isfinite(gates).all() or (gates < -1).any() or (gates > 1).any():
            raise ValueError("Gate history requires finite gates in [-1,1].")
        mapping = prediction["group_mapping"]
        if len(mapping) != 16 or [g["group_id"] for g in mapping] != list(range(16)):
            raise ValueError("Expected an ordered mapping of 16 groups.")
        if self.group_mapping is not None and self.group_mapping != mapping:
            raise ValueError("Group mapping changed within an episode.")
        self.group_mapping = mapping
        self._current = gates[batch_index].copy()
        self.replan_gates.append(self._current.copy())

    def append_step(self, env_step: int):
        """Call for every executed action, including the terminal step, not settling steps."""
        if self._current is not None:
            self.gates.append(self._current.copy())
            self.env_steps.append(int(env_step))
            self.replan_ids.append(len(self.replan_gates) - 1)

    def _modalities(self, gates):
        if self.group_mapping is None:
            return np.empty((0, 4), dtype=np.float32)
        return np.stack([
            gates[:, [g["group_id"] for g in self.group_mapping
                      if g["modality"] == ("dyn" if name == "tracker" else name)]].mean(axis=1)
            for name in self.modality_order
        ], axis=1)

    def arrays(self):
        gates = np.asarray(self.gates, dtype=np.float32).reshape(-1, 16)
        replans = np.asarray(self.replan_gates, dtype=np.float32).reshape(-1, 16)
        return dict(gate_history=gates, modality_history=self._modalities(gates),
                    replan_gate_history=replans, replan_modality_history=self._modalities(replans),
                    env_steps=np.asarray(self.env_steps, dtype=np.int64),
                    replan_ids=np.asarray(self.replan_ids, dtype=np.int64))

    def save(self, directory):
        """Write .npy arrays plus JSON containing the same values and semantic mapping."""
        if self.group_mapping is None:
            return  # Models without semantic routing do not create empty logs.
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = self.arrays()
        for name, values in arrays.items():
            np.save(directory / f"{name}.npy", values)
        payload = {name: values.tolist() for name, values in arrays.items()}
        payload.update(group_mapping=self.group_mapping, modality_order=self.modality_order)
        (directory / "gate_history.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def plot(self, path):
        """Optional standalone 4 x T heatmap with a fixed [-1,1] color scale."""
        import matplotlib.pyplot as plt

        values = self.arrays()["modality_history"]
        if not len(values):
            raise ValueError("No executed gate history to plot.")
        fig, ax = plt.subplots(figsize=(12, 3))
        im = ax.imshow(values.T, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_yticks(range(4), self.modality_order)
        ax.set_xlabel("Executed control step (excluding settling)")
        fig.colorbar(im, ax=ax, label="Mean Dream QK scale")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
