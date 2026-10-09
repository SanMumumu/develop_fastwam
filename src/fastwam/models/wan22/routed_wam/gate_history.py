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
        self.gate_kind = None
        self.replan_gates_per_layer = []
        self.gates_per_layer = []
        self._current_per_layer = None

    def begin_decision(self, prediction, *, batch_index=0):
        """Call once per infer_action result; supports selecting one batch member."""
        if "group_gates" not in prediction:
            self._current = None
            self._current_per_layer = None
            return
        gates = prediction["group_gates"]
        if hasattr(gates, "detach"):
            gates = gates.detach().float().cpu().numpy()
        gates = np.asarray(gates, dtype=np.float32)
        mapping = prediction["group_mapping"]
        if gates.ndim != 2 or gates.shape[1] != len(mapping) or not len(mapping):
            raise ValueError("Prediction group_gates must be [B, num_groups].")
        kind = prediction.get("gate_kind", "qk_scale")
        lower = 0 if kind == "attention_prior" else -1
        if not np.isfinite(gates).all() or (gates < lower).any() or (gates > 1).any():
            raise ValueError(f"Gate history requires finite gates in [{lower},1].")
        if [g["group_id"] for g in mapping] != list(range(len(mapping))):
            raise ValueError("Expected an ordered mapping of all groups.")
        if self.group_mapping is not None and self.group_mapping != mapping:
            raise ValueError("Group mapping changed within an episode.")
        if self.gate_kind is not None and kind != self.gate_kind:
            raise ValueError("Gate semantics changed within an episode.")
        per_layer = prediction.get("group_gates_per_layer")
        if per_layer is not None:
            if hasattr(per_layer, "detach"):
                per_layer = per_layer.detach().float().cpu().numpy()
            per_layer = np.asarray(per_layer, dtype=np.float32)
            if (per_layer.ndim != 3 or per_layer.shape[0] != gates.shape[0]
                    or per_layer.shape[1] == 0 or per_layer.shape[2] != gates.shape[1]
                    or not np.isfinite(per_layer).all()
                    or (per_layer < lower).any() or (per_layer > 1).any()):
                raise ValueError("group_gates_per_layer must contain valid [B,L,Ng] gates.")
            if not np.allclose(per_layer.mean(axis=1), gates, rtol=1e-5, atol=1e-6):
                raise ValueError("group_gates must be the mean over recorded layers.")
            if self.replan_gates_per_layer and per_layer.shape[1:] != self.replan_gates_per_layer[0].shape:
                raise ValueError("Gate layer count changed within an episode.")
        if self.replan_gates and bool(self.replan_gates_per_layer) != (per_layer is not None):
            raise ValueError("Per-layer gate recording changed within an episode.")
        self.gate_kind = kind
        self.group_mapping = mapping
        self._current = gates[batch_index].copy()
        self.replan_gates.append(self._current.copy())
        self._current_per_layer = None if per_layer is None else per_layer[batch_index].copy()
        if self._current_per_layer is not None:
            self.replan_gates_per_layer.append(self._current_per_layer.copy())

    def append_step(self, env_step: int):
        """Call for every executed action, including the terminal step, not settling steps."""
        if self._current is not None:
            self.gates.append(self._current.copy())
            self.env_steps.append(int(env_step))
            self.replan_ids.append(len(self.replan_gates) - 1)
            if self._current_per_layer is not None:
                self.gates_per_layer.append(self._current_per_layer.copy())

    def _modalities(self, gates):
        if self.group_mapping is None:
            return np.empty((0, 4), dtype=np.float32)
        values = []
        for name in self.modality_order:
            indices = [g["group_id"] for g in self.group_mapping
                       if g["modality"] in ("all", "dyn" if name == "tracker" else name)]
            # An absent modality has no exposed Dream knowledge.
            values.append(gates[:, indices].mean(axis=1) if indices else np.zeros(len(gates)))
        return np.stack(values, axis=1)

    def arrays(self):
        num_groups = len(self.group_mapping) if self.group_mapping is not None else 16
        gates = np.asarray(self.gates, dtype=np.float32).reshape(-1, num_groups)
        replans = np.asarray(self.replan_gates, dtype=np.float32).reshape(-1, num_groups)
        result = dict(gate_history=gates, modality_history=self._modalities(gates),
                    replan_gate_history=replans, replan_modality_history=self._modalities(replans),
                    env_steps=np.asarray(self.env_steps, dtype=np.int64),
                    replan_ids=np.asarray(self.replan_ids, dtype=np.int64))
        if self.replan_gates_per_layer:
            layers = self.replan_gates_per_layer[0].shape[0]
            result["gate_history_per_layer"] = np.asarray(
                self.gates_per_layer, dtype=np.float32).reshape(-1, layers, num_groups)
            result["replan_gate_history_per_layer"] = np.asarray(self.replan_gates_per_layer)
        return result

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
        if self.gate_kind == "attention_prior":
            payload.update(gate_kind=self.gate_kind, gate_denoising_step="last", gate_layer_reduction="mean")
        (directory / "gate_history.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def plot(self, path, *, layer=None):
        """Learned group priors: Ng x T, [0,1]; legacy QK scales: 4 x T, [-1,1]."""
        import matplotlib.pyplot as plt

        prior = self.gate_kind == "attention_prior"
        arrays = self.arrays()
        values = arrays["gate_history"] if prior else arrays["modality_history"]
        if layer is not None:
            values = arrays["gate_history_per_layer"][:, layer]
        labels = ([g.get("name", str(g["group_id"])) for g in self.group_mapping]
                  if prior or layer is not None else self.modality_order)
        if not len(values):
            raise ValueError("No executed gate history to plot.")
        fig, ax = plt.subplots(figsize=(12, max(3, len(labels) * 0.3)))
        im = ax.imshow(values.T, aspect="auto", vmin=0 if prior else -1, vmax=1,
                       cmap="viridis" if prior else "coolwarm")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Executed control step (excluding settling)")
        fig.colorbar(im, ax=ax, label="Dream group attention prior" if prior else "Mean Dream QK scale")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
