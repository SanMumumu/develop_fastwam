from collections import defaultdict
from typing import Optional, Sequence

import numpy as np


class ActionEnsembler:
    """Temporal ensembling of overlapping action chunks.

    With ``action_horizon=32`` but ``replan_steps=5``, a given environment
    timestep is covered by several predicted chunks. Averaging those predictions
    smooths the executed trajectory.

    Two behaviours worth knowing:

    * ``exclude_dims`` skips averaging for dimensions where the mean is not
      meaningful. The gripper command is binary (open/close); averaging it across
      chunks that disagree yields a fractional value that is neither, so by
      default the newest prediction wins for that dimension.
    * Entries older than the current timestep are dropped on every query, so the
      cache stays bounded over an episode instead of growing to its full length.
    """

    def __init__(self, exclude_dims: Optional[Sequence[int]] = (-1,)):
        self.action_cache = defaultdict(list)
        self.exclude_dims = tuple(exclude_dims) if exclude_dims else ()

    def reset(self):
        self.action_cache.clear()

    def add_actions(self, action_chunk: np.ndarray, start_timestamp: int):
        if action_chunk.ndim == 3:
            action_chunk = action_chunk.squeeze(0)
        horizon, action_dim = action_chunk.shape

        for i in range(horizon):
            target_ts = start_timestamp + i
            self.action_cache[target_ts].append(action_chunk[i, :])

    def get_action(self, timestamp: int) -> np.ndarray:
        if timestamp not in self.action_cache:
            raise ValueError(f"No actions cached for timestamp {timestamp}")
        preds = self.action_cache[timestamp]
        stacked_preds = np.stack(preds, axis=0)
        averaged_action = np.mean(stacked_preds, axis=0)
        # Keep non-averageable dimensions (gripper) from the most recent chunk.
        for dim in self.exclude_dims:
            averaged_action[dim] = stacked_preds[-1, dim]
        return averaged_action

    def cleanup(self, current_timestamp: int):
        """Drop cached predictions for timesteps already executed."""
        stale = [ts for ts in self.action_cache if ts < current_timestamp]
        for ts in stale:
            del self.action_cache[ts]

    # Backwards-compatible alias for the previously private name.
    _cleanup = cleanup
