"""Reproducibility utilities for the small LIBERO routing experiment.

No model, simulator, or dataset imports: manifests and summaries run on CPU.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def make_split(dataset_root, seed=42, validation_fraction=0.1):
    root = Path(dataset_root)
    records = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines()]
    groups = {}
    for record in records:
        if len(record["tasks"]) != 1:
            raise ValueError("Routing split requires one task per episode.")
        groups.setdefault(record["tasks"][0], []).append(int(record["episode_index"]))
    rng = np.random.default_rng(seed)
    train, val, by_task = [], [], {}
    for task, ids in sorted(groups.items()):
        ids = np.array(sorted(ids))
        rng.shuffle(ids)
        count = max(1, int(round(len(ids) * validation_fraction)))
        if len(ids) <= count:
            raise ValueError("Every task needs at least one training and validation episode.")
        held, fitted = sorted(ids[:count].tolist()), sorted(ids[count:].tolist())
        train.extend(fitted)
        val.extend(held)
        by_task[task] = {"train": fitted, "val": held}
    return dict(version=1, seed=seed, dataset_root=str(root.resolve()),
                episodes_sha256=file_sha256(root / "meta/episodes.jsonl"),
                train=sorted(train), val=sorted(val), by_task=by_task,
                total_episodes=len(records), total_frames=sum(r["length"] for r in records))


def load_split(path, dataset_root, training):
    payload = json.loads(Path(path).read_text())
    if payload["episodes_sha256"] != file_sha256(Path(dataset_root) / "meta/episodes.jsonl"):
        raise ValueError("Dataset metadata differs from the locked episode manifest.")
    train, val = payload["train"], payload["val"]
    if set(train) & set(val) or len(set(train + val)) != payload["total_episodes"]:
        raise ValueError("Episode split contains overlap or missing episodes.")
    return train if training else val


def paired_action_noise(action, sample_ids, *, scheduler, seed=42):
    """Per-sample RNG independent of model construction, batches, ranks and dropout."""
    noises, times = [], []
    for sample_id in torch.as_tensor(sample_ids).reshape(-1).cpu().tolist():
        generator = torch.Generator().manual_seed((int(seed) * 1_000_003 + int(sample_id)) % (2**63 - 1))
        noises.append(torch.randn(action.shape[1:], generator=generator, dtype=torch.float32))
        u = torch.rand((), generator=generator)
        times.append(scheduler._phi(u, scheduler.shift) * scheduler.num_train_timesteps)
    if len(noises) != len(action):
        raise ValueError("One stable sample ID is required per action sample.")
    return torch.stack(noises).to(action), torch.stack(times).to(action)


def frozen_checksum(model):
    digest = hashlib.sha256()
    for name, param in model.named_parameters():
        if param.requires_grad:
            continue
        digest.update(name.encode())
        # Byte views support bfloat16; never convert to float (would change bits).
        data = param.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(data))
    return digest.hexdigest()


def paired_bootstrap(records, left, right, *, seed=42, repeats=2000):
    """Task-stratified paired bootstrap; observations are episodes, not frames."""
    groups = {}
    for row in records:
        groups.setdefault(row["task"], []).append(float(row[left]) - float(row[right]))
    if not groups:
        raise ValueError("No paired episodes to summarize.")
    arrays = [np.asarray(group) for _, group in sorted(groups.items())]
    rng = np.random.default_rng(seed)
    means = np.stack([rng.choice(a, size=(repeats, len(a)), replace=True).mean(axis=1) for a in arrays]).mean(axis=0)
    return {"difference": float(np.mean([a.mean() for a in arrays])),
            "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
            "episodes": sum(map(len, arrays)), "tasks": len(arrays)}
