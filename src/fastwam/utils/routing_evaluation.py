"""Opt-in evaluation protocol. Legacy callers without routing_eval are unchanged."""
from pathlib import Path
import json

from omegaconf import OmegaConf

from .routing_experiment import file_sha256, json_hash


def rollout_seed(seed, task, initial_state, replan=0):
    return (int(seed) + int(task) * 1_000_000 + int(initial_state) * 10_000 + int(replan)) % (2**63 - 1)


def initial_state_ids(cfg, count):
    routing = cfg.get("routing_eval")
    if routing is None:
        return list(range(int(cfg.EVALUATION.num_trials)))
    ids = [int(x) for x in routing.initial_state_ids]
    if not ids or len(ids) != len(set(ids)) or min(ids) < 0 or max(ids) >= count:
        raise ValueError("routing_eval.initial_state_ids must be unique existing initial states.")
    return ids


def apply_routing_policy(model, cfg):
    routing = cfg.get("routing_eval")
    if routing is None:
        return None
    policy = str(routing.get("policy", "native"))
    gates = None
    if policy == "calibrated_mean":
        payload = json.loads(Path(routing.calibration_path).read_text())
        if payload["checkpoint_sha256"] != file_sha256(cfg.ckpt):
            raise ValueError("Gate calibration belongs to another checkpoint.")
        gates = payload["group_gates"]
    model.set_routing_evaluation(policy, gates)
    return policy


def evaluation_identity(cfg, dataset_stats_path):
    """Refuse resume after checkpoint/protocol changes; GPU/output location is irrelevant."""
    evaluation = OmegaConf.to_container(cfg.EVALUATION, resolve=True)
    for name in ("output_dir", "device", "num_trials", "task_id", "initial_state_index"):
        evaluation.pop(name, None)
    routing = OmegaConf.to_container(cfg.routing_eval, resolve=True)
    routing.pop("resume", None)
    if routing.get("calibration_path"):
        routing["calibration_sha256"] = file_sha256(routing["calibration_path"])
    import importlib.metadata
    versions = {}
    for name in ("mujoco", "robosuite", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    payload = dict(checkpoint_sha256=file_sha256(cfg.ckpt),
                   dataset_stats_sha256=file_sha256(dataset_stats_path),
                   model=OmegaConf.to_container(cfg.model, resolve=True),
                   evaluation=evaluation, routing=routing, seed=cfg.get("seed"), versions=versions,
                   mixed_precision=cfg.get("mixed_precision"))
    comparable = dict(evaluation=evaluation, seed=cfg.get("seed"), versions=versions,
                      max_steps=routing.get("max_steps",700), mixed_precision=cfg.get("mixed_precision"),
                      initial_state_ids=routing.get("initial_state_ids"),
                      stats=json_hash(json.loads(Path(dataset_stats_path).read_text())))
    return dict(protocol_sha256=json_hash(payload), pairing_sha256=json_hash(comparable), **payload)


def resume_episode(path, identity, task, initial_state, *, resume=True):
    path = Path(path)
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if (record["protocol_sha256"] != identity["protocol_sha256"]
            or record["task_id"] != task or record["initial_state_id"] != initial_state):
        raise ValueError(f"Existing episode has different provenance: {path}")
    if not resume:
        raise FileExistsError(f"Episode already exists: {path}")
    return record
