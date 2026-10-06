"""CPU-only audit of Dream flow math, schedules and checkpoint conditioning weights.

This does not claim rollout quality from decoder skip weights alone. It neither
loads the Video expert nor modifies a checkpoint or training configuration.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf

from fastwam.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from fastwam.models.wan22.wan_video_dit import sinusoidal_embedding_1d


def oracle_check(scheduler, steps, dtype):
    generator = torch.Generator().manual_seed(13)
    clean = torch.randn(2, 4, 8, generator=generator).to(dtype)
    noise = torch.randn(2, 4, 8, generator=generator).to(dtype)
    times, deltas = scheduler.build_inference_schedule(steps, torch.device('cpu'), dtype)
    current = noise.clone()
    for time, delta in zip(times, deltas):
        velocity = scheduler.training_target(clean, noise, time)
        current = scheduler.step(velocity, delta, current)
    return float((current.float() - clean.float()).abs().max())


def audit_checkpoint(path):
    path = Path(path)
    config_path = next(parent / 'config.yaml' for parent in path.parents if (parent / 'config.yaml').exists())
    cfg = OmegaConf.load(config_path)
    sc = cfg.model.dream_scheduler
    scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=int(sc.num_train_timesteps), shift=float(sc.infer_shift))
    steps = int(sc.inference_steps)
    times, deltas = scheduler.build_inference_schedule(steps, torch.device('cpu'), torch.float32)
    payload = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    state = payload['mot']
    prefix = 'mixtures.dream.'
    time_state = {k.removeprefix(prefix+'dream_time_embedding.'): v.float()
                  for k, v in state.items() if k.startswith(prefix+'dream_time_embedding.')}
    if not time_state:
        raise ValueError(f'{path} has no generative Dream time embedding.')
    hidden, freq = time_state['0.weight'].shape
    time_net = nn.Sequential(nn.Linear(freq, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
    time_net.load_state_dict(time_state)
    # Match timestep rounding in the shipped bf16 runtime, evaluate small MLP
    # in float32 for a cheap CPU diagnostic. This is not exact bf16 execution.
    embeddings = time_net(sinusoidal_embedding_1d(freq, times.to(torch.bfloat16)).float())
    result = dict(checkpoint=str(path), step=payload.get('step'), training_config=str(config_path),
                  times=times.tolist(), deltas=deltas.tolist(),
                  last_sigma=float(times[-1] / sc.num_train_timesteps),
                  schedule_weight=scheduler.training_weight(times).tolist(),
                  oracle_max_error_fp32=oracle_check(scheduler, steps, torch.float32),
                  oracle_max_error_bf16=oracle_check(scheduler, steps, torch.bfloat16),
                  conditioner_key_count=sum('decoder_conditioners.' in k for k in state), modalities={})
    for name in ('dyn', 'depth', 'dino', 'sam'):
        root = prefix + f'decoder_conditioners.{name}.skip_gain.'
        if root+'weight' not in state:
            result['modalities'][name] = {'missing_local_noise_path': True}
            continue
        gain = torch.nn.functional.linear(embeddings, state[root+'weight'].float(), state[root+'bias'].float())
        output_weight = state[prefix+f'decoders.{name}.output_proj.weight']
        # Linear skip ONLY: excludes learned nonlinear decoder and its coupling
        # across integration steps. Never interpret as measured full-model noise.
        skip_only_retention = (1 + deltas[:, None] * gain).prod(dim=0)
        result['modalities'][name] = dict(
            target_encoder_out_scale=float(state[prefix+f'target_encoders.{name}.out_scale'].float()),
            skip_gain_means=gain.mean(-1).tolist(),
            skip_only_noise_retention_rms=float(skip_only_retention.square().mean().sqrt()),
            output_projection_shape=list(output_weight.shape),
        )
    # Audit the actual dataset/decoder indexing convention without changing it.
    tokens = int(cfg.model.dream_query_config.dream_decoder.dyn.target_shape[0])
    n = tokens // 2
    side = int(n**0.5)
    indices = torch.arange(tokens).reshape(side, 2*side)[:, :side].reshape(-1)
    result['dyn_primary_indices_from_wrist_targets'] = int((indices >= n).sum())
    result['dyn_positions_per_camera'] = n
    return result


def target_stats(root):
    result = {}
    for name, subdir, key in [('dino', 'dinov2', 'features'), ('sam', 'sam', 'features'),
                              ('depth', 'depth_anything_v3_metric', 'depth')]:
        path = Path(root) / subdir / 'image/episode_000000.npz'
        with np.load(path) as archive:
            data = archive[key]
        # An explicitly limited sample, not a dataset-wide estimate.
        sampled = np.asarray(data[np.linspace(0, len(data)-1, min(8, len(data)), dtype=int)], dtype=np.float32)
        result[name] = dict(path=str(path), sampled_frames=len(sampled), shape=list(sampled.shape),
                            mean=float(sampled.mean()), std=float(sampled.std()),
                            rms=float(np.sqrt(np.mean(sampled**2))),
                            percentiles=np.percentile(sampled, [1, 50, 99]).tolist())
        del data
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, action='append', required=True)
    parser.add_argument('--extras-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with torch.no_grad():
        report = dict(checkpoints=[audit_checkpoint(p) for p in args.checkpoint])
    if args.extras_root:
        report['sampled_target_stats'] = target_stats(args.extras_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
