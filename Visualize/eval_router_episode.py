#!/usr/bin/env python
"""Run one LIBERO episode and export an observation/gate storyboard plus separate Dream images."""
from pathlib import Path
import json
import sys

import hydra
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Visualize.infer_dream_episode import run_one_episode, _jsonable
from Visualize.dream_prediction_visualization import render_saved_episode
from Visualize.router_gate_visualization import render_gate_episode


def evaluate(cfg: DictConfig):
    output_dir = Path(cfg.EVALUATION.output_dir)
    if bool(cfg.GATE_VISUALIZATION.get("render_only", False)):
        manifest = json.loads((output_dir / 'episode_manifest.json').read_text())
    else:
        if any((output_dir / 'raw_predictions').glob('replan_*.pt')):
            raise FileExistsError(f"{output_dir} already contains an episode; choose a new output_dir.")
        manifest = run_one_episode(cfg)
    if not manifest['num_replans']:
        manifest['gate_visualization'] = dict(num_replans=0, reason='No model control decisions executed.')
    else:
        settings = cfg.GATE_VISUALIZATION
        manifest['gate_visualization'] = render_gate_episode(
            output_dir / 'raw_predictions', output_dir / 'gate_visualization',
            camera=str(settings.camera), column_width=int(settings.column_width),
            image_height=int(settings.image_height), columns_per_page=int(settings.columns_per_page),
            fps=float(settings.fps), save_video=bool(settings.save_video),
        )
        if bool(settings.render_dream_images):
            vis = cfg.VISUALIZATION
            manifest['dream_visualization'] = render_saved_episode(
                output_dir / 'raw_predictions', output_dir / 'dream_visualization',
                fps=int(vis.fps), panel_size=int(vis.panel_size), alpha=float(vis.overlay_alpha),
                sam_clusters=int(vis.sam_clusters), draw_contours=bool(vis.draw_contours),
                max_projection_samples=int(vis.max_projection_samples),
                sam_render_mode=str(vis.get('sam_render_mode', 'regions')),
            )
    (output_dir / 'episode_manifest.json').write_text(
        json.dumps(manifest, indent=2, default=_jsonable), encoding='utf-8')
    return manifest


@hydra.main(version_base='1.3', config_path='../configs', config_name='visualize_router_episode')
def main(cfg: DictConfig):
    print(json.dumps(evaluate(cfg), indent=2, default=_jsonable))


if __name__ == '__main__':
    main()
