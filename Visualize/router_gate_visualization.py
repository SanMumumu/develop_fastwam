"""Two-row replan storyboard: observation above, pooled semantic gates below.

Pure offline renderer: no environment, model weights, or LIBERO dependency.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch


MODALITIES = ("dino", "dyn", "sam", "depth")
LABELS = ("DINO", "CoTracker", "SAM", "Depth")
COLORS = ((45, 220, 255), (255, 166, 55), (199, 115, 255), (70, 245, 153))  # RGB
BACKGROUND = (9, 13, 23)


def pool_group_gates(routing: dict) -> np.ndarray:
    """Average the four camera/horizon groups of each modality, without normalization."""
    gates = routing["group_gates"]
    if torch.is_tensor(gates):
        gates = gates.detach().float().cpu().numpy()
    gates = np.asarray(gates, dtype=np.float32)
    if gates.shape == (1, 16):
        gates = gates[0]
    if gates.shape != (16,) or not np.isfinite(gates).all():
        raise ValueError("A replan record must contain 16 finite gates for one episode.")
    if (gates < -1).any() or (gates > 1).any():
        raise ValueError("Gate values must be in [-1,1].")
    mapping = routing["group_mapping"]
    if len(mapping) != 16 or sorted(g["group_id"] for g in mapping) != list(range(16)):
        raise ValueError("Expected a one-to-one mapping for all 16 groups.")
    result = []
    for modality in MODALITIES:
        groups = [g for g in mapping if g["modality"] == modality]
        if len(groups) != 4 or len({(g['view'], g['future_offset']) for g in groups}) != 4:
            raise ValueError(f"Expected four distinct camera/horizon groups for {modality}.")
        result.append(gates[[g["group_id"] for g in groups]].mean())
    return np.asarray(result, dtype=np.float32)


def load_replan_summaries(raw_dir: str | Path) -> list[dict]:
    """Read one large Dream record at a time; retain only RGB and four gate values."""
    paths = sorted(Path(raw_dir).glob("replan_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No replan records in {raw_dir}.")
    summaries = []
    last_step = -1
    expected_mapping = None
    for path in paths:
        record = torch.load(path, map_location="cpu", weights_only=False)
        if "routing" not in record:
            raise ValueError(f"{path.name} has no recorded semantic gates. Do not substitute fabricated values.")
        routing = record["routing"]
        activations = pool_group_gates(routing)
        mapping = sorted(routing['group_mapping'], key=lambda group: group['group_id'])
        if expected_mapping is not None and mapping != expected_mapping:
            raise ValueError("Group mapping changed within this episode.")
        expected_mapping = mapping
        metadata = record['metadata']
        step = int(metadata['env_step'])
        index = int(metadata['replan_index'])
        if step <= last_step or index != len(summaries):
            raise ValueError("Replan records must be contiguous and in increasing environment-step order.")
        last_step = step
        summaries.append(dict(rgb=record['rgb'], activations=activations,
                              env_step=step, replan_index=index))
        del record  # dense Dream targets are deliberately not retained
    return summaries


def _text(canvas, value, xy, scale=0.45, color=(206, 214, 230)):
    cv2.putText(canvas, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _fit_rgb(rgb, width, height):
    if torch.is_tensor(rgb):
        rgb = rgb.detach().cpu().numpy()
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Recorded rollout RGB must be uint8 [H,W,3].")
    h, w = rgb.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))))
    result = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
    dh, dw = resized.shape[:2]
    result[(height-dh)//2:(height-dh)//2+dh, (width-dw)//2:(width-dw)//2+dw] = resized
    return result


def render_column(summary, *, camera="image", column_width=320, image_height=224):
    """Signed QK scales: zero-centered bars, negative left and positive right."""
    if camera not in ("image", "wrist_image", "both"):
        raise ValueError("camera must be image, wrist_image or both.")
    if column_width < 240 or image_height < 96:
        raise ValueError("column_width must be >=240 and image_height >=96.")
    header, separator, row_height, footer = 34, 32, 46, 24
    height = header + image_height + separator + 4 * row_height + footer
    canvas = np.full((height, column_width, 3), BACKGROUND, dtype=np.uint8)
    _text(canvas, f"REPLAN {summary['replan_index']:03d}   STEP {summary['env_step']}", (12, 23))
    rgb = summary['rgb']
    if camera == 'both':
        left = _fit_rgb(rgb['image'], column_width // 2, image_height)
        right = _fit_rgb(rgb['wrist_image'], column_width - column_width // 2, image_height)
        observation = np.concatenate([left, right], axis=1)
    else:
        observation = _fit_rgb(rgb[camera], column_width, image_height)
    canvas[header:header+image_height] = observation
    y0 = header + image_height
    _text(canvas, "DREAM QK SCALE   pooled views / horizons", (12, y0 + 22), scale=0.36)
    for i, (label, color, value) in enumerate(zip(LABELS, COLORS, summary['activations'])):
        y = y0 + separator + i * row_height
        _text(canvas, label, (14, y + 15), color=color)
        _text(canvas, f"{value:+.3f}", (column_width - 70, y + 15), color=(238, 243, 250))
        x1, x2 = 14, column_width - 14
        cv2.rectangle(canvas, (x1, y + 24), (x2, y + 32), (25, 33, 48), -1)
        center = (x1 + x2) // 2
        length = int(round(abs(float(value)) * (x2 - x1) / 2))
        if length:
            glow = np.zeros_like(canvas)
            intensity = tuple(int(c * abs(float(value))) for c in color)
            left, right = (center - length, center) if value < 0 else (center, center + length)
            cv2.rectangle(glow, (left, y + 24), (right, y + 32), intensity, -1)
            # Fixed kernel/strength across the episode; no framewise rescaling.
            halo = cv2.GaussianBlur(glow, (0, 0), 4)
            canvas = np.clip(canvas.astype(np.float32) + halo.astype(np.float32) * 0.65,
                             0, 255).astype(np.uint8)
            cv2.rectangle(canvas, (left, y + 24), (right, y + 32), intensity, -1)
        cv2.line(canvas, (center, y + 22), (center, y + 34), (130, 144, 165), 1)
    _text(canvas, "-1                             0                             +1",
          (14, height - 8), scale=0.30, color=(130, 144, 165))
    return canvas


def _save_rgb(path, image):
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed to save {path}")


def render_gate_episode(raw_dir, output_dir, *, camera="image", column_width=320,
                        image_height=224, columns_per_page=8, fps=5, save_video=True):
    if columns_per_page < 1 or fps <= 0:
        raise ValueError("columns_per_page and fps must be positive.")
    summaries = load_replan_summaries(raw_dir)
    output_dir = Path(output_dir)
    frames_dir, pages_dir = output_dir / 'frames', output_dir / 'pages'
    frames_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)
    columns = [render_column(s, camera=camera, column_width=column_width,
                             image_height=image_height) for s in summaries]
    def join(frames):
        gap = np.full((frames[0].shape[0], 8, 3), (30, 39, 55), dtype=np.uint8)
        parts = []
        for frame in frames:
            if parts:
                parts.append(gap)
            parts.append(frame)
        return np.concatenate(parts, axis=1)
    timeline = output_dir / 'timeline.png'
    _save_rgb(timeline, join(columns))
    for start in range(0, len(columns), columns_per_page):
        _save_rgb(pages_dir / f'page_{start // columns_per_page:03d}.png',
                  join(columns[start:start + columns_per_page]))
    video_path = output_dir / 'gate_rollout.mp4'
    writer = None
    try:
        if save_video:
            h, w = columns[0].shape[:2]
            # Video codecs require even dimensions; pad, never crop the labels.
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'),
                                     float(fps), (w + w % 2, h + h % 2))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {video_path}")
        for summary, column in zip(summaries, columns):
            _save_rgb(frames_dir / f"replan_{summary['replan_index']:04d}.png", column)
            if writer is not None:
                h, w = column.shape[:2]
                frame = cv2.copyMakeBorder(column, 0, h % 2, 0, w % 2, cv2.BORDER_CONSTANT)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        if writer is not None:
            writer.release()
    activations = np.stack([s['activations'] for s in summaries])
    np.save(output_dir / 'replan_modality_history.npy', activations)
    result = dict(timeline=str(timeline), pages=str(pages_dir), frames=str(frames_dir),
                  video=str(video_path) if save_video else None, num_replans=len(summaries),
                  camera=camera, modality_order=list(LABELS), scale=[-1, 1], fps=float(fps),
                  playback='one frame per replan; not simulator realtime',
                  replans=[dict(replan_index=s['replan_index'], env_step=s['env_step'],
                                activations=s['activations'].tolist()) for s in summaries])
    (output_dir / 'render_manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--camera', choices=['image', 'wrist_image', 'both'], default='image')
    parser.add_argument('--columns-per-page', type=int, default=8)
    parser.add_argument('--column-width', type=int, default=320)
    parser.add_argument('--image-height', type=int, default=224)
    parser.add_argument('--fps', type=float, default=5)
    parser.add_argument('--no-video', action='store_true')
    args = parser.parse_args()
    print(json.dumps(render_gate_episode(args.raw_dir, args.output_dir, camera=args.camera,
        columns_per_page=args.columns_per_page, column_width=args.column_width,
        image_height=args.image_height, fps=args.fps, save_video=not args.no_video), indent=2))


if __name__ == '__main__':
    main()
