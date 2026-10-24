"""Dense perspective road-structure targets ("read the road" supervision, M2-b).

Targets live on the encoder's patch grid: the full frame resized to 224x448 gives 14x28 tokens per tubelet,
so each target cell is the matching 1/14 x 1/28 block of the original image. Stored per clip in
clips/<dataset>/<clip_id>/dense.npz: f (n,) 10 fps frame indices, y (n, C, 14, 28) float16 in [0, 1],
ch (C,) bool = channels annotated in this dataset (others are unknown, not negative).

Region channels hold the covered area fraction; thin structures and small objects hold presence (any pixel
above a small area fraction).
"""

import numpy as np

DENSE = ["ego_lane", "other_lanes", "marking_parallel", "marking_perp", "road_curb", "construction", "blocked",
         "junction_area"]
REGION = {"ego_lane", "other_lanes", "blocked", "junction_area"}
GRID = (14, 28)


def to_grid(masks: dict, shape, present_thr: float = 0.01) -> np.ndarray:
    """masks: {channel: (H, W) bool} -> (C, 14, 28) float16 targets (unlisted channels are 0)."""
    import cv2

    out = np.zeros((len(DENSE),) + GRID, np.float32)
    for c, name in enumerate(DENSE):
        m = masks.get(name)
        if m is None:
            continue
        frac = cv2.resize(m.astype(np.float32), (GRID[1], GRID[0]), interpolation=cv2.INTER_AREA)
        out[c] = frac if name in REGION else (frac > present_thr)
    return out.astype(np.float16)


def save(clip_dir: str, frames, targets, channels):
    import os

    np.savez_compressed(os.path.join(clip_dir, "dense.npz"), f=np.asarray(frames, np.int32),
                        y=np.asarray(targets, np.float16), ch=np.array([c in channels for c in DENSE]))
