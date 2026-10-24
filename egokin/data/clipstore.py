"""Unified clip store: $VEGO_DATA/clips/<dataset>/<clip_id>/{video.mp4, labels.npz, meta.json}.

video.mp4 frame k <-> labels row k (both at OUT_FPS from the same clip origin).
"""

import json
import os

import av
import numpy as np

from egokin.data.transcode import OUT_FPS
from egokin.labels.factory import ClipLabels
from egokin.labels.kinematics import EgoTrack
from egokin.taxonomy import Lat

ROOT = os.environ.get("VEGO_DATA", "./vego_data")


def clip_dir(dataset: str, clip_id: str) -> str:
    return os.path.join(ROOT, "clips", dataset, clip_id)


def sky_luma(video_path: str, n: int = 5) -> float:
    """Mean luminance of the top fifth of n evenly spaced frames (night if < ~60)."""
    with av.open(video_path) as c:
        frames = [f.to_ndarray(format="gray") for f in c.decode(video=0)]
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    return float(np.mean([frames[i][: frames[i].shape[0] // 5].mean() for i in idx]))


def light_from_luma(luma: float) -> str:
    return "night" if luma < 60 else ("dusk" if luma < 90 else "day")


def save_labels(dataset: str, clip_id: str, tr: EgoTrack, lab: ClipLabels, meta: dict, n_video_frames: int):
    assert abs(tr.hz - OUT_FPS) < 1e-6, "labels must be on the video grid"
    d = clip_dir(dataset, clip_id)
    os.makedirs(d, exist_ok=True)
    n = min(len(tr), n_video_frames)
    ctx = {f"ctx_{k}": np.asarray(v[:n], bool) for k, v in lab.context.items()}
    if getattr(lab, "road", None) is not None:
        ctx["road"] = lab.road[:n].astype(np.float32)
    np.savez_compressed(
        os.path.join(d, "labels.npz"),
        long=lab.long[:n].astype(np.int8), lat=lab.lat[:n].astype(np.int8),
        strong=lab.strong[:n], hard_brake=lab.hard_brake[:n],
        kin=np.stack([tr.v, tr.a, tr.yaw_rate, tr.curvature], 1)[:n].astype(np.float32), **ctx)
    meta = dict(meta, n_frames=n, fps=OUT_FPS, label_source=lab.source, label_version=lab.version,
                events=[dict(kind=Lat(e.kind).name, start=int(e.start), end=int(e.end), disp=round(float(e.disp), 2),
                             conf=float(e.conf), note=e.note) for e in lab.events if e.start < n])
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
