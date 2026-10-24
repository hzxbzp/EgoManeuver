"""Write our labels for every HDD session into the clip store (videos come from build_hdd_videos.py).

labels.npz: long, lat (10 Hz, real time), strong, hard_brake, kin, stop_cause, ctx_*,
            official_goal (3 fps, official TRN targets) + official_t (real time of each official frame).
Usage: python data/build_hdd_labels.py [--workers 16]
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import av
import numpy as np

from egokin.data.clipstore import clip_dir
from egokin.data.sources import hdd
from egokin.taxonomy import Lat

ROOT = os.environ.get("VEGO_DATA", "./vego_data")
SPLITS = json.load(open(f"{ROOT}/raw/hdd/splits/trn_data_info.json"))["HDD"]


def work(item):
    sid, sdir = item
    d = clip_dir("hdd", sid)
    with av.open(os.path.join(d, "video.mp4")) as c:
        n_video = c.streams.video[0].frames
    tr, t0, long_lab, lat, strong, hard, ctx, stop_cause, ann = hdd.label_session(sid, sdir)
    n = min(len(tr), n_video)
    official = np.load(f"{ROOT}/raw/hdd/data/target/{sid}.npy").astype(np.int8)
    np.savez_compressed(
        os.path.join(d, "labels.npz"),
        long=long_lab[:n].astype(np.int8), lat=lat[:n].astype(np.int8), strong=strong[:n], hard_brake=hard[:n],
        kin=np.stack([tr.v, tr.a, tr.yaw_rate, tr.curvature], 1)[:n].astype(np.float32),
        stop_cause=stop_cause[:n], official_goal=official,
        official_t=hdd.trn_frame_real_times(sid, len(official)).astype(np.float32),
        **{f"ctx_{k}": v[:n] for k, v in ctx.items()})
    split = "train" if sid in SPLITS["train_session_set"] else "test" if sid in SPLITS["test_session_set"] else "unused"
    counts = {Lat(k).name: int(v) for k, v in zip(*np.unique(lat[:n][lat[:n] >= 0], return_counts=True))}
    meta = dict(dataset="hdd", clip_id=sid, split=split, n_frames=int(n), fps=10, label_source="hdd_eaf+can",
                label_version="v0", time_base="real (EAF converted from frame-index time)",
                lat_frame_counts=counts, has_work_zone=bool(ctx["work_zone"].any()),
                reverse_frames=int((long_lab[:n] == 6).sum()))
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return sid, split, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    total = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for sid, split, counts in ex.map(work, list(hdd.sessions().items())):
            for k, v in counts.items():
                total[k] = total.get(k, 0) + v
    print("lateral frame counts over all sessions:", dict(sorted(total.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
