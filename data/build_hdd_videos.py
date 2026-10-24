"""Transcode all HDD session videos into the unified clip store (video only; labels are written
after the per-session EAF time alignment, see docs/progress.md).

Usage: python data/build_hdd_videos.py [--workers 6]
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

from egokin.data.clipstore import clip_dir
from egokin.data.sources import hdd
from egokin.data.transcode import transcode_video


def work(item):
    sid, sdir = item
    out = os.path.join(clip_dir("hdd", sid), "video.mp4")
    if os.path.exists(out):
        return sid, -1
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ft = hdd.frame_times(sid)
    ft = ft - ft[0]
    n = transcode_video(hdd.video_path(sdir), ft, float(ft[-1]), out + ".tmp.mp4")
    os.replace(out + ".tmp.mp4", out)
    return sid, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    items = list(hdd.sessions().items())
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (sid, n) in enumerate(ex.map(work, items)):
            print(f"{i + 1}/{len(items)} {sid} frames={n}", flush=True)


if __name__ == "__main__":
    main()
