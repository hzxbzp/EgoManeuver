"""Fetch BDD100K videos (BDD-X subset first) from the mirror, transcode, and write unified clips.

Usage: python data/build_bdd.py --subset bddx [--workers 6]
"""

import argparse
import json
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from egokin.data.clipstore import clip_dir
from egokin.data.sources import bdd
from egokin.data.transcode import transcode_video

ROOT = os.environ.get("VEGO_DATA", "./vego_data")


def work(args):
    name, split, attrs, acts = args
    d = clip_dir("bdd", name)
    if os.path.exists(os.path.join(d, "meta.json")):
        return name, "skip"
    tmp = tempfile.mkdtemp(dir=f"{ROOT}/cache")
    try:
        info = bdd.read_info(bdd.info_zip(), split, name)
        tr, diag = bdd.track_from_info(info)
        mov = bdd.fetch_video(split, name, tmp)
        import av
        with av.open(mov) as c:
            s = c.streams.video[0]
            nfr = s.frames or int(float(c.duration / 1e6) * float(s.average_rate))
            fps = float(s.average_rate)
        ft = np.arange(nfr) / fps
        os.makedirs(d, exist_ok=True)
        n = transcode_video(mov, ft, min(float(tr.t[-1]), float(ft[-1])), os.path.join(d, "video.mp4"))
        long_lab, lat, strong, hard, ctx = bdd.label_video(tr, pd.DataFrame(acts), attrs)
        n = min(n, len(tr))
        np.savez_compressed(os.path.join(d, "labels.npz"), long=long_lab[:n].astype(np.int8), lat=lat[:n].astype(np.int8),
                            strong=strong[:n], hard_brake=hard[:n],
                            kin=np.stack([tr.v, tr.a, tr.yaw_rate, tr.curvature], 1)[:n].astype(np.float32),
                            **{f"ctx_{k}": v[:n] for k, v in ctx.items()})
        meta = dict(dataset="bdd", clip_id=name, split=split, n_frames=int(n), fps=10, attrs=attrs, gyro=diag,
                    label_source="gps+gyro" + ("+bddx" if len(acts) else ""), label_version="v0",
                    bddx_actions=[{k: a[k] for k in ("start", "end", "action", "lat", "reverse")} for a in acts])
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1, default=str)
        return name, "ok"
    except Exception as e:
        return name, f"error {type(e).__name__}: {e}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="bddx")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    attrs = pd.read_csv(f"{ROOT}/raw/bdd/bdd_attributes.csv").set_index("video")
    acts = bdd.bddx_actions()
    names = [v for v in acts.video.unique() if v in attrs.index]
    # rare-action videos first
    rare = set(acts[acts.lat.isin(["PULL_OVER", "PULL_OUT", "U_TURN"]) | acts.reverse].video)
    names = sorted(names, key=lambda v: (v not in rare, v))[: a.limit or None]
    by = {v: g[["start", "end", "action", "lat", "reverse"]].to_dict("records") for v, g in acts.groupby("video")}
    jobs = [(v, attrs.at[v, "split"], {k: attrs.at[v, k] for k in ("timeofday", "weather", "scene")}, by.get(v, [])) for v in names]
    print(f"{len(jobs)} videos ({len(rare & set(names))} with rare actions first)", flush=True)
    ok = err = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (name, status) in enumerate(ex.map(work, jobs)):
            ok += status in ("ok", "skip"); err += status.startswith("error")
            if status.startswith("error") and err <= 20:
                print(name, status, flush=True)
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(jobs)} ok={ok} err={err}", flush=True)
    print(f"done ok={ok} err={err}", flush=True)


if __name__ == "__main__":
    main()
