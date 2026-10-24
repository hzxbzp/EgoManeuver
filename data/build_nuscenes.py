"""Build unified nuScenes clips (CAM_FRONT sweeps -> 10 fps video, map labels, scene context).
Idempotent: scenes whose images are not all on disk yet are skipped and picked up by a later run.

Usage: python data/build_nuscenes.py [--workers 12]
"""

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from egokin.data.clipstore import clip_dir
from egokin.data.sources import nuscenes as ns
from egokin.data.transcode import transcode_images
from egokin.labels.factory import label_clip
from egokin.taxonomy import Lat

IDX = None
SPLIT = None


def scene_split(name):
    global SPLIT
    if SPLIT is None:
        from nuscenes.utils.splits import create_splits_scenes
        sp = create_splits_scenes()
        SPLIT = {**{n: "train" for n in sp["train"]}, **{n: "val" for n in sp["val"]}}
    return SPLIT.get(name, "train")


def work(scene_token):
    global IDX
    n = ns.nusc()
    sc = n.get("scene", scene_token)
    d = clip_dir("nuscenes", sc["name"])
    if os.path.exists(os.path.join(d, "meta.json")):
        return sc["name"], "skip"
    frames = ns.scene_frames(sc)
    names = [f["filename"].split("/")[-1] for f in frames]
    if not all(x in IDX for x in names):
        return sc["name"], "missing"
    loc = ns.scene_location(sc)
    g, xw = ns.lanegraph(loc)
    tr, ft = ns.scene_track(frames)
    lab = label_clip(tr, loc not in ns.LEFT_HAND, g, xw)
    os.makedirs(d, exist_ok=True)
    nv = transcode_images([IDX[x] for x in names], ft, float(min(ft[-1], tr.t[-1])), os.path.join(d, "video.mp4"))
    k = min(nv, len(tr))
    desc = sc["description"].lower()
    flags = {"night": "night" in desc, "rain": "rain" in desc, "work_zone": bool(re.search(r"construct|road ?work|cone", desc))}
    ctx = {f"ctx_{c}": v[:k] for c, v in lab.context.items()}
    ctx.update({f"ctx_{c}": np.full(k, v) for c, v in flags.items()})
    ctx["ctx_left_hand_traffic"] = np.full(k, loc in ns.LEFT_HAND)
    np.savez_compressed(os.path.join(d, "labels.npz"), long=lab.long[:k].astype(np.int8), lat=lab.lat[:k].astype(np.int8),
                        strong=lab.strong[:k], hard_brake=lab.hard_brake[:k],
                        kin=np.stack([tr.v, tr.a, tr.yaw_rate, tr.curvature], 1)[:k].astype(np.float32), **ctx)
    meta = dict(dataset="nuscenes", clip_id=sc["name"], scene_token=scene_token, location=loc, split=scene_split(sc["name"]),
                description=sc["description"], n_frames=int(k), fps=10, label_source="map", label_version=lab.version,
                rht=loc not in ns.LEFT_HAND, **flags,
                events=[dict(kind=Lat(e.kind).name, start=int(e.start), end=int(e.end), conf=float(e.conf), note=e.note)
                        for e in lab.events if e.start < k])
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
    return sc["name"], "ok"


def init():
    global IDX
    IDX = ns.image_index()
    ns.nusc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    tokens = [s["token"] for s in ns.nusc().scene]
    stats = {}
    with ProcessPoolExecutor(a.workers, initializer=init) as ex:
        for name, status in ex.map(work, tokens, chunksize=4):
            stats[status] = stats.get(status, 0) + 1
    print(stats)


if __name__ == "__main__":
    main()
