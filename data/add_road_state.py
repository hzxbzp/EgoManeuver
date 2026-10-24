"""Add the map-based road-relative state (labels.npz key "road", columns lateral_map.ROAD_FIELDS) to
existing AV2 and nuScenes clips. Videos and other labels are left untouched.

Usage: python data/add_road_state.py [--workers 16]
"""

import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from egokin.data.clipstore import ROOT, clip_dir


def write_road(dataset, clip_id, road):
    path = os.path.join(clip_dir(dataset, clip_id), "labels.npz")
    z = dict(np.load(path))
    n = len(z["long"])
    r = np.full((n, road.shape[1]), np.nan, np.float32)
    r[: min(n, len(road))] = road[:n]
    z["road"] = r
    np.savez_compressed(path, **z)


def av2_job(log_dir):
    from egokin.data.sources.av2 import lead_queue_mask, load_lanegraph, load_track
    from egokin.labels.factory import label_clip

    log_id = log_dir.rstrip("/").split("/")[-1]
    if not os.path.exists(os.path.join(clip_dir("av2", log_id), "labels.npz")):
        return "missing"
    tr, t0 = load_track(log_dir)
    g, xw = load_lanegraph(log_dir)
    write_road("av2", log_id, label_clip(tr, True, g, xw, lead_queue=lead_queue_mask(log_dir, tr, t0)).road)
    return "ok"


def nusc_job(meta_path):
    from egokin.data.sources import nuscenes as ns
    from egokin.labels.factory import label_clip

    m = json.load(open(meta_path))
    sc = ns.nusc().get("scene", m["scene_token"])
    g, xw = ns.lanegraph(m["location"])
    tr, _ = ns.scene_track(ns.scene_frames(sc))
    write_road("nuscenes", m["clip_id"], label_clip(tr, m["rht"], g, xw).road)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    with ProcessPoolExecutor(a.workers) as ex:
        r1 = list(ex.map(av2_job, sorted(glob.glob(f"{ROOT}/raw/av2/*/*/")), chunksize=8))
    print("av2:", {k: r1.count(k) for k in set(r1)}, flush=True)
    with ProcessPoolExecutor(8) as ex:
        r2 = list(ex.map(nusc_job, sorted(glob.glob(f"{ROOT}/clips/nuscenes/*/meta.json")), chunksize=8))
    print("nuscenes:", {k: r2.count(k) for k in set(r2)}, flush=True)


if __name__ == "__main__":
    main()
