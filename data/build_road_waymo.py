"""Build unified clips from ROAD-Waymo (frontal Waymo videos + human AV-action labels).

Labels: AV-Stop -> STATIONARY, AV-Rev -> REVERSE, AV-Brake -> DECELERATE (DECEL_TO_STOP right before a
stop), first 2 s of motion after a stop -> START; AV-TurLft/Rht -> TURN_*, AV-MovLft/Rht -> LANE_CHANGE_*,
AV-Ovtak -> IGNORE (composite). Frames without a lateral maneuver stay IGNORE on axis B (straight vs curve
is unknown without poses). Kinematics are unknown (NaN). Night from sky luminance.

Usage: python data/build_road_waymo.py [--workers 8]
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from egokin.data.clipstore import ROOT, clip_dir, light_from_luma, sky_luma
from egokin.data.transcode import transcode_video
from egokin.labels.segments import runs
from egokin.taxonomy import IGNORE, Lat, Long

RAW = f"{ROOT}/raw/waymo/road_waymo"
RELABEL = os.environ.get("RELABEL") == "1"  # recompute labels.npz of existing clips only
CROP = (0, 100, 1920, 1180)  # 3:2 -> 16:9 band around the horizon


def labels_from_actions(acts: list[set], names: list[str]):
    n = len(acts)
    has = lambda k: np.array([names.index(k) in a for a in acts]) if k in names else np.zeros(n, bool)
    long_lab = np.full(n, IGNORE, np.int64)
    lat = np.full(n, IGNORE, np.int64)
    stop, rev, brake = has("AV-Stop"), has("AV-Rev"), has("AV-Brake")
    long_lab[brake] = Long.DECELERATE
    for s, e in runs(stop):
        long_lab[s:e] = Long.STATIONARY
        j = s - 1
        while j >= 0 and brake[j]:
            long_lab[j] = Long.DECEL_TO_STOP
            j -= 1
        moving = [k for k in range(e, min(n, e + 20)) if not stop[k] and not rev[k]]
        long_lab[moving] = Long.START
    long_lab[rev] = Long.REVERSE
    for key, cls in [("AV-TurLft", Lat.TURN_LEFT), ("AV-TurRht", Lat.TURN_RIGHT),
                     ("AV-MovLft", Lat.LANE_CHANGE_LEFT), ("AV-MovRht", Lat.LANE_CHANGE_RIGHT)]:
        lat[has(key)] = cls
    lat[has("AV-Ovtak") | stop | rev] = IGNORE
    # moving frames without any annotated maneuver: known to be lane keeping (straight or curve)
    lat_neg = (lat == IGNORE) & ~(has("AV-Ovtak") | stop | rev)
    return long_lab, lat, lat_neg


def work(args):
    name, split, acts, names = args
    d = clip_dir("road_waymo", name)
    if os.path.exists(os.path.join(d, "meta.json")):
        if RELABEL:
            n = len(acts)
            long_lab, lat, lat_neg = labels_from_actions(acts, names)
            z = dict(np.load(os.path.join(d, "labels.npz")))
            k = len(z["long"])
            z.update(long=long_lab[:k].astype(np.int8), lat=lat[:k].astype(np.int8), lat_neg=lat_neg[:k])
            np.savez_compressed(os.path.join(d, "labels.npz"), **z)
            return "relabeled"
        return "skip"
    os.makedirs(d, exist_ok=True)
    n = transcode_video(f"{RAW}/videos/{name}.mp4", np.arange(len(acts)) / 10.0, (len(acts) - 1) / 10.0,
                        os.path.join(d, "video.mp4"), crop_box=CROP)
    long_lab, lat, lat_neg = labels_from_actions(acts, names)
    k = min(n, len(acts))
    np.savez_compressed(os.path.join(d, "labels.npz"), long=long_lab[:k].astype(np.int8), lat=lat[:k].astype(np.int8),
                        lat_neg=lat_neg[:k], strong=np.zeros(k, bool), hard_brake=np.zeros(k, bool),
                        kin=np.full((k, 4), np.nan, np.float32))
    luma = sky_luma(os.path.join(d, "video.mp4"))
    meta = dict(dataset="road_waymo", clip_id=name, split=split, n_frames=int(k), fps=10, rht=True,
                label_source="road_waymo_human", label_version="v0", sky_luma=round(luma, 1), light=light_from_luma(luma))
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    d = json.load(open(f"{RAW}/road_waymo_trainval_v1.0.json"))
    names = d["all_av_action_labels"]
    jobs = []
    for v, e in d["db"].items():
        fr = e["frames"]
        acts = [set(fr[k].get("av_action_ids", [])) if k in fr else set() for k in sorted(fr, key=int)]
        split = "val" if "val" in e["split_ids"] else "train"
        jobs.append((v, split, acts, names))
    del d
    stats = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for s in ex.map(work, jobs):
            stats[s] = stats.get(s, 0) + 1
    print(stats)


if __name__ == "__main__":
    main()
