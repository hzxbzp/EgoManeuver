"""Unified lateral event boundaries, label version u1 (from an audit of the source boundaries).

Adds `lat_u1` to every labels.npz (the original `lat` is kept; select with VEGO_LAT_KEY=lat_u1). Rules:
1. Rule-generated lane changes (map datasets: nuScenes, AV2, nuPlan): positive only on the core every source agrees
   on, [t_cross - 1.5 s, t_cross + 1.0 s] (t_cross = lane-crossing time from the lane assignment), intersected with
   the original interval; the rest of the original interval becomes IGNORE.
2. Rule-generated turns and U-turns (map datasets, PAI pose rules, BDD turns): positive only where the heading change
   is between 10 % and 90 % of the event total; the rest of the original interval becomes IGNORE.
3. Every other lateral event (human labels: HDD, ROAD-Waymo, BDD-X text; rule events without a usable anchor): the
   0.5 s next to each boundary becomes IGNORE (annotation jitter), if the event is long enough to keep 0.5 s.
Straight / curve / NO_MANEUVER and the longitudinal axis are unchanged.

Usage: python data/relabel_boundaries.py [--datasets ...] [--dry-run]
"""

import argparse
import collections
import glob
import json
import os

import numpy as np

from egokin.data.index import ROOT
from egokin.taxonomy import IGNORE, Lat

HZ = 10
BAND = int(0.5 * HZ)
LC_CORE = (int(1.5 * HZ), int(1.0 * HZ))  # frames before / after the lane crossing
RULE_MAP = {"nuscenes", "av2", "nuplan"}
RULE_TURN = RULE_MAP | {"pai", "bdd"}
LC = {int(Lat.LANE_CHANGE_LEFT), int(Lat.LANE_CHANGE_RIGHT)}
TURNS = {int(Lat.TURN_LEFT), int(Lat.TURN_RIGHT), int(Lat.U_TURN)}
KEEP = {int(Lat.STRAIGHT), int(Lat.CURVE)}


def segments(lat):
    """(start, end, class) runs of lateral event classes."""
    out, s = [], 0
    for i in range(1, len(lat) + 1):
        if i == len(lat) or lat[i] != lat[s]:
            if int(lat[s]) >= 0 and int(lat[s]) not in KEEP:
                out.append((s, i, int(lat[s])))
            s = i
    return out


def lane_crossing(road, s, e):
    if road is None:
        return None
    off = road[s:e, 0]
    d = np.abs(np.diff(off))
    if not len(d) or np.nanmax(np.nan_to_num(d)) <= 1.5:
        return None
    return s + int(np.nanargmax(np.nan_to_num(d))) + 1


def turn_core(kin, s, e):
    yaw = kin[s:e, 2]
    if np.isnan(yaw).mean() > 0.2:
        return None
    h = np.cumsum(np.nan_to_num(yaw)) / HZ
    if abs(np.degrees(h[-1])) < 30:
        return None
    frac = h / h[-1]
    a, b = np.flatnonzero(frac >= 0.1), np.flatnonzero(frac >= 0.9)
    return (s + int(a[0]), s + int(b[0]) + 1) if len(a) and len(b) else None


def relabel(ds, lat, kin, road):
    new = lat.copy()
    how = collections.Counter()
    for s, e, c in segments(lat):
        core = None
        if ds in RULE_MAP and c in LC:
            tc = lane_crossing(road, s, e)
            if tc is not None:
                core = (max(s, tc - LC_CORE[0]), min(e, tc + LC_CORE[1]))
        elif ds in RULE_TURN and c in TURNS:
            core = turn_core(kin, s, e)
        if core is not None and core[1] - core[0] >= BAND:
            new[s:e] = IGNORE
            new[core[0]:core[1]] = c
            how["core"] += 1
        elif e - s >= 2 * BAND + BAND:
            new[s:s + BAND] = IGNORE
            new[e - BAND:e] = IGNORE
            how["band"] += 1
        else:
            how["short_kept"] += 1
    return new, how


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    dsets = a.datasets.split(",") if a.datasets else sorted(os.listdir(f"{ROOT}/clips"))
    stats = collections.defaultdict(collections.Counter)
    for ds in dsets:
        for f in sorted(glob.glob(f"{ROOT}/clips/{ds}/*/labels.npz")):
            z = np.load(f)
            arrays = {k: z[k] for k in z.files}
            lat = arrays["lat"]
            new, how = relabel(ds, lat, arrays["kin"].astype(np.float64),
                               arrays["road"].astype(np.float64) if "road" in arrays else None)
            stats[ds].update(how)
            for c in np.unique(lat[lat >= 0]):
                stats[ds][f"{Lat(int(c)).name}:before"] += int((lat == c).sum())
                stats[ds][f"{Lat(int(c)).name}:after"] += int((new == c).sum())
            if not a.dry_run:
                arrays["lat_u1"] = new.astype(lat.dtype)
                tmp = f[:-4] + ".tmp.npz"
                np.savez_compressed(tmp, **arrays)
                os.replace(tmp, f)
    report = {ds: dict(c) for ds, c in stats.items()}
    for ds, c in report.items():
        ev = {k: v for k, v in c.items() if ":" not in k}
        cls = sorted({k.split(":")[0] for k in c if ":" in k})
        ch = {k: f"{c[k + ':before'] / 10:.0f}s->{c[k + ':after'] / 10:.0f}s" for k in cls
              if k not in ("STRAIGHT", "CURVE") and c[k + ":before"]}
        print(ds, ev, ch, flush=True)
    if not a.dry_run:
        json.dump(report, open(f"{ROOT}/labels/relabel_u1_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
