"""Lateral label version u2 = u1 boundaries + label-rule fixes and test-pool corrections from the
verification (226 events reviewed, $VEGO_DATA/verify/review_assistant.csv).

Rule fixes, applied to the original `lat` before the u1 boundary rules (scripts/relabel_boundaries.relabel):
1. ROAD-Waymo: AV-MovLft/MovRht were mapped to lane changes, but most are not (2 of 12 confirmed: also
   borrows around stopped trucks, turns, drifts) -> IGNORE.
2. Map datasets (AV2, nuScenes, nuPlan): rule nudges / borrows (offset excursions) are mostly lane re-matching,
   lane widening or a biased habitual offset, one with the wrong direction (2 of 9 confirmed) -> IGNORE.
3. BDD-X captions (fixed regexes, first match wins): "pulls over to the right lane" is a lane change, not a pull
   over; "turn around a bend" is not a U-turn; "u turn" with a space and "merges into the lane to its left" were
   missed (12 and 168 captions).
4. BDD pose turns / U-turns from GPS heading: spurious at crawling speed (heading spikes) -> IGNORE when the mean
   speed is below 2 m/s or the yaw rate exceeds 1.2 rad/s inside the event, unless a caption says the same.
Then the u1 boundary rules, then the verified test-pool corrections: events judged wrong get the suggested
straight / curve label or IGNORE; events the assistant could not decide become IGNORE until an expert reviews them
(except human HDD events, which are kept).

Writes `lat_u2` into every labels.npz (other keys untouched; atomic replace). Usage:
  python data/relabel_u2.py [--datasets ...] [--dry-run]
"""

import argparse
import collections
import csv
import glob
import importlib.util
from egokin.scripts import script_path  # noqa: E402
import json
import os
import re

import numpy as np

from egokin.data.index import ROOT
from egokin.taxonomy import IGNORE, Lat, Long

_spec = importlib.util.spec_from_file_location("rb", script_path("relabel_boundaries.py"))
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)

HZ = 10
LC = {int(Lat.LANE_CHANGE_LEFT), int(Lat.LANE_CHANGE_RIGHT)}
NUDGE = {int(Lat.NUDGE_LEFT), int(Lat.NUDGE_RIGHT), int(Lat.BORROW_LEFT), int(Lat.BORROW_RIGHT)}
TURNS = {int(Lat.TURN_LEFT), int(Lat.TURN_RIGHT), int(Lat.U_TURN)}
MAP = {"av2", "nuscenes", "nuplan"}
SIDE = r"(left|right)"
# BDD-X action text -> lateral class, first match wins (egokin/data/sources/bdd.BDDX_LAT with the fixes above)
BDDX_LAT_V2 = [
    ("LANE_CHANGE_LEFT", r"pull(s|ed|ing)? over (in)?to the left lane"),
    ("LANE_CHANGE_RIGHT", r"pull(s|ed|ing)? over (in)?to the right lane"),
    ("PULL_OVER", r"pull(s|ed|ing)? (over|to the (side|curb))"),
    ("PULL_OUT", r"pull(s|ed|ing)? (out|away from the curb|into (the )?(traffic|road|lane))"),
    ("U_TURN", r"\bu[- ]?turn|turn(s|ing)? around(?! (a|the) (bend|curve|corner))"),
    ("LANE_CHANGE_LEFT", r"((change|switch|merg)(es|ing|e|ed)? (in)?to the left( lane)?)|((change|switch)(es|ing)? lanes? (to the )?left)"
                         r"|((change|switch|merg|mov)\w* (in)?to the lane (to|on) (its|the) left)"),
    ("LANE_CHANGE_RIGHT", r"((change|switch|merg)(es|ing|e|ed)? (in)?to the right( lane)?)|((change|switch)(es|ing)? lanes? (to the )?right)"
                          r"|((change|switch|merg|mov)\w* (in)?to the lane (to|on) (its|the) right)"),
    ("TURN_LEFT", r"(turn(s|ing|ed)? (to the )?left)|(left turn)"),
    ("TURN_RIGHT", r"(turn(s|ing|ed)? (to the )?right)|(right turn)"),
]


def bddx_class(text):
    t = text.lower()
    for name, pat in BDDX_LAT_V2:
        if re.search(pat, t):
            return name
    return None


def fix_bdd(lat, kin, long, meta, how):
    n = len(lat)
    span = lambda a: slice(max(0, int(a["start"] * HZ)), min(n, int(np.ceil(a["end"] * HZ)) + 1))
    captions = np.full(n, -1, np.int64)  # caption class per frame (after the fix), for rule 4
    for a in meta.get("bddx_actions", []) or []:
        if a.get("reverse"):
            continue
        old, new = a.get("lat"), bddx_class(a["action"])
        s = span(a)
        if new is not None:
            captions[s] = int(Lat[new])
        if old == new:
            continue
        if new is None:  # e.g. "turn around a bend": drop the wrong caption class
            seg = lat[s]
            seg[seg == int(Lat[old])] = IGNORE
            lat[s] = seg
        else:
            lat[s] = int(Lat[new])
        how[f"bddx {old}->{new}"] += 1
    for s, e, c in rb.segments(lat):
        if c not in TURNS or (captions[s:e] == c).mean() > 0.5:
            continue
        v, yaw = kin[s:e, 0], kin[s:e, 2]
        if np.nanmean(np.abs(v)) < 2.0 or np.nanmax(np.abs(yaw)) > 1.2:
            lat[s:e] = IGNORE
            how["bdd pose turn at crawl / spike -> IGNORE"] += 1
    lat[long == int(Long.REVERSE)] = IGNORE  # as in the original builder
    return lat


def fix_rules(ds, lat, kin, long, meta, how):
    lat = lat.astype(np.int64).copy()
    if ds == "road_waymo":
        m = np.isin(lat, list(LC))
        how["road_waymo MovLft/Rht frames -> IGNORE"] += int(m.sum())
        lat[m] = IGNORE
    if ds in MAP:
        m = np.isin(lat, list(NUDGE))
        how["map nudge/borrow frames -> IGNORE"] += int(m.sum())
        lat[m] = IGNORE
    if ds == "bdd":
        lat = fix_bdd(lat, kin, long, meta, how)
    return lat


def corrections():
    """(dataset, clip_id) -> [(s, e, new label)] from the verification (u1 core spans, 10 Hz frames)."""
    vdir = os.path.join(ROOT, "verify")
    ev = {r["eid"]: r for r in csv.DictReader(open(os.path.join(vdir, "events.csv")))}
    out = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(vdir, "review_assistant.csv"))):
        e = ev[r["eid"]]
        if r["verdict"] == "ok" or (r["verdict"] == "unsure" and e["dataset"] == "hdd" and not r["suggest"]):
            continue  # confirmed, or a human HDD event the assistant could not refute
        sug = r["suggest"].split()[0] if r["suggest"] else ""
        new = int(Lat[sug]) if r["verdict"] == "wrong" and sug in ("STRAIGHT", "CURVE") else IGNORE
        out[(e["dataset"], e["clip_id"])].append((int(e["s"]), int(e["e"]), new, r["eid"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    dsets = a.datasets.split(",") if a.datasets else sorted(os.listdir(f"{ROOT}/clips"))
    corr = corrections()
    stats = collections.defaultdict(collections.Counter)
    applied = set()
    for ds in dsets:
        for f in sorted(glob.glob(f"{ROOT}/clips/{ds}/*/labels.npz")):
            cid = os.path.basename(os.path.dirname(f))
            z = np.load(f)
            arrays = {k: z[k] for k in z.files}
            meta = json.load(open(os.path.join(os.path.dirname(f), "meta.json")))
            kin = arrays["kin"].astype(np.float64)
            how = stats[ds]
            lat_v2 = fix_rules(ds, arrays["lat"], kin, arrays["long"], meta, how)
            new, h = rb.relabel(ds, lat_v2, kin, arrays["road"].astype(np.float64) if "road" in arrays else None)
            for s, e, lab, eid in corr.get((ds, cid), []):
                new[s:e] = lab
                how["verified correction"] += 1
                applied.add(eid)
            old = arrays["lat_u1"].astype(np.int64)
            for c in np.unique(np.r_[old[old >= 0], new[new >= 0]]):
                how[f"{Lat(int(c)).name}:u1"] += int((old == c).sum())
                how[f"{Lat(int(c)).name}:u2"] += int((new == c).sum())
            if not a.dry_run:
                arrays["lat_u2"] = new.astype(arrays["lat"].dtype)
                tmp = f[:-4] + ".tmp.npz"
                np.savez_compressed(tmp, **arrays)
                os.replace(tmp, f)
    missing = {eid for v in corr.values() for *_, eid in v} - applied
    if missing and not a.datasets:
        print("corrections not applied (clip not found):", sorted(missing))
    report = {ds: dict(c) for ds, c in stats.items()}
    for ds, c in report.items():
        fixes = {k: v for k, v in c.items() if ":" not in k}
        cls = sorted({k.split(":")[0] for k in c if ":" in k} - {"STRAIGHT", "CURVE"})
        ch = {k: f"{c.get(k + ':u1', 0) / 10:.0f}s->{c.get(k + ':u2', 0) / 10:.0f}s" for k in cls
              if c.get(k + ":u1", 0) != c.get(k + ":u2", 0)}
        print(ds, fixes, ch, flush=True)
    if not a.dry_run:
        os.makedirs(f"{ROOT}/labels", exist_ok=True)
        json.dump(report, open(f"{ROOT}/labels/relabel_u2_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
