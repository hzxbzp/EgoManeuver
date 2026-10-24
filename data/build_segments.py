"""Benchmark segments v0: cut every clip into event segments with ONE clean lateral and ONE clean
longitudinal label each, only where the behaviour is visually evident.

Longitudinal (less sensitive than the training labels, egokin/labels/longitudinal.py): accelerate / decelerate only
when the smoothed acceleration is >= A_DYN m/s^2 for >= DYN_S seconds AND the speed changes by >= DV m/s over the
run; otherwise cruise. Stationary, start, decelerate-to-stop and reverse as in the training labels.
Lateral: every maneuver run of lat_u2 (ignore bands inside one maneuver bridged) is one segment with up to PAD
seconds of context, trimmed so that no neighbouring maneuver enters it (consecutive maneuvers: each kept as one
complete behaviour, the neighbour excluded). Lane-keeping stretches (straight / curve) between maneuvers are cut
by longitudinal state into STRETCH_MIN-STRETCH_MAX second segments.
A segment is kept only if one longitudinal state covers >= PURITY of it, one lateral class covers >= PURITY of the
maneuver (or of the stretch), and at most IGN_MAX of its frames are ignore.

v2 (default; --v1 restores v1): lane-keeping stretches keep only clean lane keeping: ignore frames at their ends (the
boundary bands of the neighbouring maneuvers) are trimmed off, and unresolved lateral motion inside a stretch (ignore runs,
e.g. pose-only candidates or maneuvers set to ignore by u2) is cut out with PAD seconds on each side; --pai-events writes
Slow-down-then-speed-up: a lane-keeping stretch is split at its speed minimum into a decelerate and an accelerate
segment (a piece shorter than STRETCH_MIN or without a visually evident change is dropped); inside a maneuver (mostly turns)
it is not split and the longitudinal label becomes the partial label MOVING (lateral only). Validation split: a
fixed 10% of each dataset's training scenes / sessions / logs (lowest md5 of the group), never PAI; column `part`.
label overrides (--overrides, comma-separated CSVs with dataset, clip_id, start, end in s and lat = a class name or IGNORE:
e.g. verified test-label corrections) are written over the
lateral labels first. Segments whose heading
change contradicts their lateral label are dropped (kinematic_conflict; the test-pool audit found turns in
parking lots and driveways labelled straight, and turns without any heading change).

Usage: python data/build_segments.py [--splits test,val] [--overrides $VEGO_DATA/benchmark/pai_events.csv,...]
       [--out $VEGO_DATA/benchmark/segments_v2.csv]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from egokin.data.index import ROOT, build_index
from egokin.data.scope import filter_index, frame_mask
from egokin.taxonomy import IGNORE, Lat, Long

HZ = 10
A_DYN, DYN_S, DV = 1.0, 1.5, 3.0
PAD, STRETCH_MIN, STRETCH_MAX = 1.0, 3.0, 10.0
PURITY, IGN_MAX = 0.8, 0.2
KEEP = {int(Lat.STRAIGHT), int(Lat.CURVE)}
SMOOTH = {int(Long.ACCELERATE), int(Long.DECELERATE), int(Long.CRUISE)}


def runs(mask):
    m = np.r_[False, mask, False]
    d = np.flatnonzero(np.diff(m.astype(int)))
    return list(zip(d[::2], d[1::2]))


def clean_longitudinal(long, kin):
    """Training longitudinal labels with accelerate / decelerate kept only where visually evident."""
    out = long.astype(int).copy()
    v, a = kin[:, 0].astype(float), kin[:, 1].astype(float)
    if np.isnan(a).all():  # no kinematics (ROAD-Waymo): only the human stop / move information is usable
        out[np.isin(out, list(SMOOTH - {int(Long.CRUISE)}))] = IGNORE
        return out
    k = int(HZ)  # 1 s moving average
    a_s = np.convolve(np.nan_to_num(a), np.ones(k) / k, mode="same")
    smooth = np.isin(out, list(SMOOTH))
    out[smooth] = int(Long.CRUISE)
    for sign, cls in ((1, Long.ACCELERATE), (-1, Long.DECELERATE)):
        for s, e in runs((sign * a_s >= A_DYN) & smooth):
            if (e - s) >= DYN_S * HZ and abs(np.nanmax(v[s:e]) - np.nanmin(v[s:e])) >= DV:
                out[s:e] = int(cls)
    return out


DV_VIS = 10 / 3.6  # m/s: a speed change counts only when visually evident
V_STOP = 0.3
REMOVED = {int(Lat.ROUNDABOUT), int(Lat.BORROW_RIGHT)}  # too few examples


def segment_longitudinal(lon, kin, lo, hi):
    """One longitudinal label for a whole segment from its speed profile (visually evident changes only);
    without kinematics (ROAD-Waymo) only stationary / start / stop / moving from the human stop labels."""
    v = kin[lo:hi, 0].astype(float)
    if np.isnan(v).all():
        st = lon[lo:hi] == int(Long.STATIONARY)
        if st.all():
            return "STATIONARY"
        if (lon[lo:hi] == int(Long.REVERSE)).any():
            return "REVERSE"
        k = max(1, (hi - lo) // 5)
        if st[:k].all() and not st[-k:].any():
            return "START"
        if not st[:k].any() and st[-k:].all():
            return "STOP"
        return "MOVING" if not st.any() else ""
    v = np.nan_to_num(v)
    if (v < -0.2).sum() >= 5:
        return "REVERSE"
    k = max(1, min(10, (hi - lo) // 5))  # first / last second
    v0, v1, vmin = v[:k].mean(), v[-k:].mean(), v.min()
    if v.max() < V_STOP:
        return "STATIONARY"
    if v0 < V_STOP and v1 >= DV_VIS / 2:
        return "START"
    if v0 >= DV_VIS / 2 and v1 < V_STOP:
        return "STOP"
    if (v < V_STOP).sum() >= 10:  # stops in the middle: not one clean behaviour
        return ""
    if v1 - v0 >= DV_VIS:
        return "ACCELERATE"
    if v0 - v1 >= DV_VIS:
        return "DECELERATE"
    if min(v0, v1) - vmin >= DV_VIS:
        return "DIP"  # slows down then speeds up again (may be merged into CRUISE)
    return "CRUISE"


def maneuvers(lat):
    ev, i, n = [], 0, len(lat)
    while i < n:
        c = lat[i]
        if c >= 0 and c not in KEEP:
            j = i
            while j < n and (lat[j] == c or (lat[j] == IGNORE and c in lat[j + 1:j + 8])):
                j += 1
            ev.append((i, j, int(c)))
            i = j
        else:
            i += 1
    return ev


# v2 kinematic consistency: heading change (degrees) over the segment, yaw-rate bias removed (HDD's CAN yaw rate reads
# about +0.2 deg/s while standing)
DPSI_STRAIGHT, DPSI_TURN, DPSI_UTURN, DPSI_STILL, DPSI_MOVE = 30.0, 30.0, 120.0, 15.0, 45.0
LATERAL_MOVES = {"LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT", "MERGE_LEFT", "MERGE_RIGHT", "BRANCH_LEFT", "BRANCH_RIGHT",
                 "NUDGE_LEFT", "NUDGE_RIGHT", "BORROW_LEFT"}


def yaw_bias(kin, long):
    """Median yaw rate while standing (rad/s), 0 without enough stationary frames."""
    y = kin[:, 2].astype(float)
    st = (long[:len(y)] == int(Long.STATIONARY)) & np.isfinite(y) & (y != 0)
    return float(np.median(y[st])) if st.sum() >= 50 else 0.0


def heading_change(kin, lo, hi, bias=0.0):
    """Heading change in degrees over frames [lo, hi) (left > 0); None without yaw data (ROAD-Waymo, BDD gaps)."""
    y = kin[lo:hi, 2].astype(float)
    if not len(y) or np.all(np.nan_to_num(y) == 0):
        return None
    return float(np.degrees(np.nansum(y - bias) / HZ))


def kinematic_conflict(lat_name, dpsi):
    """Drop reason when the heading change contradicts the lateral label, else ''."""
    if dpsi is None:
        return ""
    a = abs(dpsi)
    if (lat_name == "STRAIGHT" and a >= DPSI_STRAIGHT) or (lat_name == "NONE" and a >= DPSI_STILL):
        return "kinematics: turning"
    if (lat_name == "TURN_LEFT" and dpsi < DPSI_TURN) or (lat_name == "TURN_RIGHT" and dpsi > -DPSI_TURN) or \
            (lat_name == "U_TURN" and a < DPSI_UTURN):
        return "kinematics: no such turn"
    if lat_name in LATERAL_MOVES and a >= DPSI_MOVE:
        return "kinematics: turning"
    return ""


VAL_FRAC = 0.10


def val_group(dataset, clip_id):
    """Scene / session / log a clip belongs to (clips of one group never straddle train and val)."""
    return clip_id.rsplit("_", 2)[0] if dataset == "nuplan" else clip_id  # nuPlan clips: <log>_<start>_<end>


def val_groups(idx):
    """{(dataset, group)} of the validation split: per dataset the VAL_FRAC of training groups with the lowest md5."""
    import hashlib
    out = set()
    tr = idx[(idx.split == "train") & (idx.dataset != "pai")]
    for ds, g in tr.groupby("dataset"):
        groups = sorted({val_group(ds, c) for c in g.clip_id}, key=lambda x: hashlib.md5(f"{ds}/{x}".encode()).hexdigest())
        out |= {(ds, x) for x in groups[:int(round(VAL_FRAC * len(groups)))]}
    return out


def split_dip(lon_raw, kin, lo, hi):
    """Lane-keeping slow-down-then-speed-up -> [(lo, m, label), (m, hi, label)] split at the speed minimum, each piece kept
    only if it is >= STRETCH_MIN long and is a clean decelerate / accelerate (else label '')."""
    v = np.nan_to_num(kin[lo:hi, 0].astype(float))
    m = lo + int(np.argmin(v))
    out = []
    for a_, b_, want in ((lo, m, ("DECELERATE", "STOP")), (m, hi, ("ACCELERATE", "START"))):
        ok = b_ - a_ >= STRETCH_MIN * HZ
        lab = segment_longitudinal(lon_raw, kin, a_, b_) if ok else ""
        out.append((a_, b_, lab if lab in want else ""))
    return out


NONE = 101  # stationary segments have no lateral behaviour
LANE_KEEP = 100  # partial label "no lateral maneuver, straight or curve unknown" (ROAD-Waymo), named LANE_KEEP


def purity(x):
    x = np.where(x == -2, LANE_KEEP, x)
    x = x[x >= 0]
    if not len(x):
        return None, 0.0
    vals, cnt = np.unique(x, return_counts=True)
    return int(vals[cnt.argmax()]), cnt.max() / len(x)


def clean_pieces(lat, s, e):
    """Parts of the moving stretch [s, e) without unresolved lateral frames: ignore runs touching the ends are
    trimmed, inner ignore runs are cut out with PAD seconds on each side."""
    ign = lat[s:e] == IGNORE
    keep = ~ign
    for a, b in runs(ign):
        if a > 0 and b < e - s:
            keep[max(0, a - int(PAD * HZ)):b + int(PAD * HZ)] = False
    return [(s + a, s + b) for a, b in runs(keep)]


def segments(lat, lon, v1=False):
    n, out = len(lat), []
    ev = maneuvers(lat)
    for k, (s, e, c) in enumerate(ev):
        lo = max(0, s - int(PAD * HZ), ev[k - 1][1] if k else 0)
        hi = min(n, e + int(PAD * HZ), ev[k + 1][0] if k + 1 < len(ev) else n)
        out.append(("maneuver", lo, hi, s, e))
    # lane-keeping stretches between maneuvers: stops are their own segments, moving parts are cut into
    # STRETCH_MIN-STRETCH_MAX second pieces
    bounds = [0] + [x for s, e, _ in ev for x in (s, e)] + [n]
    for a, b in zip(bounds[::2], bounds[1::2]):
        if b - a < STRETCH_MIN * HZ:
            continue
        st = lon[a:b] == int(Long.STATIONARY)
        cuts = [a] + [a + i for i in range(1, b - a) if st[i] != st[i - 1]] + [b]
        pieces = []
        for s, e in zip(cuts[:-1], cuts[1:]):
            pieces += [(s, e)] if v1 or st[s - a] else clean_pieces(lat, s, e)
        for s, e in pieces:
            for s2 in range(s, e, int(STRETCH_MAX * HZ)):
                e2 = min(e, s2 + int(STRETCH_MAX * HZ))
                if e - e2 < STRETCH_MIN * HZ:  # do not leave a too short tail
                    e2 = e
                if e2 - s2 >= STRETCH_MIN * HZ:
                    out.append(("stretch", s2, e2, s2, e2))
                if e2 == e:
                    break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--datasets", default="hdd,bdd,av2,nuscenes,road_waymo,nuplan,pai")
    ap.add_argument("--out", default=f"{ROOT}/benchmark/segments_v2.csv")
    ap.add_argument("--lat-key", default="lat_u2")
    ap.add_argument("--overrides", default="", help="label overrides: CSVs (dataset, clip_id, start, end, lat), comma separated")
    ap.add_argument("--v1", action="store_true", help="v1 rules (stretches with ignore frames, no kinematic checks, DIP kept)")
    a = ap.parse_args()
    idx = filter_index(build_index(a.datasets.split(",")))
    idx = idx[idx.split.isin(a.splits.split(","))]
    val = val_groups(idx)
    print("validation groups:", pd.Series([d_ for d_, _ in val]).value_counts().to_dict())
    over = {}
    for f in filter(None, a.overrides.split(",")):
        for e in pd.read_csv(f, dtype={"clip_id": str}).itertuples():
            c = IGNORE if e.lat == "IGNORE" else int(Lat[e.lat])
            over.setdefault((e.dataset, e.clip_id), []).append((int(round(e.start * HZ)), int(round(e.end * HZ)), c))
    print("label overrides:", sum(map(len, over.values())), "spans in", len(over), "clips")
    rows = []
    for r in idx.itertuples():
        z = np.load(os.path.join(r.dir, "labels.npz"))
        lat = z[a.lat_key].astype(int)
        for s, e, c in over.get((r.dataset, r.clip_id), []):
            lat[s:e] = c
        lon = clean_longitudinal(z["long"], z["kin"])
        drop = frame_mask(z, r)[:len(lat)]
        lat[drop], lon[drop] = IGNORE, IGNORE
        kin = z["kin"]
        bias = yaw_bias(kin, z["long"].astype(int))
        lon_raw = z["long"].astype(int)
        part = "test" if r.split != "train" else "val" if (r.dataset, val_group(r.dataset, r.clip_id)) in val else "train"
        if r.dataset == "hdd" and r.split not in ("train", "test"):
            part = ""  # HDD: official train / test sessions only
        items = []
        for kind, lo, hi, s, e in segments(lat, lon, a.v1):
            ol = segment_longitudinal(lon_raw, kin, lo, hi)
            if ol == "DIP" and not a.v1:
                if kind == "maneuver":
                    items.append((kind, lo, hi, s, e, "MOVING"))  # slows down into / speeds up out of it: lateral only
                else:
                    items += [(kind, a_, b_, a_, b_, lab or "DIP_PIECE") for a_, b_, lab in split_dip(lon_raw, kin, lo, hi)]
            else:
                items.append((kind, lo, hi, s, e, ol))
        for kind, lo, hi, s, e, ol in items:
            seg_lat = lat[s:e].copy()
            if ol == "STATIONARY":  # standing still: no lateral behaviour
                seg_lat[:] = NONE
            lc, lp = purity(seg_lat)
            ign = float(((seg_lat < 0) & (seg_lat != -2)).mean())  # inside the event (boundary bands are context)
            if lc is None or lp < PURITY:
                why = "lateral mixed"
            elif lc in REMOVED:
                why = "class removed"
            elif ign > IGN_MAX:
                why = "ignore"
            elif not ol:
                why = "longitudinal mixed"
            elif ol == "DIP_PIECE":
                why = "slow-down-speed-up piece too short or unclear"
            elif ol == "REVERSE" and not a.v1:
                why = "reverse"  # not a benchmark class (6 longitudinal classes); 18 segments of v2, all training
            else:
                why = ""
            name = {LANE_KEEP: "LANE_KEEP", NONE: "NONE"}.get(lc, Lat(lc).name if lc is not None and lc < 100 else "")
            dpsi = heading_change(kin, lo, hi, bias)
            if not why and not a.v1:
                why = kinematic_conflict(name, dpsi)
            rows.append(dict(dataset=r.dataset, clip_id=r.clip_id, split=r.split, part=part, kind=kind, start=lo / HZ, end=hi / HZ,
                             event_start=s / HZ, event_end=e / HZ, lat=name, long=ol, lat_purity=round(lp, 2),
                             ignore=round(ign, 2), dpsi=None if dpsi is None else round(dpsi, 1), kept=not why, reason=why))
    d = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    d.to_csv(a.out, index=False)
    m = d[d.kind == "maneuver"]
    print(f"maneuver segments {len(m)}: kept {m.kept.mean():.0%}; dropped by", m[~m.kept].reason.value_counts().to_dict())
    s = d[d.kind == "stretch"]
    print(f"lane-keeping segments {len(s)}: kept {s.kept.mean():.0%}; dropped by", s[~s.kept].reason.value_counts().to_dict())
    k = d[d.kept & (d.part != "")]
    print(pd.crosstab(k.lat, [k.part]).to_string())
    print(pd.crosstab(k.long, [k.part]).to_string())
    print("hours kept:", (k.end - k.start).groupby(k.part).sum().div(3600).round(1).to_dict())


if __name__ == "__main__":
    main()
