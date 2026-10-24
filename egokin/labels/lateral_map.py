"""Axis B (lateral/path) labels from ego track + lane graph. v0 thresholds."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from egokin.labels.kinematics import EgoTrack
from egokin.labels.lanegraph import LaneGraph
from egokin.labels.lateral_nomap import LatEvent, LatResult, label_lateral_nomap
from egokin.labels.segments import drop_short, fill_gaps, runs
from egokin.taxonomy import IGNORE, Lat, Long

OFF = -1  # not on any car lane (parking lane, shoulder, driveway, lot)


@dataclass
class MapCfg:
    cand_buffer: float = 0.5
    max_heading_err_deg: float = 60.0
    trans_cost: tuple = (("same", 0.0), ("succ", 0.1), ("pred", 0.3), ("left", 2.0), ("right", 2.0),
                         ("left_succ", 2.2), ("right_succ", 2.2), ("other", 6.0))
    off_emission: float = 3.0
    off_switch: float = 3.0
    lc_max_half_s: float = 6.0
    v_lat_edge: float = 0.05
    borrow_max_hold_s: float = 6.0
    turn_min_deg: float = 45.0
    uturn_min_deg: float = 150.0
    turn_extend_rate: float = 0.04
    split_min_disp: float = 1.0
    nudge_on: float = 0.5
    nudge_off: float = 0.3
    nudge_max_s: float = 12.0
    car_half_width: float = 0.95
    curve_min_kappa: float = 1 / 300.0
    pull_min_disp: float = 0.8
    pull_edge_dist: float = 2.2  # m, ego center to drivable-area boundary at the stop
    pull_min_stop_s: float = 4.0


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def assign_lanes(tr: EgoTrack, g: LaneGraph, cfg: MapCfg):
    """Viterbi lane sequence. Returns lane ids (OFF if none), lateral offsets d and lane headings."""
    n = len(tr)
    tcost = dict(cfg.trans_cost)
    cands, emis = [], []
    for k in range(n):
        ids, costs = [OFF], [cfg.off_emission]
        for lid in g.candidates(tr.x[k], tr.y[k], cfg.cand_buffer):
            lane = g.lanes[lid]
            _, d, hdg = lane.project(tr.x[k], tr.y[k])
            herr = abs(_wrap(tr.yaw[k] - hdg))
            if np.degrees(herr) > cfg.max_heading_err_deg:
                continue
            half = max(lane.polygon.area / max(lane.length, 1.0) / 2, 1.5)
            ids.append(lid)
            costs.append((d / half) ** 2 + 2.0 * herr ** 2)
        cands.append(ids)
        emis.append(np.array(costs))

    acc, back = [emis[0]], []
    for k in range(1, n):
        prev_ids, cur_ids = cands[k - 1], cands[k]
        T = np.empty((len(prev_ids), len(cur_ids)))
        for i, a in enumerate(prev_ids):
            for j, b in enumerate(cur_ids):
                if a == OFF or b == OFF:
                    T[i, j] = 0.0 if a == b else cfg.off_switch
                else:
                    T[i, j] = tcost[g.relation(a, b)]
        tot = acc[-1][:, None] + T
        back.append(np.argmin(tot, axis=0))
        acc.append(tot.min(axis=0) + emis[k])
    j = int(np.argmin(acc[-1]))
    seq = [cands[-1][j]]
    for k in range(n - 2, -1, -1):
        j = int(back[k][j])
        seq.append(cands[k][j])
    seq = np.array(seq[::-1], dtype=object)  # lane ids may be ints (AV2) or string tokens (nuScenes)

    d = np.full(n, np.nan)
    hdg = np.full(n, np.nan)
    for k, lid in enumerate(seq):
        if lid != OFF:
            _, d[k], hdg[k] = g.lanes[lid].project(tr.x[k], tr.y[k])
    # carry the last known lane heading through OFF stretches; smooth away the piecewise-constant
    # heading of coarse centerline polylines (it shows up as a sawtooth in lateral velocity)
    idx = np.arange(n)
    ok = ~np.isnan(hdg)
    if ok.any():
        hdg = np.interp(idx, idx[ok], np.unwrap(hdg[ok]))
        hdg = median_filter(hdg, size=max(3, int(tr.hz * 1.0) | 1), mode="nearest")
        hdg = np.convolve(np.pad(hdg, 5, mode="edge"), np.ones(11) / 11, mode="valid")
    else:
        hdg = tr.yaw.copy()
    return seq, d, hdg


def label_lateral_map(tr: EgoTrack, long_lab: np.ndarray, g: LaneGraph, right_hand_traffic: bool = True,
                      cfg: MapCfg = MapCfg(), lead_queue: np.ndarray | None = None) -> LatResult:
    """lead_queue: optional per-frame bool, a vehicle directly ahead in the ego corridor (queue)."""
    hz, n = tr.hz, len(tr)
    sec = lambda s: max(1, int(round(s * hz)))
    seq, d, lane_hdg = assign_lanes(tr, g, cfg)
    v_lat = tr.v * np.sin(_wrap(tr.yaw - lane_hdg))
    lab = np.full(n, Lat.STRAIGHT, dtype=np.int64)
    events: list[LatEvent] = []
    in_inter = np.array([lid != OFF and g.lanes[lid].is_intersection for lid in seq])

    def lateral_span(k, sign):
        """Extend around frame k while lateral velocity keeps the given sign."""
        s = k
        while s > 0 and k - s < sec(cfg.lc_max_half_s) and sign * v_lat[s - 1] > cfg.v_lat_edge:
            s -= 1
        e = k
        while e < n and e - k < sec(cfg.lc_max_half_s) and sign * v_lat[e] > cfg.v_lat_edge:
            e += 1
        return s, max(e, k + 1)

    # 1) lane changes (and merges where the old lane ends), from neighbor transitions
    changes = []
    for k in range(1, n):
        a, b = seq[k - 1], seq[k]
        if a == OFF or b == OFF:
            continue
        if g.lanes[a].is_intersection or g.lanes[b].is_intersection:
            continue  # parallel connectors inside an intersection are not lane changes
        rel = g.relation(a, b)
        if rel in ("left", "right", "left_succ", "right_succ"):
            sign = 1.0 if rel.startswith("left") else -1.0
            la = g.lanes[a]
            # the old lane joins another lane (merge). Lanes without known successors are not
            # treated as ending: maps are cropped, so a missing successor is usually the crop edge.
            lane_ends = (len(la.successors) > 0 and all(s in g.lanes for s in la.successors)
                         and all(set(g.lanes[s].predecessors) - {a} for s in la.successors))
            changes.append((k, sign, a, lane_ends))
    i = 0
    while i < len(changes):
        k, sign, a, lane_ends = changes[i]
        s, e = lateral_span(k, sign)
        nxt = changes[i + 1] if i + 1 < len(changes) else None
        if nxt and nxt[1] == -sign and (nxt[0] - k) <= sec(cfg.borrow_max_hold_s):
            s2, e2 = lateral_span(nxt[0], -sign)
            kind = Lat.BORROW_LEFT if sign > 0 else Lat.BORROW_RIGHT
            lab[s:e2] = kind
            events.append(LatEvent(int(kind), s, e2, 0.0, 0.8, "map: out-and-back across lane boundary"))
            i += 2
            continue
        if lane_ends:
            kind = Lat.MERGE_LEFT if sign > 0 else Lat.MERGE_RIGHT
        else:
            kind = Lat.LANE_CHANGE_LEFT if sign > 0 else Lat.LANE_CHANGE_RIGHT
        lab[s:e] = kind
        events.append(LatEvent(int(kind), s, e, 0.0, 0.9, f"map: lane {a} -> neighbor"))
        i += 1

    # 2) branch / merge through lane splits and joins (successor transitions outside intersections)
    # Lanes leaving an intersection always have several (connector) predecessors and approach
    # lanes several successors, so only splits/joins between non-intersection lanes count.
    def road_lanes(ids, exclude):
        return [g.lanes[o] for o in ids if o != exclude and o in g.lanes and not g.lanes[o].is_intersection]

    for k in range(1, n):
        a, b = seq[k - 1], seq[k]
        if a == OFF or b == OFF or g.relation(a, b) != "succ":
            continue
        la, lb = g.lanes[a], g.lanes[b]
        if la.is_intersection or lb.is_intersection:
            continue
        lo, hi = max(0, k - sec(2.0)), min(n, k + sec(4.0))
        if (lab[lo:hi] != Lat.STRAIGHT).any() or (np.abs(tr.yaw_rate[lo:hi]) > cfg.turn_extend_rate * 3).any():
            continue
        disp = float(np.sum(v_lat[lo:hi]) / hz)
        if road_lanes(la.successors, b) and abs(disp) >= cfg.split_min_disp:
            kind = Lat.BRANCH_LEFT if disp > 0 else Lat.BRANCH_RIGHT
            lab[lo:hi] = kind
            events.append(LatEvent(int(kind), lo, hi, disp, 0.7, "map: lane split"))
        elif road_lanes(lb.predecessors, a) and abs(disp) >= cfg.split_min_disp:
            kind = Lat.MERGE_LEFT if disp > 0 else Lat.MERGE_RIGHT
            lab[lo:hi] = kind
            events.append(LatEvent(int(kind), lo, hi, disp, 0.7, "map: lane join"))

    # 3) turns / U-turns through intersections; pose-only turns elsewhere (driveways, lots).
    # Each intersection run is extended only while the yaw rate stays above a fraction of its own
    # peak and never into another intersection run (two consecutive right turns were once merged
    # into a fake 217-degree U-turn).
    inter_runs = runs(fill_gaps(in_inter, sec(1.0)))
    owner = np.full(n, -1)
    for r, (s, e) in enumerate(inter_runs):
        owner[s:e] = r
    seen = set()
    for r, (s, e) in enumerate(inter_runs):
        peak = np.abs(tr.yaw_rate[s:e]).max()
        edge = max(cfg.turn_extend_rate, 0.3 * peak)
        s0, e0 = s, e
        while s > 0 and s0 - s < sec(3.0) and abs(tr.yaw_rate[s - 1]) > edge and owner[s - 1] in (-1, r):
            s -= 1
        while e < n and e - e0 < sec(3.0) and abs(tr.yaw_rate[e]) > edge and owner[e] in (-1, r):
            e += 1
        dpsi = np.degrees(tr.yaw[e - 1] - tr.yaw[s])
        speed = np.maximum(np.abs(tr.v[s:e]), 0.5)
        min_radius = 1.0 / max(np.percentile(np.abs(tr.yaw_rate[s:e]) / speed, 90), 1e-6)
        if abs(dpsi) >= cfg.uturn_min_deg and min_radius <= 15.0:
            kind = Lat.U_TURN
        elif abs(dpsi) >= cfg.turn_min_deg:
            kind = Lat.TURN_LEFT if dpsi > 0 else Lat.TURN_RIGHT
        else:
            continue
        if (kind, s, e) in seen:
            continue
        seen.add((kind, s, e))
        lab[s:e] = kind
        events.append(LatEvent(int(kind), s, e, 0.0, 0.9, f"map: intersection dpsi={dpsi:.0f}deg rmin={min_radius:.0f}m"))
    pose_only = label_lateral_nomap(tr, long_lab, right_hand_traffic)
    for ev in pose_only.events:
        if ev.kind in (Lat.TURN_LEFT, Lat.TURN_RIGHT, Lat.U_TURN) and not (lab[ev.start:ev.end] != Lat.STRAIGHT).any():
            lab[ev.start:ev.end] = ev.kind
            events.append(LatEvent(ev.kind, ev.start, ev.end, 0.0, 0.6, "pose-only turn outside intersection"))

    # 4) in-lane nudges / borrows: offset excursions relative to the habitual offset, which is
    # estimated only from settled straight driving (not turns, stops, or just after them)
    near_inter = fill_gaps(in_inter, sec(1.0))
    for s0, e0 in runs(in_inter):
        near_inter[max(0, s0 - sec(3.0)):min(n, e0 + sec(3.0))] = True
    # connector lanes do not match the driven path inside intersections, so no excursions there
    free = (lab == Lat.STRAIGHT) & (np.abs(tr.v) > 3.0) & ~near_inter
    settled = free.copy()
    for s, e in runs(~free):
        settled[s:min(n, e + sec(2.0))] = False
    dd = np.where(np.isnan(d), 0.0, d)
    if settled.sum() < sec(4.0):
        settled = free  # too little settled driving to define a habitual offset
    base = np.median(dd[settled]) if settled.any() else 0.0
    d_rel = dd - base
    for s, e in runs(drop_short((np.abs(d_rel) > cfg.nudge_on) & settled & ~np.isnan(d), sec(1.0))):
        lo, hi = s, e
        while lo > 0 and abs(d_rel[lo - 1]) > cfg.nudge_off and free[lo - 1]:
            lo -= 1
        while hi < n and abs(d_rel[hi]) > cfg.nudge_off and free[hi]:
            hi += 1
        if hi >= n or lo == 0 or (hi - lo) > sec(cfg.nudge_max_s):
            continue  # not a complete excursion inside the clip
        peak = float(d_rel[lo:hi][np.argmax(np.abs(d_rel[lo:hi]))])
        lid = seq[(lo + hi) // 2]
        half = g.lanes[lid].polygon.area / max(g.lanes[lid].length, 1.0) / 2 if lid != OFF else 1.75
        borrow = abs(dd[(lo + hi) // 2]) + cfg.car_half_width > half + 0.2
        kind = (Lat.BORROW_LEFT if peak > 0 else Lat.BORROW_RIGHT) if borrow else (Lat.NUDGE_LEFT if peak > 0 else Lat.NUDGE_RIGHT)
        lab[lo:hi] = kind
        events.append(LatEvent(int(kind), lo, hi, peak, 0.7, "map: in-lane excursion"))

    # 5) curve following along curved lanes
    kap = np.abs(np.gradient(lane_hdg) * hz / np.maximum(np.abs(tr.v), 1.0))
    curve = drop_short((median_filter(kap, size=sec(2.0) | 1) > cfg.curve_min_kappa) & (lab == Lat.STRAIGHT)
                       & (np.abs(tr.v) > 3.0), sec(2.0))
    lab[curve] = Lat.CURVE

    # 6) pull over / pull out. Strict on purpose (AV2 review: queues in the rightmost lane and
    # stop-sign starts followed by turns were false positives). The stop must be off the travel
    # lanes, or hugging the curb in the curb-side lane, away from intersections, without a queued
    # lead vehicle, and the window must not contain a turn.
    toward = -1.0 if right_hand_traffic else 1.0
    stationary = long_lab == Long.STATIONARY

    def curb_stop(k):
        lid = seq[k]
        if lid == OFF:
            return True
        lane = g.lanes[lid]
        curb_side_lane = (lane.right_neighbor if right_hand_traffic else lane.left_neighbor) is None
        off_center = toward * (dd[k] - base) >= 0.7
        return curb_side_lane and off_center and g.dist_to_edge(tr.x[k], tr.y[k]) < cfg.pull_edge_dist

    def turn_free(lo, hi):
        return (lab[lo:hi] != Lat.TURN_LEFT).all() and (lab[lo:hi] != Lat.TURN_RIGHT).all() \
            and (lab[lo:hi] != Lat.U_TURN).all() and abs(np.degrees(tr.yaw[hi - 1] - tr.yaw[lo])) < 30

    for s, e in runs(stationary):
        long_stop = e - s >= sec(cfg.pull_min_stop_s) or (e == n and e - s >= sec(2.0)) or (s == 0 and e - s >= sec(2.0))
        if not long_stop:
            continue
        queued = lead_queue is not None and lead_queue[s:e].any()
        if s > 0 and curb_stop(s) and not near_inter[s] and not queued:
            lo = max(0, s - sec(10.0))
            disp = float(np.sum(v_lat[lo:s]) / hz)
            if toward * disp >= cfg.pull_min_disp and turn_free(lo, s):
                first = next((j for j in range(lo, s) if toward * v_lat[j] > cfg.v_lat_edge), lo)
                lab[first:s] = Lat.PULL_OVER
                events.append(LatEvent(int(Lat.PULL_OVER), first, s, disp, 0.8, "map: curb stop"))
        if e < n and curb_stop(e - 1) and not near_inter[e - 1]:
            hi = min(n, e + sec(10.0))
            disp = float(np.sum(v_lat[e:hi]) / hz)
            if -toward * disp >= cfg.pull_min_disp and turn_free(e, hi):
                last = max((j for j in range(e, hi) if -toward * v_lat[j] > cfg.v_lat_edge), default=hi - 1) + 1
                lab[e:last] = Lat.PULL_OUT
                events.append(LatEvent(int(Lat.PULL_OUT), e, last, disp, 0.8, "map: leave curb after stop"))

    lab[(long_lab == Long.STATIONARY) & (lab == Lat.STRAIGHT)] = IGNORE
    lab[long_lab == Long.REVERSE] = IGNORE
    # keep only events that still own most of their frames after later steps overwrote labels
    events = [ev for ev in events if np.mean(lab[ev.start:ev.end] == ev.kind) >= 0.5]
    res = LatResult(lab, sorted(events, key=lambda ev: ev.start), v_lat, None)
    res.lane_seq, res.lane_offset, res.in_intersection = seq, d, in_inter
    res.road = road_state(tr, g, seq, d, lane_hdg, in_inter)
    return res


ROAD_FIELDS = ("lane_offset", "lane_half_width", "dist_road_edge", "heading_dev", "in_intersection")


def road_state(tr: EgoTrack, g: LaneGraph, seq, d, lane_hdg, in_inter) -> np.ndarray:
    """Per-frame ego state relative to the road ("anchors" = current lane and road edge), NaN when off-lane.
    Columns: ROAD_FIELDS. Used as auxiliary targets / bottleneck inputs (never as model input)."""
    from shapely.geometry import Point

    n = len(tr)
    out = np.full((n, len(ROAD_FIELDS)), np.nan, np.float32)
    edge = g.local_edge(tr.x, tr.y)
    for k in range(n):
        lid = seq[k]
        if lid != OFF:
            lane = g.lanes[lid]
            out[k, 0] = d[k]
            out[k, 1] = lane.polygon.area / max(lane.length, 1.0) / 2
        out[k, 3] = _wrap(tr.yaw[k] - lane_hdg[k]) if lid != OFF else np.nan
        if edge is not None and not edge.is_empty:
            out[k, 2] = edge.distance(Point(tr.x[k], tr.y[k]))
        out[k, 4] = float(in_inter[k])
    return out
