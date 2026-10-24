"""Axis B (lateral/path) labels from ego kinematics only (no map). v0 thresholds.

Pose-only lateral displacement cannot separate lane changes, out-of-lane nudges around
obstacles and S-shaped road bends (verified on PAI review sheets). So by default
only turns / U-turns / curve / straight are emitted as frame labels; lane change, nudge,
borrow and pull over/out are kept as low-confidence candidate events (frames -> IGNORE)
for later vision-based verification. Merge/branch/roundabout are never emitted here.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter

from egokin.labels.kinematics import EgoTrack
from egokin.labels.segments import drop_short, fill_gaps, runs
from egokin.taxonomy import IGNORE, Lat, Long


@dataclass
class LatCfg:
    yaw_rate_turn: float = 0.10  # rad/s, turning core
    yaw_rate_turn_edge: float = 0.04  # rad/s, turn segment extends while above
    turn_min_deg: float = 45.0
    turn_max_radius: float = 40.0  # m, tighter than this at peak => turn, else curve
    uturn_min_deg: float = 150.0
    uturn_max_radius: float = 15.0
    curve_min_kappa: float = 1 / 300.0
    curve_min_speed: float = 5.0
    base_kappa_win_s: float = 10.0  # median window for the road-curvature baseline
    dev_highpass_s: float = 12.0  # removes slow heading-deviation drift
    v_lat_on: float = 0.20  # m/s, lateral event core
    v_lat_off: float = 0.05  # m/s, lateral event boundary
    lc_min_disp: float = 2.2  # m
    lc_max_disp: float = 5.0
    lc_min_s: float = 2.0
    lc_max_s: float = 12.0
    nudge_min_disp: float = 0.5
    nudge_max_disp: float = 1.5
    nudge_return_s: float = 12.0
    pair_max_gap_s: float = 3.0  # out-and-back: return starts within this after the first move
    borrow_min_disp: float = 1.5
    pull_min_disp: float = 1.0
    pull_stop_min_s: float = 8.0
    min_speed_lateral: float = 2.0  # m/s, ignore lateral events at crawling speed
    emit_lateral_fine: bool = False  # True: write candidate classes into frame labels


@dataclass
class LatEvent:
    kind: int  # Lat value
    start: int
    end: int
    disp: float  # m, signed lateral displacement (left > 0)
    conf: float
    note: str = ""


@dataclass
class LatResult:
    labels: np.ndarray
    events: list = field(default_factory=list)
    v_lat: np.ndarray = None
    heading_dev: np.ndarray = None


def _lateral_signals(tr: EgoTrack, cfg: LatCfg, turn_mask: np.ndarray):
    hz, dt = tr.hz, 1.0 / tr.hz
    kappa = tr.curvature.copy()
    kappa[turn_mask] = np.nan
    win = max(3, int(cfg.base_kappa_win_s * hz) | 1)
    # robust baseline of road curvature; NaNs (turns) are bridged by interpolation
    idx = np.arange(len(kappa))
    good = ~np.isnan(kappa)
    kb = np.interp(idx, idx[good], kappa[good]) if good.any() else np.zeros_like(kappa)
    kb = median_filter(kb, size=win, mode="nearest")
    rel_rate = tr.yaw_rate - tr.v * kb
    rel_rate[turn_mask] = 0.0
    dev = np.cumsum(rel_rate) * dt
    hp = max(3, int(cfg.dev_highpass_s * hz) | 1)
    dev = dev - median_filter(dev, size=hp, mode="nearest")
    v_lat = tr.v * np.sin(dev)
    v_lat[turn_mask] = 0.0
    return v_lat, dev


def label_lateral_nomap(tr: EgoTrack, long_lab: np.ndarray, right_hand_traffic: bool = True,
                        cfg: LatCfg = LatCfg()) -> LatResult:
    hz, n = tr.hz, len(tr)
    sec = lambda s: max(1, int(round(s * hz)))
    lab = np.full(n, Lat.STRAIGHT, dtype=np.int64)
    events = []

    # 1) turns and U-turns from heading change of high-yaw-rate runs. A run is split where the yaw
    # rate drops below 30% of its peak for >= 1 s, and extended only while it stays above that level
    # (max 3 s): consecutive turns separated by a straight stretch were merged into fake U-turns.
    core = fill_gaps(np.abs(tr.yaw_rate) > cfg.yaw_rate_turn, sec(1.0))
    turn_mask = np.zeros(n, bool)
    pieces = []
    for s, e in runs(core):
        w = np.abs(tr.yaw_rate[s:e])
        low = drop_short(w < 0.3 * w.max(), sec(1.0))
        for a, b in runs(~low):
            pieces.append((s + a, s + b))
    for s, e in pieces:
        edge = max(cfg.yaw_rate_turn_edge, 0.3 * np.abs(tr.yaw_rate[s:e]).max())
        s0, e0 = s, e
        while s > 0 and s0 - s < sec(3.0) and abs(tr.yaw_rate[s - 1]) > edge:
            s -= 1
        while e < n and e - e0 < sec(3.0) and abs(tr.yaw_rate[e]) > edge:
            e += 1
        dpsi = np.degrees(tr.yaw[e - 1] - tr.yaw[s])
        speed = np.abs(tr.v[s:e])
        kap = np.abs(tr.yaw_rate[s:e]) / np.maximum(speed, 0.5)
        min_radius = 1.0 / max(np.percentile(kap, 90), 1e-6)
        if abs(dpsi) >= cfg.uturn_min_deg and min_radius <= cfg.uturn_max_radius and (e - s) <= sec(20.0):
            kind = Lat.U_TURN
        elif abs(dpsi) >= cfg.turn_min_deg and min_radius <= cfg.turn_max_radius:
            kind = Lat.TURN_LEFT if dpsi > 0 else Lat.TURN_RIGHT
        else:
            continue  # large-radius heading change: handled as curve below
        lab[s:e] = kind
        turn_mask[s:e] = True
        events.append(LatEvent(int(kind), s, e, 0.0, 0.8, f"dpsi={dpsi:.0f}deg rmin={min_radius:.0f}m"))

    # 2) curve following
    kap_s = median_filter(np.abs(tr.curvature), size=sec(2.0) | 1, mode="nearest")
    curve = drop_short((kap_s > cfg.curve_min_kappa) & (np.abs(tr.v) > cfg.curve_min_speed) & ~turn_mask, sec(2.0))
    lab[curve] = Lat.CURVE

    # 3) lateral events relative to the road direction: lane change / nudge
    v_lat, dev = _lateral_signals(tr, cfg, turn_mask)
    moving = np.abs(tr.v) > cfg.min_speed_lateral
    on = fill_gaps((np.abs(v_lat) > cfg.v_lat_on) & moving, sec(0.5))
    off = np.abs(v_lat) > cfg.v_lat_off
    lat_events = []
    for s, e in runs(on):
        while s > 0 and off[s - 1] and not turn_mask[s - 1]:
            s -= 1
        while e < n and off[e] and not turn_mask[e]:
            e += 1
        disp = float(np.sum(v_lat[s:e]) / hz)
        lat_events.append((s, e, disp))

    cand = np.zeros(n, bool)  # frames covered by ambiguous lateral candidates
    paired = set()
    for k in range(len(lat_events) - 1):
        # out-and-back: LC pair (overtake), borrow around an obstacle, or an S-bend
        s, e, disp = lat_events[k]
        s2, e2, d2 = lat_events[k + 1]
        if (np.sign(d2) == -np.sign(disp) and (s2 - e) / hz <= cfg.pair_max_gap_s
                and 0.5 <= abs(d2) / max(abs(disp), 1e-3) <= 2.0 and abs(disp) >= cfg.nudge_min_disp):
            if abs(disp) >= cfg.borrow_min_disp:
                kind = Lat.BORROW_LEFT if disp > 0 else Lat.BORROW_RIGHT
            else:
                kind = Lat.NUDGE_LEFT if disp > 0 else Lat.NUDGE_RIGHT
            events.append(LatEvent(int(kind), s, e2, disp, 0.3, "out-and-back (borrow / LC pair / S-bend?)"))
            lab[s:e2] = kind
            cand[s:e2] = True
            paired.update({k, k + 1})
    singles = [ev for k, ev in enumerate(lat_events) if k not in paired]
    for s, e, disp in singles:
        dur = (e - s) / hz
        if cfg.lc_min_disp <= abs(disp) <= cfg.lc_max_disp and cfg.lc_min_s <= dur <= cfg.lc_max_s:
            kind = Lat.LANE_CHANGE_LEFT if disp > 0 else Lat.LANE_CHANGE_RIGHT
            events.append(LatEvent(int(kind), s, e, disp, 0.4, f"dur={dur:.1f}s (LC / bend?)"))
            lab[s:e] = kind
            cand[s:e] = True
        elif abs(disp) >= cfg.nudge_min_disp:
            cand[s:e] = True  # unexplained lateral motion: do not call it straight

    # 4) pull over / pull out candidates (no curb evidence => low confidence)
    toward_curb = -1.0 if right_hand_traffic else 1.0
    stationary = long_lab == Long.STATIONARY
    for s, e, disp in singles:
        if np.sign(disp) == toward_curb and abs(disp) >= cfg.pull_min_disp:
            # followed by a long stop shortly after the lateral move
            j = e
            while j < n and j - e < sec(5.0) and not stationary[j]:
                j += 1
            if j < n and stationary[j]:
                stop_len = next((b - a for a, b in runs(stationary) if a <= j < b), 0)
                if stop_len >= sec(cfg.pull_stop_min_s) or j + stop_len >= n - 1:
                    lab[s:j] = Lat.PULL_OVER
                    cand[s:j] = True
                    events.append(LatEvent(int(Lat.PULL_OVER), s, j, disp, 0.3, "candidate"))
        if np.sign(disp) == -toward_curb and abs(disp) >= cfg.pull_min_disp:
            # preceded by a long stop shortly before the lateral move
            i = s - 1
            while i >= 0 and s - i < sec(8.0) and not stationary[i]:
                i -= 1
            if i >= 0 and stationary[i]:
                stop = next(((a, b) for a, b in runs(stationary) if a <= i < b), None)
                if stop and (stop[1] - stop[0] >= sec(cfg.pull_stop_min_s) or stop[0] == 0):
                    lab[stop[1]:e] = Lat.PULL_OUT
                    cand[stop[1]:e] = True
                    events.append(LatEvent(int(Lat.PULL_OUT), stop[1], e, disp, 0.3, "candidate"))

    if not cfg.emit_lateral_fine:
        lab[cand] = IGNORE
    # lateral behavior is undefined while stationary or reversing
    lab[(long_lab == Long.STATIONARY) & (lab == Lat.STRAIGHT)] = IGNORE
    lab[long_lab == Long.REVERSE] = IGNORE
    return LatResult(lab, events, v_lat, dev)
