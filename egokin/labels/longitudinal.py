"""Axis A (longitudinal) labels from ego kinematics. Thresholds are v0; tune after spot checks."""

from dataclasses import dataclass

import numpy as np

from egokin.labels.kinematics import EgoTrack
from egokin.labels.segments import drop_short, fill_gaps, runs
from egokin.taxonomy import Long


@dataclass
class LongCfg:
    v_stop: float = 0.3  # m/s, below = stationary
    min_stop_s: float = 1.0
    v_start_done: float = 3.0  # m/s, a start ends when this speed is reached
    max_start_s: float = 4.0
    v_reverse: float = -0.2
    min_reverse_s: float = 0.5
    a_brake: float = -0.3  # decel-to-stop extends backwards while a < a_brake
    max_brake_s: float = 15.0
    brake_tolerance_s: float = 0.3  # brief non-braking blips allowed inside a braking run
    a_dyn: float = 0.5  # |a| above this for min_dyn_s => accelerate / decelerate
    min_dyn_s: float = 1.0
    a_strong: float = 2.0  # intensity attribute
    a_hard_brake: float = -3.0


def label_longitudinal(tr: EgoTrack, cfg: LongCfg = LongCfg()):
    """Returns (labels[int], strong[bool], hard_brake[bool]) per kinematics frame."""
    hz = tr.hz
    n = len(tr)
    lab = np.full(n, Long.CRUISE, dtype=np.int64)
    sec = lambda s: max(1, int(round(s * hz)))

    stationary = fill_gaps(np.abs(tr.v) < cfg.v_stop, sec(0.5))
    stationary = drop_short(stationary, sec(cfg.min_stop_s))
    reverse = drop_short((tr.v < cfg.v_reverse) & ~stationary, sec(cfg.min_reverse_s))

    # accelerate / decelerate first; stop-related phases overwrite them below
    lab[drop_short(tr.a > cfg.a_dyn, sec(cfg.min_dyn_s))] = Long.ACCELERATE
    lab[drop_short(tr.a < -cfg.a_dyn, sec(cfg.min_dyn_s))] = Long.DECELERATE

    for s, e in runs(stationary):
        # decelerate-to-stop: walk back from the stop while braking (tolerating short blips)
        if s > 0:
            i, blip, first = s - 1, 0, s
            lo = max(0, s - sec(cfg.max_brake_s))
            while i >= lo and tr.v[i] > cfg.v_stop:
                if tr.a[i] < cfg.a_brake:
                    first, blip = i, 0
                else:
                    blip += 1
                    if blip > sec(cfg.brake_tolerance_s):
                        break
                i -= 1
            if s - first >= sec(1.0):
                lab[first:s] = Long.DECEL_TO_STOP
        # start from stop: from the end of the stop until cruising speed (forward only)
        if e < n and tr.v[e] >= 0:
            j = e
            while j < min(n, e + sec(cfg.max_start_s)) and tr.v[j] < cfg.v_start_done and tr.v[j] >= 0:
                j += 1
            if j - e >= sec(0.5):
                lab[e:j] = Long.START

    lab[stationary] = Long.STATIONARY
    lab[reverse] = Long.REVERSE
    strong = np.abs(tr.a) > cfg.a_strong
    hard_brake = tr.a < cfg.a_hard_brake
    return lab, strong, hard_brake
