"""Ego-behavior taxonomy: the single source of truth for class names and ids.

Two per-frame single-label axes (A: longitudinal, B: lateral/path), plus a
multi-label context axis (C) and an optional stop-cause axis (D). See
docs/proposal_v0.md section 2 for definitions.
"""

from enum import IntEnum

IGNORE = -1  # frame excluded from the loss (unknown / boundary band / unsupported)


class Long(IntEnum):
    STATIONARY = 0
    START = 1  # start from stop
    ACCELERATE = 2
    CRUISE = 3
    DECELERATE = 4  # decelerate without stopping
    DECEL_TO_STOP = 5
    REVERSE = 6


class Lat(IntEnum):
    STRAIGHT = 0  # lane keeping on a straight road
    CURVE = 1  # lane keeping along a curved road
    TURN_LEFT = 2
    TURN_RIGHT = 3
    U_TURN = 4
    LANE_CHANGE_LEFT = 5
    LANE_CHANGE_RIGHT = 6
    MERGE_LEFT = 7
    MERGE_RIGHT = 8
    BRANCH_LEFT = 9
    BRANCH_RIGHT = 10
    PULL_OVER = 11
    PULL_OUT = 12
    NUDGE_LEFT = 13  # in-lane offset, returns within the lane
    NUDGE_RIGHT = 14
    BORROW_LEFT = 15  # out-of-lane nudge into the adjacent lane, then returns
    BORROW_RIGHT = 16
    ROUNDABOUT = 17
    # PARKING (18) and ABORT (19) were removed: no data.


CONTEXT = (
    "intersection", "crosswalk", "railroad", "highway", "ramp", "urban",
    "curbside", "parking_lot", "work_zone", "tunnel",
    "day", "dusk", "night",
    "rain", "snow", "fog",
    "left_hand_traffic",
)

STOP_CAUSE = ("red_light", "stop_sign", "lead_vehicle", "pedestrian", "crossing_vehicle", "yield", "other")

# Horizontal flip swaps left/right; pull over/out keep their id (the side flips with traffic handedness).
LAT_MIRROR = {
    Lat.TURN_LEFT: Lat.TURN_RIGHT,
    Lat.LANE_CHANGE_LEFT: Lat.LANE_CHANGE_RIGHT,
    Lat.MERGE_LEFT: Lat.MERGE_RIGHT,
    Lat.BRANCH_LEFT: Lat.BRANCH_RIGHT,
    Lat.NUDGE_LEFT: Lat.NUDGE_RIGHT,
    Lat.BORROW_LEFT: Lat.BORROW_RIGHT,
}
LAT_MIRROR.update({v: k for k, v in list(LAT_MIRROR.items())})

# Lateral classes whose rule needs lane-level map topology; map-free sources mark them IGNORE.
LAT_NEEDS_MAP = frozenset({
    Lat.MERGE_LEFT, Lat.MERGE_RIGHT, Lat.BRANCH_LEFT, Lat.BRANCH_RIGHT, Lat.ROUNDABOUT,
})


def mirror_lat(label: int) -> int:
    if label == IGNORE:
        return label
    return int(LAT_MIRROR.get(Lat(label), Lat(label)))
