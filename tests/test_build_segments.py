"""Benchmark v2 lane-keeping stretches (data/build_segments.py): ignore frames at a stretch's ends are trimmed off,
unresolved lateral motion inside it is cut out with PAD seconds on each side; v1 keeps the stretch whole."""

import importlib.util
import os

import numpy as np

from egokin.taxonomy import IGNORE, Lat, Long

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("build_segments", os.path.join(ROOT, "data", "build_segments.py"))
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)


def test_clean_pieces_trims_ends_and_cuts_inner_ignore():
    lat = np.full(200, int(Lat.STRAIGHT))
    lat[:5] = IGNORE  # boundary band of a previous maneuver
    lat[100:110] = IGNORE  # unresolved lateral move
    lat[195:] = IGNORE
    assert bs.clean_pieces(lat, 0, 200) == [(5, 90), (120, 195)]


def test_stretches_v1_vs_v2():
    n = 300
    lat = np.full(n, int(Lat.STRAIGHT))
    lon = np.full(n, int(Long.CRUISE))
    lat[150:160] = IGNORE
    v1 = [(lo, hi) for kind, lo, hi, _, _ in bs.segments(lat, lon, v1=True) if kind == "stretch"]
    v2 = [(lo, hi) for kind, lo, hi, _, _ in bs.segments(lat, lon) if kind == "stretch"]
    assert v1 == [(0, 100), (100, 200), (200, 300)]
    assert v2 == [(0, 100), (100, 140), (170, 270), (270, 300)]  # 1 s cut on each side of the ignore run


def test_stationary_stretch_is_not_trimmed():
    n = 100
    lat = np.full(n, IGNORE)  # lateral behaviour is undefined while standing
    lon = np.full(n, int(Long.STATIONARY))
    assert [(lo, hi) for kind, lo, hi, _, _ in bs.segments(lat, lon) if kind == "stretch"] == [(0, 100)]


def test_kinematic_conflict():
    assert bs.kinematic_conflict("STRAIGHT", 50.0) == "kinematics: turning"
    assert bs.kinematic_conflict("STRAIGHT", -10.0) == ""
    assert bs.kinematic_conflict("TURN_LEFT", 80.0) == ""
    assert bs.kinematic_conflict("TURN_LEFT", -80.0) == "kinematics: no such turn"  # wrong direction
    assert bs.kinematic_conflict("TURN_RIGHT", -5.0) == "kinematics: no such turn"
    assert bs.kinematic_conflict("U_TURN", -170.0) == ""
    assert bs.kinematic_conflict("NONE", 20.0) == "kinematics: turning"
    assert bs.kinematic_conflict("LANE_CHANGE_LEFT", 10.0) == ""
    assert bs.kinematic_conflict("TURN_LEFT", None) == ""  # no yaw data: not checked


def test_heading_change_removes_bias_and_skips_missing_yaw():
    kin = np.zeros((100, 4))
    kin[:, 2] = 0.01  # rad/s offset only
    long = np.full(100, int(Long.STATIONARY))
    b = bs.yaw_bias(kin, long)
    assert abs(bs.heading_change(kin, 0, 100, b)) < 1e-9
    assert bs.heading_change(np.zeros((50, 4)), 0, 50) is None
