"""Planar ego kinematics from a pose time series, resampled to a uniform grid."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

LABEL_HZ = 10.0  # kinematics grid; behavior labels are later pooled to the model rate


@dataclass
class EgoTrack:
    t: np.ndarray  # s, uniform grid starting at 0
    x: np.ndarray  # m, planar frame (any fixed origin)
    y: np.ndarray
    yaw: np.ndarray  # rad, unwrapped heading of the vehicle body
    v: np.ndarray  # m/s along the body heading; negative when reversing
    a: np.ndarray  # m/s^2, derivative of v
    yaw_rate: np.ndarray  # rad/s
    curvature: np.ndarray  # 1/m; 0 where the vehicle is (nearly) stopped

    @property
    def hz(self) -> float:
        return 1.0 / float(self.t[1] - self.t[0])

    def __len__(self) -> int:
        return len(self.t)


def yaw_from_quat(qx, qy, qz, qw):
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _savgol(x, window_s, hz, deriv=0, order=2):
    n = len(x)
    win = max(order + 2, int(round(window_s * hz)) | 1)
    if win > n:
        win = n if n % 2 else n - 1
    if win <= order:
        return np.gradient(x, 1.0 / hz) if deriv else x.copy()
    return savgol_filter(x, win, order, deriv=deriv, delta=1.0 / hz, mode="interp")


def track_from_poses(t, x, y, yaw, hz=LABEL_HZ, smooth_s=0.7) -> EgoTrack:
    """t in seconds (any offset), x/y in meters, yaw in radians (wrapped or not)."""
    t = np.asarray(t, dtype=np.float64)
    order = np.argsort(t)
    t, x, y = t[order], np.asarray(x, float)[order], np.asarray(y, float)[order]
    yaw = np.unwrap(np.asarray(yaw, float)[order])
    keep = np.concatenate([[True], np.diff(t) > 1e-6])
    t, x, y, yaw = t[keep], x[keep], y[keep], yaw[keep]

    tg = np.arange(0.0, t[-1] - t[0] + 1e-9, 1.0 / hz)
    ts = t - t[0]
    xg, yg, yawg = (np.interp(tg, ts, s) for s in (x, y, yaw))

    xs = _savgol(xg, smooth_s, hz)
    ys = _savgol(yg, smooth_s, hz)
    yaws = _savgol(yawg, smooth_s, hz)
    vx = _savgol(xg, smooth_s, hz, deriv=1)
    vy = _savgol(yg, smooth_s, hz, deriv=1)
    v = vx * np.cos(yaws) + vy * np.sin(yaws)  # signed: <0 when moving backwards
    a = _savgol(v, smooth_s, hz, deriv=1)
    yaw_rate = _savgol(yawg, smooth_s, hz, deriv=1)
    speed = np.abs(v)
    curvature = np.where(speed > 1.0, yaw_rate / np.maximum(speed, 1e-3), 0.0)
    return EgoTrack(tg, xs, ys, yaws, v, a, yaw_rate, curvature)


def heading_change(track: EgoTrack, horizon_s: float) -> np.ndarray:
    """Signed yaw change over a centered window of `horizon_s` seconds (left turn > 0)."""
    k = max(1, int(round(horizon_s * track.hz / 2)))
    yaw = track.yaw
    idx = np.arange(len(yaw))
    lo, hi = np.clip(idx - k, 0, len(yaw) - 1), np.clip(idx + k, 0, len(yaw) - 1)
    return yaw[hi] - yaw[lo]
