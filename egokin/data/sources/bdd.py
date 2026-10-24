"""BDD100K (+ BDD-X) adapter. Videos come from the authors' mirror http://dl.yf.io/bdd100k/
(the official site is down); one .mov is fetched at a time out of the 1.97 TB zip via HTTP ranges.

Ego motion: GPS at 1 Hz (speed, course) + phone gyro (~50 Hz). The phone mount orientation varies,
so the gyro axis/sign used as yaw rate is chosen per video by correlation with the GPS course rate.
"""

import json
import os
import re
import zipfile

import numpy as np
import pandas as pd

from egokin.labels.kinematics import EgoTrack, LABEL_HZ, _savgol

ROOT = os.environ.get("VEGO_DATA", "./vego_data")
MIRROR = "http://dl.yf.io/bdd100k"
RAW = f"{ROOT}/raw/bdd"

# BDD-X action text -> our lateral class (first match wins); reversing goes to the longitudinal axis
BDDX_LAT = [
    ("PULL_OVER", r"pull(s|ed|ing)? (over|to the (side|curb))"),
    ("PULL_OUT", r"pull(s|ed|ing)? (out|away from the curb|into (the )?(traffic|road|lane))"),
    ("U_TURN", r"u-?turn|turn(s|ing)? around"),
    ("LANE_CHANGE_LEFT", r"((change|switch|merg)(es|ing|e|ed)? (in)?to the left( lane)?)|((change|switch)(es|ing)? lanes? (to the )?left)"),
    ("LANE_CHANGE_RIGHT", r"((change|switch|merg)(es|ing|e|ed)? (in)?to the right( lane)?)|((change|switch)(es|ing)? lanes? (to the )?right)"),
    ("TURN_LEFT", r"(turn(s|ing|ed)? (to the )?left)|(left turn)"),
    ("TURN_RIGHT", r"(turn(s|ing|ed)? (to the )?right)|(right turn)"),
]
BDDX_REVERSE = r"revers(e|es|ing)|back(s|ing)? up"


def info_zip():
    return zipfile.ZipFile(f"{RAW}/bdd100k_info.zip")


def read_info(z: zipfile.ZipFile, split: str, name: str) -> dict:
    return json.loads(z.read(f"100k/{split}/{name}.json"))


def track_from_info(info: dict) -> tuple[EgoTrack, dict]:
    """10 Hz ego track (clip time from info startTime) and diagnostics."""
    t0 = info["startTime"] / 1000.0
    dur = (info["endTime"] - info["startTime"]) / 1000.0
    tg = np.arange(0.0, dur, 1.0 / LABEL_HZ)
    loc = pd.DataFrame(info["locations"])
    tl = loc["timestamp"].to_numpy() / 1000.0 - t0
    v = np.interp(tg, tl, loc["speed"].clip(lower=0).to_numpy())
    course = np.radians(np.unwrap(np.degrees(np.radians(loc["course"].to_numpy())), period=360))
    course_g = np.interp(tg, tl, course)
    course_rate = -np.gradient(course_g) * LABEL_HZ  # compass course is clockwise; left turn > 0
    diag = {"gyro_axis": None, "gyro_corr": 0.0}
    w = _savgol(course_rate, 1.5, LABEL_HZ)
    gy = pd.DataFrame(info.get("gyro") or [])
    if len(gy) > 50:
        tgy = gy["timestamp"].to_numpy() / 1000.0 - t0
        best = (0.0, None, 1)
        for ax in ("x", "y", "z"):
            g = np.interp(tg, tgy, gy[ax].to_numpy())
            c = np.corrcoef(g, course_rate)[0, 1] if np.std(g) > 0 and np.std(course_rate) > 0 else 0.0
            if abs(c) > abs(best[0]):
                best = (c, ax, np.sign(c))
        if best[1] is not None and abs(best[0]) > 0.5:  # trust the gyro only if it clearly tracks the course
            w = best[2] * _savgol(np.interp(tg, tgy, gy[best[1]].to_numpy()), 0.5, LABEL_HZ)
            diag = {"gyro_axis": best[1], "gyro_corr": float(best[0])}
    v = np.maximum(_savgol(v, 1.0, LABEL_HZ), 0.0)
    a = _savgol(v, 1.0, LABEL_HZ, deriv=1)
    heading = np.cumsum(w) / LABEL_HZ
    x, y = np.cumsum(v * np.cos(heading)) / LABEL_HZ, np.cumsum(v * np.sin(heading)) / LABEL_HZ
    curv = np.where(v > 1.0, w / np.maximum(v, 1e-3), 0.0)
    return EgoTrack(tg, x, y, heading, v, a, w, curv), diag


def bddx_actions() -> pd.DataFrame:
    df = pd.read_csv(f"{RAW}/bddx/bddx_actions.csv")
    for c in ("start", "end"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["start", "end"])
    nc = lambda p: re.sub(r"\((?!\?)", "(?:", p)  # non-capturing groups (no pandas match-group warning)
    df["lat"] = None
    for name, pat in BDDX_LAT[::-1]:  # reverse so that the first pattern in the list wins
        df.loc[df.action.str.contains(nc(pat), regex=True), "lat"] = name
    df["reverse"] = df.action.str.contains(nc(BDDX_REVERSE), regex=True)
    return df


def fetch_video(split: str, name: str, out_dir: str) -> str:
    from remotezip import RemoteZip

    member = f"bdd100k/videos/100k/{split}/{name}.mov"
    with RemoteZip(f"{MIRROR}/bdd100k_videos.zip") as z:
        z.extract(member, out_dir)
    return os.path.join(out_dir, member)


def label_video(tr: EgoTrack, actions: pd.DataFrame, attrs: dict):
    """Our labels for one BDD video: kinematic A axis + pose turns, overridden by BDD-X human actions."""
    from egokin.labels.lateral_nomap import label_lateral_nomap
    from egokin.labels.longitudinal import label_longitudinal
    from egokin.taxonomy import IGNORE, Lat, Long

    n = len(tr)
    long_lab, strong, hard = label_longitudinal(tr)
    sl = lambda r: slice(max(0, int(r.start * LABEL_HZ)), min(n, int(np.ceil(r.end * LABEL_HZ)) + 1))
    for r in actions[actions.reverse].itertuples():
        long_lab[sl(r)] = Long.REVERSE
    lat = label_lateral_nomap(tr, long_lab, right_hand_traffic=True).labels
    for r in actions[actions.lat.notna()].itertuples():
        lat[sl(r)] = Lat[r.lat]
    lat[long_lab == Long.REVERSE] = IGNORE
    tod = attrs.get("timeofday")
    ctx = {"night": np.full(n, tod == "night"), "dusk": np.full(n, tod == "dawn/dusk"),
           "highway": np.full(n, attrs.get("scene") == "highway"), "tunnel": np.full(n, attrs.get("scene") == "tunnel"),
           "parking_lot": np.full(n, attrs.get("scene") == "parking lot"),
           "rain": np.full(n, attrs.get("weather") == "rainy"), "snow": np.full(n, attrs.get("weather") == "snowy"),
           "fog": np.full(n, attrs.get("weather") == "foggy")}
    return long_lab, lat, strong, hard, ctx
