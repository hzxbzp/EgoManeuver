"""Honda Research Institute Driving Dataset (HDD) adapter.

Layout (release_2019_07_08): <date>_ITS1/<session>/{general/csv/*.csv, camera/center/<start>_new_0.75.mp4}
and EAF/<start>.eaf (ELAN, times in ms from the video start). The official benchmark uses the
3 fps goal-oriented targets in data/target/<session>.npy (TRN protocol); here we additionally map
the raw EAF tiers (stimulus, cause, attention, area, notes) into our taxonomy.

CAN quirks: vel.csv speed is km/h, yaw.csv is deg/s, rtk_pos.csv has its lng/lat headers swapped.
"""

import glob
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from egokin.labels.kinematics import EgoTrack, LABEL_HZ, _savgol

ROOT = os.environ.get("VEGO_DATA", "./vego_data")
RAW = f"{ROOT}/raw/hdd/data/raw/release_2019_07_08"

GOAL_CLASSES = ["background", "intersection passing", "left turn", "right turn", "left lane change",
                "right lane change", "left lane branch", "right lane branch", "crosswalk passing",
                "railroad passing", "merge", "U-turn"]


def sessions() -> dict[str, str]:
    """session id (e.g. 201703061033) -> session dir."""
    out = {}
    for d in glob.glob(f"{RAW}/*_ITS1/*/"):
        out[os.path.basename(d.rstrip("/"))] = d
    return dict(sorted(out.items()))


def video_path(sdir: str) -> str:
    return glob.glob(os.path.join(sdir, "camera", "center", "*.mp4"))[0]


def eaf_path(sdir: str) -> str:
    start = os.path.basename(video_path(sdir)).split("_new")[0]
    return os.path.join(RAW, "EAF", f"{start}.eaf")


def frame_times(session: str) -> np.ndarray:
    """Unix time of every 30 fps video frame (from the 2019_07_25 release)."""
    f = glob.glob(f"{ROOT}/raw/hdd/data/release_2019_07_25/*/{session}/camera/center/png_timestamp.csv")[0]
    return pd.read_csv(f).iloc[:, 0].to_numpy(np.float64)


def index_time_to_real(session: str, t_index) -> np.ndarray:
    """Map "frame-index time" (frame / 30, used by the EAF annotations and the official TRN 3 fps
    images/targets) to real time since the first frame. The recordings drop frames (up to ~25 s per
    session), so the two clocks drift apart by seconds; see docs/hdd_alignment.md."""
    ft = frame_times(session)
    f = np.clip(np.round(np.asarray(t_index, float) * 30).astype(int), 0, len(ft) - 1)
    return ft[f] - ft[0]


def trn_frame_real_times(session: str, n: int) -> np.ndarray:
    """Real time of official 3 fps frame i (= video frame 10 i)."""
    ft = frame_times(session)
    return ft[np.clip(np.arange(n) * 10, 0, len(ft) - 1)] - ft[0]


def read_csv(sdir: str, name: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(sdir, "general", "csv", f"{name}.csv"))
    df.columns = [c.strip("# ").strip() for c in df.columns]
    return df


def read_eaf(path: str, session: str | None = None) -> pd.DataFrame:
    """Rows (tier, label, t0, t1); labels keep the English part. Times are frame-index seconds, or
    real seconds since the first frame when `session` is given (recommended)."""
    root = ET.parse(path).getroot()
    slots = {s.get("TIME_SLOT_ID"): int(s.get("TIME_VALUE", 0)) for s in root.find("TIME_ORDER")}
    rows = []
    for tier in root.findall("TIER"):
        tid = tier.get("TIER_ID")
        for a in tier.iter("ALIGNABLE_ANNOTATION"):
            text = (a.findtext("ANNOTATION_VALUE") or "").strip()
            eng = re.sub(r"^.*[\u4e00-\u9fff]\s*", "", text).strip().lower()  # English part after the last CJK char
            rows.append((tid, eng or text, slots[a.get("TIME_SLOT_REF1")] / 1000.0, slots[a.get("TIME_SLOT_REF2")] / 1000.0))
    df = pd.DataFrame(rows, columns=["tier", "label", "t0", "t1"])
    if session is not None and len(df):
        df["t0"] = index_time_to_real(session, df["t0"].to_numpy())
        df["t1"] = index_time_to_real(session, df["t1"].to_numpy())
    return df


def load_track(session: str, sdir: str) -> tuple[EgoTrack, float]:
    """Ego track from CAN speed + yaw rate on the 10 Hz grid starting at the first video frame."""
    t_video0 = frame_times(session)[0]
    vel, yaw = read_csv(sdir, "vel"), read_csv(sdir, "yaw")
    t_end = min(vel.unix_timestamp.iloc[-1], yaw.unix_timestamp.iloc[-1]) - t_video0
    tg = np.arange(0.0, t_end, 1.0 / LABEL_HZ)
    v = np.interp(tg, vel.unix_timestamp - t_video0, vel.iloc[:, 2].to_numpy() / 3.6)
    w = -np.interp(tg, yaw.unix_timestamp - t_video0, np.radians(yaw.iloc[:, 2].to_numpy()))  # CAN yaw is clockwise-positive
    v = np.maximum(_savgol(v, 0.7, LABEL_HZ), 0.0)  # speed has no sign; reversing comes from the notes tier
    w = _savgol(w, 0.7, LABEL_HZ)
    a = _savgol(v, 0.7, LABEL_HZ, deriv=1)
    heading = np.cumsum(w) / LABEL_HZ
    x = np.cumsum(v * np.cos(heading)) / LABEL_HZ
    y = np.cumsum(v * np.sin(heading)) / LABEL_HZ
    curv = np.where(v > 1.0, w / np.maximum(v, 1e-3), 0.0)
    return EgoTrack(tg, x, y, heading, v, a, w, curv), t_video0


def turn_signal(sdir: str, t_video0: float, tg: np.ndarray) -> np.ndarray:
    """-1 right, +1 left, 0 none on the label grid."""
    ts = read_csv(sdir, "turn_signal")
    t = ts.unix_timestamp.to_numpy() - t_video0
    left = np.interp(tg, t, ts.iloc[:, 2].to_numpy()) > 0.5
    right = np.interp(tg, t, ts.iloc[:, 3].to_numpy()) > 0.5
    return left.astype(int) - right.astype(int)


# EAF label (English part, lower case) -> our classes
GOAL_TO_LAT = {"left turn": "TURN_LEFT", "right turn": "TURN_RIGHT", "u-turn": "U_TURN",
               "left lane change": "LANE_CHANGE_LEFT", "right lane change": "LANE_CHANGE_RIGHT",
               "left lane branch": "BRANCH_LEFT", "right lane branch": "BRANCH_RIGHT"}
GOAL_TO_CTX = {"intersection passing": "intersection", "crosswalk passing": "crosswalk", "railroad passing": "railroad"}
AVOID = {"avoid parked car", "avoid tp", "avoid pedestrian near ego lane", "avoid on-road bicyclist"}
CAUSE_TO_STOP = {"sign": "stop_sign", "red light": "red_light", "yellow light": "red_light", "congestion": "lead_vehicle",
                 "crossing pedestrian": "pedestrian", "pedestrian near ego lane": "pedestrian",
                 "crossing vehicle": "crossing_vehicle", "turning vehicle": "yield", "merging vehicle": "yield"}
STIM_TO_STOP = {"stop 4 sign": "stop_sign", "stop 4 light": "red_light", "stop 4 congestion": "lead_vehicle",
                "stop 4 pedestrian": "pedestrian", "stop for others": "other"}
AREA_TO_CTX = {"freeway": "highway", "downtown": "urban", "tunnel": "tunnel"}


def label_session(session: str, sdir: str):
    """Our per-frame labels (10 Hz, video clock) for one HDD session from CAN + raw EAF tiers."""
    from egokin.labels.lateral_nomap import label_lateral_nomap
    from egokin.labels.longitudinal import label_longitudinal
    from egokin.taxonomy import IGNORE, STOP_CAUSE, Lat, Long

    tr, t0 = load_track(session, sdir)
    n, hz = len(tr), tr.hz
    ann = read_eaf(eaf_path(sdir), session)
    sl = lambda r: slice(max(0, int(r.t0 * hz)), min(n, int(np.ceil(r.t1 * hz))))

    long_lab, strong, hard = label_longitudinal(tr)
    for r in ann[ann.label.str.contains("倒车")].itertuples():  # reversing is only in the notes tier
        long_lab[sl(r)] = Long.REVERSE
    pose = label_lateral_nomap(tr, long_lab, right_hand_traffic=True)
    lat = pose.labels.copy()
    lat[np.isin(lat, [Lat.TURN_LEFT, Lat.TURN_RIGHT, Lat.U_TURN])] = Lat.STRAIGHT  # humans label turns
    signal = turn_signal(sdir, t0, tr.t)
    ctx = {k: np.zeros(n, bool) for k in ("intersection", "crosswalk", "railroad", "highway", "urban", "tunnel", "work_zone")}
    stop_cause = np.full(n, -1, np.int8)

    for r in ann.itertuples():
        s = sl(r)
        if r.label in GOAL_TO_LAT:
            lat[s] = Lat[GOAL_TO_LAT[r.label]]
        elif r.label == "merge":
            sig = np.sign(signal[s].sum()) or np.sign(tr.yaw[s][-1] - tr.yaw[s][0])
            lat[s] = Lat.MERGE_LEFT if sig > 0 else Lat.MERGE_RIGHT if sig < 0 else IGNORE
        elif r.label in ("park", "park park"):
            lat[s] = Lat.PULL_OVER
        elif r.label in GOAL_TO_CTX:
            ctx[GOAL_TO_CTX[r.label]][s] = True
        elif r.label in AVOID:
            disp = np.cumsum(pose.v_lat[s]) / hz
            peak = disp[np.argmax(np.abs(disp))] if len(disp) else 0.0
            left = peak > 0
            lat[s] = (Lat.BORROW_LEFT if left else Lat.BORROW_RIGHT) if abs(peak) >= 1.5 else (Lat.NUDGE_LEFT if left else Lat.NUDGE_RIGHT)
        elif r.label in STIM_TO_STOP and stop_cause[s].max() < 0:
            stop_cause[s] = STOP_CAUSE.index(STIM_TO_STOP[r.label])
        elif r.label == "road work":
            ctx["work_zone"][max(0, s.start - int(2 * hz)):min(n, s.stop + int(2 * hz))] = True
        elif r.label in AREA_TO_CTX:
            ctx[AREA_TO_CTX[r.label]][s] = True
        elif r.label in ("roundabout",) and (r.t1 - r.t0) > 2.0:
            lat[s] = Lat.ROUNDABOUT
    for r in ann[ann.tier.str.contains("Cause")].itertuples():  # cause refines the stop reason
        if r.label in CAUSE_TO_STOP:
            s = sl(r)
            stopped = stop_cause[s] >= 0
            stop_cause[s][stopped] = STOP_CAUSE.index(CAUSE_TO_STOP[r.label])
    lat[long_lab == Long.REVERSE] = IGNORE
    return tr, t0, long_lab, lat, strong, hard, ctx, stop_cause, ann
