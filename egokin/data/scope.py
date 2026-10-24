"""Data scope for training and evaluation: night and work-zone data are isolated for now.

VEGO_SCOPE=day_nowz (default):
- drop clips flagged night (metadata or sky luminance; dusk is kept; BDD clips annotated dawn/dusk or undefined
  whose sky luminance is < 60 carry meta light=night);
- drop short clips flagged as work zone, and EPOSH (work zones only);
- in long sessions (HDD, HSD) keep the session but exclude flagged work-zone frames (+-2 s) and night frames.
VEGO_SCOPE=all: everything.

Exempt: HDD official test sessions are used exactly as published: no frame is masked there, in the
official benchmark or in any evaluation on those sessions.

Only flagged data can be isolated: work zones are flagged in HDD (frames), nuScenes (clips), HSD (frames) and
EPOSH; PAI, BDD, nuPlan, AV2 and ROAD-Waymo have no work-zone flags. Feature caching ignores the scope.
"""

import os

import numpy as np

LONG_SESSIONS = {"hdd", "hsd"}
EXEMPT = {("hdd", "test")}  # (dataset, split) never masked
WORK_ZONE_ONLY = {"eposh"}
MARGIN = 20  # 10 Hz frames around flagged work-zone frames (2 s)


def scope() -> str:
    s = os.environ.get("VEGO_SCOPE", "day_nowz")
    if s not in ("day_nowz", "all"):
        raise ValueError(f"VEGO_SCOPE={s!r}: expected day_nowz or all")
    return s


def filter_index(idx):
    """Drop clips outside the scope (clip-level flags)."""
    if scope() == "all" or not len(idx):
        return idx
    drop = idx.night | (idx.work_zone & ~idx.dataset.isin(LONG_SESSIONS)) | idx.dataset.isin(WORK_ZONE_ONLY)
    return idx[~drop]


def exempt(r) -> bool:
    """r: index row / object with .dataset and .split (None = not exempt)."""
    return r is not None and (getattr(r, "dataset", None), getattr(r, "split", None)) in EXEMPT


def frame_mask(z, r=None) -> np.ndarray:
    """(n_frames,) bool, True for 10 Hz frames to exclude inside kept clips (frame-level flags in labels.npz).
    r: the clip's index row, needed for the HDD-test exemption."""
    n = len(z["long"])
    out = np.zeros(n, bool)
    if scope() == "all" or exempt(r):
        return out
    if "ctx_work_zone" in z.files:
        w = np.asarray(z["ctx_work_zone"]).astype(bool)[:n]
        if w.any():
            out[:len(w)] |= np.convolve(w.astype(float), np.ones(2 * MARGIN + 1), mode="same") > 0
    if "ctx_night" in z.files:
        out[:n] |= np.asarray(z["ctx_night"]).astype(bool)[:n]
    return out


def official_keep(z, r=None) -> np.ndarray:
    """(n_official,) bool: HDD official evaluation frames inside the scope (official_t in seconds of clip time).
    All True for the exempt HDD test sessions."""
    t = np.asarray(z["official_t"], np.float64)
    m = frame_mask(z, r)
    return ~m[np.clip(np.round(t * 10).astype(int), 0, len(m) - 1)]
