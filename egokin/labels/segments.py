"""Helpers for converting between per-frame masks/labels and [start, end) segments."""

import numpy as np

from egokin.taxonomy import IGNORE


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of consecutive True values."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def drop_short(mask: np.ndarray, min_len: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    for s, e in runs(mask):
        if e - s >= min_len:
            out[s:e] = True
    return out


def fill_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Close False gaps of length <= max_gap that are surrounded by True."""
    out = np.asarray(mask, bool).copy()
    for s, e in runs(~out):
        if s > 0 and e < len(out) and e - s <= max_gap:
            out[s:e] = True
    return out


def to_segments(labels: np.ndarray, skip=(IGNORE,)) -> list[tuple[int, int, int]]:
    """(label, start, end) for each run of identical labels, excluding `skip` labels."""
    labels = np.asarray(labels)
    out = []
    if len(labels) == 0:
        return out
    change = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(labels)]])
    for s, e in zip(starts, ends):
        if labels[s] not in skip:
            out.append((int(labels[s]), int(s), int(e)))
    return out


def ignore_boundaries(labels: np.ndarray, band: int) -> np.ndarray:
    """Mark +-band frames around every label transition as IGNORE (fuzzy boundaries)."""
    out = np.asarray(labels).copy()
    for i in np.flatnonzero(np.diff(labels)) + 1:
        out[max(0, i - band):i + band] = IGNORE
    return out
