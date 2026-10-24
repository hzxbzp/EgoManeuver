"""Ego-behaviour event index for the retrieval agent.

Input: DenseEgo's stage-1 per-step probabilities on the test clips, runs/e2e_off/<run>/pred_e2e/<dataset>__<clip>.npz
(pb: lateral classes of egokin.taxonomy.Lat, pa: longitudinal classes of egokin.taxonomy.Long, one row per 0.2 s step).
An event of class c is a run of steps where the 1 s median of p_c exceeds HI, extended while it stays above LO, with
gaps below GAP_S bridged and at least MIN_S long; its score is the mean probability over the run. Lateral probabilities are zeroed where p(stationary) > 0.5:
the benchmark has no lateral class at standstill, so the head was never trained there.
Output: $VEGO_DATA/agent/ego_events.csv (dataset, clip_id, axis, cls, t0, t1, score; times in clip seconds).

Usage: python agent/agent_index.py [--run trainonly_full_s0]
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from egokin.data.index import ROOT
from egokin.taxonomy import Lat, Long

STEP_S = 0.2
HI, LO = 0.5, 0.3
GAP_S = 0.6
MIN_S = {"lateral": 0.6, "longitudinal": 1.0}
LATERAL = [c for c in Lat if c not in (Lat.STRAIGHT, Lat.ROUNDABOUT, Lat.BORROW_RIGHT)]
LONGITUDINAL = [Long.STATIONARY, Long.START, Long.ACCELERATE, Long.DECELERATE, Long.DECEL_TO_STOP]


def events(p, axis):
    """[(k0, k1, score)] runs of one class probability p (per step)."""
    q = median_filter(p.astype(np.float32), size=5, mode="nearest")
    on = q > LO
    out, k = [], 0
    while k < len(q):
        if q[k] > HI:
            k0 = k
            while k0 > 0 and on[k0 - 1]:
                k0 -= 1
            k1 = k
            while k1 < len(q) and on[k1]:
                k1 += 1
            out.append([k0, k1])
            k = k1
        else:
            k += 1
    merged = []
    for k0, k1 in out:  # bridge short gaps between runs of the same class
        if merged and k0 - merged[-1][1] < GAP_S / STEP_S:
            merged[-1][1] = k1
        else:
            merged.append([k0, k1])
    return [(k0, k1, float(p[k0:k1].mean())) for k0, k1 in merged if (k1 - k0) * STEP_S >= MIN_S[axis]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="trainonly_full_s0")
    a = ap.parse_args()
    rows = []
    files = sorted(glob.glob(f"{ROOT}/runs/e2e_off/{a.run}/pred_e2e/*.npz"))
    for f in files:
        ds, cid = os.path.basename(f)[:-4].split("__", 1)
        z = np.load(f)
        # the benchmark has no lateral class while the vehicle stands; the stage-1 head is not trained there (ignore)
        lat = z["pb"].astype(np.float32) * (z["pa"][:, int(Long.STATIONARY)] < 0.5)[:, None]
        for axis, P, classes in (("lateral", lat, LATERAL), ("longitudinal", z["pa"], LONGITUDINAL)):
            for c in classes:
                for k0, k1, sc in events(P[:, int(c)], axis):
                    rows.append(dict(dataset=ds, clip_id=cid, axis=axis, cls=c.name, t0=round(k0 * STEP_S, 1),
                                     t1=round(k1 * STEP_S, 1), score=round(sc, 3)))
    out = f"{ROOT}/agent"
    os.makedirs(out, exist_ok=True)
    d = pd.DataFrame(rows)
    d.to_csv(f"{out}/ego_events.csv", index=False)
    print(f"{len(files)} clips, {len(d)} events ->", f"{out}/ego_events.csv")
    print(d.groupby(["axis", "cls"]).size().to_string())


if __name__ == "__main__":
    main()
