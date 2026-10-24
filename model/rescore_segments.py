"""Re-score saved segment predictions (<tag>_s<seed>_pred.npz with segment keys) on another benchmark version, e.g. after
test-label corrections (relabelled or dropped test segments keep their keys), and optionally with longitudinal classes
merged at evaluation time (--merge DIP:CRUISE adds the probability of DIP to CRUISE).

Usage: python model/rescore_segments.py --tags v2_ours_ft,v2_videomae_b --segments segments_v2 [--merge DIP:CRUISE]
"""

import argparse
import glob
import importlib.util
from egokin.scripts import script_path  # noqa: E402
import json
import os

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("seg", script_path("train_segments.py"))
seg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--segments", default="segments_v2")
    ap.add_argument("--merge", default="", help="longitudinal merges at evaluation time, e.g. DIP:CRUISE")
    ap.add_argument("--out", default="", help="JSON with the per-run results")
    a = ap.parse_args()
    d = seg.load_segments(None, a.segments)
    te = d[d.part == "test"].reset_index(drop=True)
    keys = seg.seg_keys(te)
    merges = [m.split(":") for m in a.merge.split(",") if m]
    for src, dst in merges:
        te.loc[te.long == src, "long"] = dst
    res = {}
    for tag in a.tags.split(","):
        for f in sorted(glob.glob(f"{seg.ROOT}/runs/segments/{tag}_s[0-9]_pred.npz")):
            z = np.load(f, allow_pickle=True)
            pos = {k: i for i, k in enumerate(z["key"])}
            hit = np.array([k in pos for k in keys])
            ix = np.array([pos[k] for k in keys[hit]])
            PL, PO = z["lat"].astype(np.float32)[ix], z["long"].astype(np.float32)[ix]
            for src, dst in merges:
                PO[:, seg.LONG.index(dst)] += PO[:, seg.LONG.index(src)]
                PO[:, seg.LONG.index(src)] = 0
            m = seg.metrics(te[hit].reset_index(drop=True), PL, PO)
            run = os.path.basename(f)[:-9]
            res[run] = dict(coverage=float(hit.mean()), **m)
            print(f"{run:28s} coverage {hit.mean():.3f} | lateral mAP {m['lateral']['mAP'] * 100:.1f} | "
                  f"longitudinal mAP {m['longitudinal']['mAP'] * 100:.1f}", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
