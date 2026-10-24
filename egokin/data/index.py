"""Dataset index over the unified clip store, with the split policy used for all experiments.

Split policy (also docs/progress.md):
- hdd: official TRN/OadTR split (train 100 / test 36 sessions) -> benchmark comparability.
- pai: train only (license: no published results on PAI).
- av2: official train / val / test logs.
- nuscenes: official train / val scenes.
- bdd: BDD100K split of each video (BDD-X videos keep their BDD100K split).
- road_waymo: ROAD-Waymo train / val.
Our test pool = hdd test + av2 val/test + nuscenes val + bdd val/test (+ road_waymo val).
"""

import glob
import json
import os

import pandas as pd

ROOT = os.environ.get("VEGO_DATA", "./vego_data")


def _split(m: dict) -> str:
    if m.get("split_final"):  # resolved once on the data machine by scripts/write_splits.py
        return m["split_final"]
    ds = m["dataset"]
    if ds == "pai":
        return "train"
    if ds == "hdd":
        return {"train": "train", "test": "test"}.get(m.get("split"), "unused")
    if ds == "nuscenes":
        return m.get("split", "train")
    if ds == "bdd":  # BDD-X videos use the BDD-X split (human action labels); others the BDD100K split
        return _bddx_split().get(m["clip_id"]) or {"train": "train", "val": "val", "test": "test"}.get(m.get("split"), "train")
    return m.get("split", "train")


_BDDX = None


def _bddx_split() -> dict:
    global _BDDX
    if _BDDX is None:
        _BDDX = {}
        base = f"{ROOT}/raw/bdd/bddx/BDD-X-Dataset"
        for sp in ("train", "val", "test"):
            if os.path.exists(f"{base}/{sp}.txt"):
                for line in open(f"{base}/{sp}.txt"):
                    name = line.strip().split("_", 1)[-1].replace(".mov", "")
                    if name:
                        _BDDX[name] = sp
    return _BDDX


def build_index(datasets=None) -> pd.DataFrame:
    rows = []
    for f in glob.glob(f"{ROOT}/clips/*/*/meta.json"):
        ds = f.split("/")[-3]
        if datasets and ds not in datasets:
            continue
        if not os.path.exists(os.path.join(os.path.dirname(f), "labels.npz")):
            continue
        m = json.load(open(f))
        night = m.get("light") == "night" or m.get("night") or (m.get("attrs") or {}).get("timeofday") == "night"
        rows.append(dict(dataset=ds, clip_id=m["clip_id"], dir=os.path.dirname(f), split=_split(m),
                         n_frames=m.get("n_frames", 0), night=bool(night), rht=m.get("rht", True),
                         work_zone=bool(m.get("work_zone") or m.get("has_work_zone"))))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_index()
    print(df.groupby(["dataset", "split"]).agg(clips=("clip_id", "count"), hours=("n_frames", lambda x: round(x.sum() / 36000, 1)),
                                               night=("night", "sum"), work_zone=("work_zone", "sum")))
