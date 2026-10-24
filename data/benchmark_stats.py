"""Benchmark statistics: size per split and dataset, class counts, and diversity (countries
/ cities, left- vs right-hand traffic, camera setups, speed range as a road-type proxy).

Usage: python data/benchmark_stats.py [--segments segments_v2]
"""

import argparse
from egokin.scripts import script_path  # noqa: E402
import json
import os

import numpy as np
import pandas as pd

from egokin.data.index import ROOT, build_index

CAMERA = {"hdd": "HDD front camera", "bdd": "BDD dashcams (phones)", "av2": "AV2 ring front center",
          "nuscenes": "nuScenes CAM_FRONT", "nuplan": "nuPlan front camera", "road_waymo": "Waymo front camera",
          "pai": "PAI front wide 120 deg"}
REGION = {"hdd": "US (SF Bay Area)", "bdd": "US (NYC, SF Bay Area, ...)", "road_waymo": "US (Waymo cities)"}


def place(ds, m):
    if ds == "pai":
        return m.get("country", "")
    if ds == "av2":
        return "US " + m.get("city", "")
    if ds in ("nuplan", "nuscenes"):
        return m.get("location", "")
    return REGION.get(ds, ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", default="segments_v2")
    a = ap.parse_args()
    import importlib.util
    spec = importlib.util.spec_from_file_location("seg", script_path("train_segments.py"))
    seg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(seg)
    d = seg.load_segments(None, a.segments)  # kept segments with their part (train / val / test)
    d["dur"] = d.end - d.start
    idx = build_index(sorted(d.dataset.unique()))
    d = d.merge(idx[["dataset", "clip_id", "dir", "rht"]], on=["dataset", "clip_id"], how="left")
    places = {}
    for (ds, cid, path) in d[["dataset", "clip_id", "dir"]].drop_duplicates().itertuples(index=False):
        places[(ds, cid)] = place(ds, json.load(open(os.path.join(path, "meta.json"))))
    d["place"] = [places[(x, c)] for x, c in zip(d.dataset, d.clip_id)]
    print(f"== {a.segments}: {len(d)} segments, {d.dur.sum() / 3600:.1f} h")
    print(d.pivot_table(index="dataset", columns="part", values="dur", aggfunc=["count", lambda x: round(x.sum() / 3600, 1)],
                        fill_value=0).to_string())
    print("\n== lateral classes")
    print(pd.crosstab(d.lat, d.part, margins=True).to_string())
    print("\n== longitudinal classes")
    print(pd.crosstab(d.long, d.part, margins=True).to_string())
    print("\n== lateral class x dataset (test)")
    print(pd.crosstab(d[d.part == "test"].lat, d[d.part == "test"].dataset).to_string())
    print("\n== diversity")
    print("places:", d.place.nunique(), "| countries / regions (hours):",
          d.groupby("place").dur.sum().div(3600).round(1).sort_values(ascending=False).head(25).to_dict())
    print("left-hand traffic hours:", round(d[~d.rht.astype(bool)].dur.sum() / 3600, 1), "of", round(d.dur.sum() / 3600, 1))
    print("camera setups:", len(set(CAMERA[x] for x in d.dataset.unique())), sorted(set(CAMERA[x] for x in d.dataset.unique())))
    # mean speed per segment as a road-type proxy (no road-type metadata); ROAD-Waymo has no speed
    v = []
    for (ds, path), g in d[d.dataset != "road_waymo"].groupby(["dataset", "dir"]):
        kin = np.load(os.path.join(path, "labels.npz"))["kin"]
        for r in g.itertuples():
            s = kin[int(r.start * 10):int(r.end * 10), 0]
            v.append((r.part, float(np.nanmean(s)) * 3.6 if len(s) else np.nan, r.dur))
    v = pd.DataFrame(v, columns=["part", "kmh", "dur"])
    v["bin"] = pd.cut(v.kmh, [-1, 5, 30, 60, 90, 300], labels=["<5", "5-30", "30-60", "60-90", ">90"])
    print("hours by mean speed (km/h):\n", v.pivot_table(index="bin", columns="part", values="dur", aggfunc="sum",
                                                        observed=False).div(3600).round(1).to_string())


if __name__ == "__main__":
    main()
