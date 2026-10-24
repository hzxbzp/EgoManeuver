"""Released benchmark (segments_v2_2) from the rule-based build (segments_v2, data/build_segments.py) and the lists of
manual corrections, all keyed by segment or event span so that segment keys never change:

1. benchmark/bdd_motion_dropped.csv  BDD-X segments labelled as moving whose frames do not change (image-motion check,
                                     segment span) -> dropped, reason "image motion: standing still";
2. benchmark/trainval_corrections.csv  expert review of the training / validation nudges and borrows (event span):
                                     new lateral class, or IGNORE -> dropped, reason "expert review: unsure / no maneuver";
3. benchmark/audit_corrections.csv   audit of the test labels (event span): new lateral class, or IGNORE -> dropped,
                                     reason "audit: lat unsure"; lane keeping becomes kind "stretch";
4. reverse is not a benchmark class: segments labelled REVERSE are dropped, reason "reverse".

The test corrections of the expert review (benchmark/test_corrections.csv) enter earlier, as --overrides of
build_segments.py. Usage: python data/finalize_benchmark.py [--src segments_v2] [--out segments_v2_2]
"""

import argparse

import pandas as pd

from egokin.data.index import ROOT

fl = lambda x: f"{float(x):.1f}"
seg_key = lambda d: d.dataset + "|" + d.clip_id + "|" + d.start.map(fl) + "|" + d.end.map(fl)
ev_key = lambda d: d.dataset + "|" + d.clip_id + "|" + d.event_start.map(fl) + "|" + d.event_end.map(fl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="segments_v2")
    ap.add_argument("--out", default="segments_v2_2")
    a = ap.parse_args()
    B = f"{ROOT}/benchmark"
    rd = lambda f: pd.read_csv(f"{B}/{f}", dtype=str, keep_default_na=False)
    d = rd(f"{a.src}.csv")
    kept = d.kept == "True"

    m = seg_key(d).isin(set(seg_key(rd("bdd_motion_dropped.csv"))))
    d.loc[m & kept, ["kept", "reason"]] = ["False", "image motion: standing still"]
    print("image motion:", int((m & kept).sum()))

    ek = ev_key(d)
    for f, why, stretch in (("trainval_corrections.csv", "expert review: unsure / no maneuver", False),
                            ("audit_corrections.csv", "audit: lat unsure", True)):
        c = rd(f)
        for k, lat in zip(seg_key(c.rename(columns={"start": "start", "end": "end"})), c.lat):
            rows = (ek == k) & (d.kept == "True")
            if lat == "IGNORE":
                d.loc[rows, ["kept", "reason"]] = ["False", why]
            else:
                d.loc[rows, "lat"] = lat
                if stretch and lat in ("STRAIGHT", "CURVE"):
                    d.loc[rows, "kind"] = "stretch"
        print(f, len(c), "corrections")

    rev = (d.long == "REVERSE") & (d.kept == "True")
    d.loc[rev, ["kept", "reason"]] = ["False", "reverse"]
    print("reverse:", int(rev.sum()))
    d.to_csv(f"{B}/{a.out}.csv", index=False)
    k = d[(d.kept == "True") & d.part.isin(["train", "val", "test"])]
    print(f"-> {B}/{a.out}.csv: {len(k)} segments", k.part.value_counts().to_dict())


if __name__ == "__main__":
    main()
