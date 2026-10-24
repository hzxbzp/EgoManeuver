"""Score a part of a benchmark version with saved segment heads (<tag>_s<seed>_head.pt of train_segments.py), e.g. the
validation split of segments_v2_2 for the ablation table: the runs evaluated validation on the labels of the version
they were trained with (segments_v2), before the train/val corrections. Writes <tag>_<part>_s<seed>_pred.npz,
which model/benchmark_eval.py reads as run <tag>_<part> (--part <part>).

Usage: python model/predict_segments.py --tags v2_ours_trainonly,v2_segonly --segments segments_v2_2 --part val [--gpu 0]
"""

import argparse
import glob
import importlib.util
from egokin.scripts import script_path  # noqa: E402
import json
import os

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("seg", script_path("train_segments.py"))
seg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--segments", default="segments_v2_2")
    ap.add_argument("--part", default="val")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device(f"cuda:{a.gpu}") if a.gpu >= 0 else torch.device("cpu")
    out = f"{seg.ROOT}/runs/segments"
    for tag in a.tags.split(","):
        for f in sorted(glob.glob(f"{out}/{tag}_s[0-9].json")):
            args = json.load(open(f))["args"]
            seed = args["seed"]
            dst = f"{out}/{tag}_{a.part}_s{seed}_pred.npz"
            if os.path.exists(dst):
                continue
            d = seg.load_segments(args["feats"], a.segments)
            d = d[d.part == a.part].reset_index(drop=True)
            store = seg.Store(args["feats"], args.get("tokens", "all"))
            n_tok = store.get(d.dataset[0], d.clip_id[0], d.start[0], d.end[0], 4).shape[1]
            head = seg.SegHead(n_tok, max_steps=args["max_steps"]).to(dev)
            head.load_state_dict(torch.load(f"{out}/{tag}_s{seed}_head.pt", map_location=dev))
            PL, PO = seg.evaluate(head, d, store, dev, args["max_steps"])
            np.savez_compressed(dst, lat=PL.astype(np.float16), long=PO.astype(np.float16), key=seg.seg_keys(d),
                                segments=a.segments, long_classes=np.array(seg.LONG))
            print(f"{tag} s{seed}: {len(d)} {a.part} segments -> {dst}", flush=True)


if __name__ == "__main__":
    main()
