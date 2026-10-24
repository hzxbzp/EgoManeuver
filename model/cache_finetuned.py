"""Cache step tokens of a fine-tuned end-to-end encoder (train_e2e_offline.py run with ckpt.pt) for every in-scope clip
(and the PAI clips with ambiguous lateral moves): features for the segment-level baseline and for pseudo-labelling.
Output: $VEGO_DATA/cache/feats/ft_<run>/<dataset>/<clip_id>.npy, float16 (T, n_tok, 1024), same layout as the cache.

--causal (HDD online protocol): step t uses only the 16 frames ending at its last frame (2t+1) and keeps the last
tubelet's tokens: no future frame enters a step's features (8x the compute). Output dir ft_causal_<run>.

Usage: python model/cache_finetuned.py --run finalfull_s2 --shard 0/4 [--gpu 0] [--causal --datasets hdd]
"""

import argparse
import json
import os

import numpy as np
import torch

from egokin.data.index import ROOT, build_index
from egokin.data.scope import filter_index
from egokin.models.backbones import Backbone

import importlib.util
from egokin.scripts import script_path  # noqa: E402

spec = importlib.util.spec_from_file_location("e2o", script_path("train_e2e_offline.py"))
e2o = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e2o)


@torch.no_grad()
def causal_features(enc, video, T, dev, batch=32):
    from egokin.models.vjepa_lora import prep

    cf = e2o._load("cache_features")
    out, hist = [], None
    for frames in cf.frame_chunks(video, 512):
        allf = frames if hist is None else np.concatenate([hist, frames])
        off = 0 if hist is None else len(hist)
        x = prep(torch.from_numpy(allf).to(dev))  # (N, 3, 224, 448)
        ends = [min(off + 2 * t + 1, len(allf) - 1) for t in range((len(frames) + 1) // 2)]
        for i in range(0, len(ends), batch):
            ix = torch.stack([torch.arange(e - 15, e + 1) for e in ends[i:i + batch]]).clamp(min=0).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                g, _ = enc(x[ix])  # (B, 8, n_tok, D): the last tubelet = frames e-1, e
            out.append(g[:, -1].float().cpu().numpy().astype(np.float16))
        hist = allf[-14:]
    G = np.concatenate(out)[:T]
    return np.pad(G, ((0, T - len(G)), (0, 0), (0, 0)), mode="edge") if len(G) < T else G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--datasets", default="pai,hdd,bdd,nuscenes,av2,road_waymo,nuplan")
    ap.add_argument("--suffix", default="", help="shard file name suffix (two nodes caching into one directory)")
    a = ap.parse_args()
    dev = torch.device(f"cuda:{a.gpu}")
    rdir = f"{ROOT}/runs/e2e_off/{a.run}"
    args = json.load(open(f"{rdir}/result.json"))["args"]
    e2o.TOK, e2o.WIN = e2o.m1.RICH_TOKENS[args["tokens"]], args["L"]
    bb = Backbone(args["backbone"], args["adapt"], args["first_trainable"], rope=args["rope"])
    ck = torch.load(f"{rdir}/ckpt.pt", map_location="cpu")
    bb.load_state_dict(ck["backbone"], strict=False)
    enc = e2o.Encoder(bb, e2o.TOK).to(dev).eval()
    idx = filter_index(build_index(a.datasets.split(",")))
    i, n = map(int, a.shard.split("/"))
    idx = idx.sort_values(["dataset", "clip_id"]).iloc[i::n]
    name = f"ft_{'causal_' if a.causal else ''}{a.run}"
    # packed output: one append-only .bin + .jsonl index per worker (m1.feature_reader); creating one file per clip
    # stalls when the scratch metadata server is slow
    pdir = f"{ROOT}/cache/feats/{name}/packed"
    os.makedirs(pdir, exist_ok=True)
    base = f"{pdir}/shard{i}of{n}{a.suffix}"
    done, end = set(), 0
    if os.path.exists(base + ".jsonl"):  # resume: keep the indexed clips, drop a partially written tail
        for line in open(base + ".jsonl"):
            e = json.loads(line)
            done.add((e["dataset"], e["clip_id"]))
            end = max(end, e["offset"] + e["T"] * e["n_tok"] * e["dim"] * 2)
    with open(base + ".bin", "ab") as fb:
        fb.truncate(end)
    fb, fj = open(base + ".bin", "ab"), open(base + ".jsonl", "a")
    for k, r in enumerate(idx.itertuples()):
        if (r.dataset, r.clip_id) in done or e2o.m1.has_features(name, r.dataset, r.clip_id):
            continue
        T = r.n_frames // 2
        video = os.path.join(r.dir, "video.mp4")
        g = causal_features(enc, video, T, dev) if a.causal else e2o.fresh_features(enc, video, T, dev)
        g = np.ascontiguousarray(g.astype(np.float16))
        off = fb.tell()
        fb.write(g.tobytes()); fb.flush()
        fj.write(json.dumps(dict(dataset=r.dataset, clip_id=r.clip_id, offset=off, T=int(g.shape[0]),
                                 n_tok=int(g.shape[1]), dim=int(g.shape[2]))) + "\n"); fj.flush()
        if (k + 1) % 200 == 0:
            print(f"{k + 1}/{len(idx)}", flush=True)
    fb.close(); fj.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
