"""Public benchmark: ROAD-Waymo AV-action recognition on the val split, frame-level mAP over the six
classes of the ROAD-Waymo paper (AV-Mov, AV-Stop, AV-TurRht, AV-TurLft, AV-MovRht, AV-MovLft; SlowFast 52.72 val).
All val videos are used (no night / work-zone isolation: public benchmark, as for the HDD test).

Two protocols for one end-to-end checkpoint of train_e2e_offline.py:
  map:  our model's class probabilities mapped to the six classes, no training on ROAD-Waymo labels;
  head: frozen encoder, a small head (attention readout + 2-layer temporal transformer) trained on the ROAD-Waymo
        train split with its own six multi-label classes, scored on val (same training data as the SlowFast baseline).

Usage: python model/bench_road_waymo.py --run final_s0 [--gpu 0]   (needs runs/e2e_off/<run>/ckpt.pt)
"""

import argparse
import importlib.util
from egokin.scripts import script_path  # noqa: E402
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score


def _load(name):
    spec = importlib.util.spec_from_file_location(name, script_path(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e2o = _load("train_e2e_offline")
m1, hb = e2o.m1, e2o.hb
from egokin.data.index import ROOT, build_index  # noqa: E402
from egokin.models.backbones import Backbone  # noqa: E402
from egokin.taxonomy import Lat, Long  # noqa: E402

CLASSES = ["AV-Mov", "AV-Stop", "AV-TurRht", "AV-TurLft", "AV-MovRht", "AV-MovLft"]
RAW = f"{ROOT}/raw/waymo/road_waymo/road_waymo_trainval_v1.0.json"


def frame_labels():
    """{video: (n_frames, 6) bool} and split per video, from the official json (frames 1..numf, 10 Hz)."""
    d = json.load(open(RAW))
    names = d["all_av_action_labels"]
    idx = [names.index(c) for c in CLASSES]
    out, split = {}, {}
    for v, e in d["db"].items():
        fr = e["frames"]
        keys = sorted(fr, key=int)
        y = np.zeros((len(keys), len(CLASSES)), bool)
        ann = np.zeros(len(keys), bool)
        for i, k in enumerate(keys):
            ann[i] = bool(fr[k].get("annotated", 0))
            ids = set(fr[k].get("av_action_ids", []))
            y[i] = [j in ids for j in idx]
        out[v] = (y, ann)
        split[v] = "val" if "val" in e["split_ids"] else "train"
    return out, split


def mapped_scores(pa, pb):
    """(T, 7) longitudinal + (T, 18) lateral probabilities -> (T, 6) scores of the ROAD-Waymo classes."""
    p = lambda arr, cs: arr[:, [int(c) for c in cs]].sum(1)
    stop = p(pa, [Long.STATIONARY])
    return np.stack([1 - stop - p(pa, [Long.REVERSE]), stop,
                     p(pb, [Lat.TURN_RIGHT]), p(pb, [Lat.TURN_LEFT, Lat.U_TURN]),
                     p(pb, [Lat.LANE_CHANGE_RIGHT, Lat.NUDGE_RIGHT, Lat.BORROW_RIGHT, Lat.MERGE_RIGHT, Lat.BRANCH_RIGHT]),
                     p(pb, [Lat.LANE_CHANGE_LEFT, Lat.NUDGE_LEFT, Lat.BORROW_LEFT, Lat.MERGE_LEFT, Lat.BRANCH_LEFT])], 1)


def frame_map(scores_by_video, labels, videos):
    """Frame-level AP per class over all annotated frames of the given videos (step t covers frames 2t, 2t+1)."""
    S, Y = [], []
    for v in videos:
        y, ann = labels[v]
        s = scores_by_video[v]
        n = min(len(y), 2 * len(s))
        f = np.arange(n)
        S.append(s[f // 2][ann[:n]]); Y.append(y[:n][ann[:n]])
    S, Y = np.concatenate(S), np.concatenate(Y)
    ap = {c: float(average_precision_score(Y[:, j], S[:, j])) for j, c in enumerate(CLASSES) if Y[:, j].any()}
    return float(np.mean(list(ap.values()))), ap, int(len(Y))


class Head(nn.Module):
    """Same readout as the behaviour head (attention over the step tokens with slot embeddings), smaller temporal model."""

    def __init__(self, n_tok, d_in=1024, d=256):
        super().__init__()
        self.norm, self.proj = nn.LayerNorm(d_in), nn.Linear(d_in, d)
        self.slot = nn.Parameter(torch.zeros(1, n_tok, d))
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, 4, batch_first=True)
        self.pos = nn.Parameter(torch.zeros(1, 512, d))
        self.temporal = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, 0.1, batch_first=True, norm_first=True), 2)
        self.out = nn.Linear(d, len(CLASSES))

    def forward(self, x, valid=None):  # x (B, T, N, D)
        B, T, N, _ = x.shape
        t = self.proj(self.norm(x)).view(B * T, N, -1) + self.slot
        h, _ = self.pool(self.q.expand(B * T, -1, -1), t, t)
        h = self.temporal(h.view(B, T, -1) + self.pos[:, :T], src_key_padding_mask=None if valid is None else ~valid)
        return self.out(h)


def step_targets(y, ann, T):
    """(T, 6) multi-hot per 5 Hz step (union of its two frames) and a mask of steps with an annotated frame."""
    n = min(len(y), 2 * T)
    yt, mt = np.zeros((T, len(CLASSES)), np.float32), np.zeros(T, bool)
    for f in range(n):
        if ann[f]:
            yt[f // 2] = np.maximum(yt[f // 2], y[f]); mt[f // 2] = True
    return yt, mt


def train_head(feats, labels, train_v, dev, seed, steps=3000, L=128, bs=16):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = [(feats[v], *step_targets(*labels[v], len(feats[v]))) for v in train_v]
    head = Head(feats[train_v[0]].shape[1]).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps, pct_start=0.05)
    for _ in range(steps):
        xs, ys, ms, vs = [], [], [], []
        for i in rng.integers(0, len(data), bs):
            g, y, m = data[i]
            s = rng.integers(0, max(1, len(g) - L + 1)); n = min(L, len(g) - s); pad = L - n
            xs.append(np.pad(g[s:s + n], ((0, pad), (0, 0), (0, 0)))); ys.append(np.pad(y[s:s + n], ((0, pad), (0, 0))))
            ms.append(np.pad(m[s:s + n], (0, pad))); vs.append(np.arange(L) < n)
        x = torch.from_numpy(np.stack(xs)).to(dev).float()
        y, m, v = (torch.from_numpy(np.stack(a)).to(dev) for a in (ys, ms, vs))
        loss = F.binary_cross_entropy_with_logits(head(x, v)[m], y[m])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    return head


@torch.no_grad()
def head_scores(head, g, dev, L=128):
    head.eval()
    T = len(g)
    out = np.zeros((T, len(CLASSES)), np.float32)
    for s, k0, k1 in m1.windows(T, L):
        x = torch.from_numpy(np.asarray(g[s:s + L], np.float32)).unsqueeze(0).to(dev)
        out[k0:k1] = torch.sigmoid(head(x))[0, k0 - s:k1 - s].cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs/e2e_off/<run> with ckpt.pt and result.json (its args are reused)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-clips", type=int, default=0, help="smoke test: clips per split")
    ap.add_argument("--head-steps", type=int, default=3000)
    a = ap.parse_args()
    dev = torch.device(f"cuda:{a.gpu}")
    rdir = f"{ROOT}/runs/e2e_off/{a.run}"
    args = json.load(open(f"{rdir}/result.json"))["args"]
    e2o.TOK, e2o.WIN = m1.RICH_TOKENS[args["tokens"]], args["L"]
    bb = Backbone(args["backbone"], args["adapt"], args["first_trainable"], rope=args["rope"])
    enc = e2o.Encoder(bb, e2o.TOK).to(dev)
    head = hb.BenchModel(1024, len(e2o.TOK), True, True, causal=False, n_kin=9 if args["road"] else 4,
                         tok_pos=args["tok_pos"]).to(dev)
    ck = torch.load(f"{rdir}/ckpt.pt", map_location=dev)
    missing = bb.load_state_dict(ck["backbone"], strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys
    head.load_state_dict(ck["head"])
    enc.eval(); head.eval()
    labels, split = frame_labels()
    idx = build_index(["road_waymo"])  # every clip, no scope filter (public benchmark)
    idx = idx[idx.clip_id.isin(labels)]
    vids = {s: sorted(idx[idx.clip_id.map(split) == s].clip_id) for s in ("train", "val")}
    if a.max_clips:
        vids = {s: v[:a.max_clips] for s, v in vids.items()}
    out = f"{ROOT}/runs/bench_road_waymo/{a.run}"
    os.makedirs(out, exist_ok=True)
    feats, mapped = {}, {}
    dirs = dict(zip(idx.clip_id, idx.dir))
    for s in ("train", "val"):
        for v in vids[s]:
            T = len(labels[v][0]) // 2
            g = e2o.fresh_features(enc, os.path.join(dirs[v], "video.mp4"), T, dev)
            feats[v] = g.astype(np.float16)
            if s == "val":
                pa, pb, _, _ = e2o.predict(head, g, dev)
                mapped[v] = mapped_scores(pa, pb)
    res = {"run": a.run, "frames_val": None}
    m, ap, n = frame_map(mapped, labels, vids["val"])
    res["map"] = {"mAP": m, "AP": ap}
    res["frames_val"] = n
    print(f"[map] mAP {m * 100:.2f}", {k: round(v * 100, 1) for k, v in ap.items()}, f"({n} val frames)", flush=True)
    res["head"] = []
    for seed in (0, 1, 2):
        h = train_head(feats, labels, vids["train"], dev, seed, steps=a.head_steps)
        m, ap, _ = frame_map({v: head_scores(h, feats[v], dev) for v in vids["val"]}, labels, vids["val"])
        res["head"].append({"seed": seed, "mAP": m, "AP": ap})
        print(f"[head s{seed}] mAP {m * 100:.2f}", {k: round(v * 100, 1) for k, v in ap.items()}, flush=True)
    json.dump(res, open(f"{out}/result.json", "w"), indent=1)


if __name__ == "__main__":
    main()
