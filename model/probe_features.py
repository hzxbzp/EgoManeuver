"""A2 / X3: linear probes on frozen features. Which readout keeps road-relative state and ego kinematics?

Targets per 5 Hz step (from labels.npz via train_m1.load_labels; road state only on map datasets):
lane offset in lane-width units d/W, heading deviation from the lane, distance to the road edge, in-intersection
(AUC), speed, yaw rate. Linear probe on LayerNorm'd, flattened tokens; trained on train-split clips, evaluated
on val/test clips (R^2 / AUC).

Usage: python model/probe_features.py --feats vjepa2x_rinterp --tokens pool9,grid33,mid33,grid33+mid33 [--gpu 0]
"""

import argparse
import importlib.util
from egokin.scripts import script_path  # noqa: E402
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

spec = importlib.util.spec_from_file_location("m1", script_path("train_m1.py"))
m1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m1)

MAP_DS = ["nuplan", "av2", "nuscenes"]
TARGETS = ["d_over_w", "heading_dev", "edge_dist", "speed", "yaw_rate"]  # regression; plus in_intersection (AUC)


def collect(idx, feats, tokens, per_clip, rng, road):
    X, Y, I = [], [], []
    for r in idx.itertuples():
        f, ya, yb, k = m1.load_clip(r, feats, tokens)
        T = len(f)
        if T == 0:
            continue
        if road:
            ok = ~np.isnan(k[:, 4]) & ~np.isnan(k[:, 5]) & (k[:, 5] > 0.2)
        else:
            ok = ~np.isnan(k[:, 0]) & ~np.isnan(k[:, 2])
        cand = np.flatnonzero(ok)
        if not len(cand):
            continue
        sel = np.sort(rng.choice(cand, min(per_clip, len(cand)), replace=False))
        x = np.stack([np.asarray(f[t:t + 1])[0] for t in sel]).astype(np.float16)  # (n, tok, D)
        X.append(x)
        kk = k[sel]
        y = np.stack([kk[:, 4] / kk[:, 5], kk[:, 7], kk[:, 6], kk[:, 0], kk[:, 2]], 1) if road else \
            np.stack([np.full(len(sel), np.nan), np.full(len(sel), np.nan), np.full(len(sel), np.nan), kk[:, 0], kk[:, 2]], 1)
        Y.append(y.astype(np.float32))
        I.append(kk[:, 8] if road else np.full(len(sel), np.nan))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(I)


def fit_probe(Xtr, Ytr, Itr, dev, steps=3000, bs=1024, lr=1e-3):
    n, tok, D = Xtr.shape
    mu, sd = np.nanmean(Ytr, 0), np.nanstd(Ytr, 0) + 1e-6
    lin = torch.nn.Linear(tok * D, Ytr.shape[1] + 1).to(dev)
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=1e-2)
    Yn = torch.from_numpy((Ytr - mu) / sd)
    Ib = torch.from_numpy(Itr.astype(np.float32))
    g = np.random.default_rng(0)
    for _ in range(steps):
        j = g.integers(0, n, bs)
        x = F.layer_norm(torch.from_numpy(Xtr[j]).to(dev).float(), (D,)).flatten(1)
        out = lin(x)
        y, ib = Yn[j].to(dev), Ib[j].to(dev)
        m = ~torch.isnan(y)
        loss = F.mse_loss(out[:, :-1][m], y[m]) if m.any() else out.sum() * 0
        mi = ~torch.isnan(ib)
        if mi.any():
            loss = loss + F.binary_cross_entropy_with_logits(out[:, -1][mi], ib[mi])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return lin, mu, sd


@torch.no_grad()
def evaluate(lin, mu, sd, X, Y, I, dev):
    D = X.shape[2]
    P = []
    for s in range(0, len(X), 4096):
        x = F.layer_norm(torch.from_numpy(X[s:s + 4096]).to(dev).float(), (D,)).flatten(1)
        P.append(lin(x).cpu().numpy())
    P = np.concatenate(P)
    out = {}
    for j, name in enumerate(TARGETS):
        y, p = Y[:, j], P[:, j] * sd[j] + mu[j]
        m = ~np.isnan(y)
        if m.sum() > 50:
            out[name] = float(1 - np.mean((y[m] - p[m]) ** 2) / np.var(y[m]))
    mi = ~np.isnan(I)
    if mi.sum() > 50 and 0 < I[mi].mean() < 1:
        out["in_intersection_auc"] = float(roc_auc_score(I[mi] > 0.5, P[mi, -1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--tokens", default="all")
    ap.add_argument("--per-clip", type=int, default=40)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", default="probe")
    a = ap.parse_args()
    dev = torch.device(f"cuda:{a.gpu}") if a.gpu >= 0 else torch.device("cpu")
    res = {}
    for tokens in a.tokens.split(","):
        rng = np.random.default_rng(0)
        idx = m1.build_index()
        idx = idx[idx.apply(lambda r: m1.has_features(a.feats, r.dataset, r.clip_id), axis=1)]
        out = {}
        for kind, dsets, per_clip, max_clips in (("road", MAP_DS, a.per_clip, 100000),
                                                 ("kin", ["hdd", "bdd", "pai", "nuplan", "av2", "nuscenes"], 10, 4000)):
            sub = idx[idx.dataset.isin(dsets)]
            tr, te = sub[sub.split == "train"], sub[sub.split.isin(["val", "test"])]
            if kind == "kin":  # PAI is train-only: hold out 10% of PAI clips as test
                pai = tr[tr.dataset == "pai"].sample(frac=0.1, random_state=0)
                tr, te = tr.drop(pai.index), pd.concat([te, pai])
            tr = tr.sample(min(len(tr), max_clips), random_state=0)
            te = te.sample(min(len(te), max_clips // 2), random_state=0)
            Xtr, Ytr, Itr = collect(tr, a.feats, tokens, per_clip, rng, kind == "road")
            Xte, Yte, Ite = collect(te, a.feats, tokens, per_clip, rng, kind == "road")
            lin, mu, sd = fit_probe(Xtr, Ytr, Itr, dev)
            r = evaluate(lin, mu, sd, Xte, Yte, Ite, dev)
            keep = ("speed", "yaw_rate") if kind == "kin" else ("d_over_w", "heading_dev", "edge_dist", "in_intersection_auc")
            out.update({k: v for k, v in r.items() if k in keep})
            print(f"[{tokens}] {kind}: train {len(Xtr)} test {len(Xte)} steps ->",
                  {k: round(v, 3) for k, v in r.items() if k in keep}, flush=True)
        res[tokens] = out
    os.makedirs(f"{m1.ROOT}/runs/probes", exist_ok=True)
    json.dump(dict(feats=a.feats, results=res), open(f"{m1.ROOT}/runs/probes/{a.tag}_{a.feats}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
