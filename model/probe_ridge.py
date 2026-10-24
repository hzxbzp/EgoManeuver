"""Stricter linear probe than model/probe_features.py: closed-form ridge regression with the regularization chosen
on held-out training clips (the gradient probe was not regularized and overfit at 33-66 tokens).

Same targets and sampling as probe_features.py: road state on the map datasets (lane offset / half lane width,
heading relative to the lane, distance to the road edge, in-intersection), kinematics on six datasets (speed, yaw
rate). Features: LayerNorm per token, flattened, centered. Training clips are split 80/20 by clip into fit /
validation to pick the ridge strength per target; the final probe is refit on all training clips and scored on
the val/test clips (R^2; AUC for in-intersection). Solved in float64 on the CPU (the Gram matrix of 33 tokens is
9 GB, more than the GPUs have free next to the end-to-end runs), one Cholesky factorization per ridge strength.

Usage: python model/probe_ridge.py --feats vjepa2x_rinterp --tokens pool9,grid33,mid33 [--gpu 0]
"""

import argparse
import importlib.util
from egokin.scripts import script_path  # noqa: E402
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

spec = importlib.util.spec_from_file_location("pf", script_path("probe_features.py"))
pf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pf)
m1 = pf.m1

ALPHAS = [10.0 ** e for e in range(-5, 6)]  # multiplied by the number of fit rows


def ln_flat(X, dev):
    """(n, tok, D) float16 -> (n, tok*D) float64: LayerNorm per token, flattened."""
    x = torch.from_numpy(np.asarray(X, np.float32)).to(dev)
    return torch.nn.functional.layer_norm(x, (x.shape[-1],)).flatten(1).double()


def stats(X, Y, dev, chunk=2048):
    """Centered Gram matrix X'X and X'Y of the probe features (accumulated in chunks), feature and target means."""
    d = X.shape[1] * X.shape[2]
    G = torch.zeros(d, d, dtype=torch.float64, device=dev)
    b = torch.zeros(d, Y.shape[1], dtype=torch.float64, device=dev)
    s = torch.zeros(d, dtype=torch.float64, device=dev)
    for i in range(0, len(X), chunk):
        x = ln_flat(X[i:i + chunk], dev)
        G += x.T @ x
        b += x.T @ torch.from_numpy(Y[i:i + chunk]).to(dev, torch.float64)
        s += x.sum(0)
    n = len(X)
    mu = s / n
    ybar = torch.from_numpy(Y.mean(0)).to(dev, torch.float64)
    G -= n * torch.outer(mu, mu)
    b -= n * torch.outer(mu, ybar)
    return G, b, mu, ybar


def apply(X, W, mu, ybar, dev, chunk=4096):
    return np.concatenate([((ln_flat(X[i:i + chunk], dev) - mu) @ W + ybar).cpu().numpy() for i in range(0, len(X), chunk)])


def solve(G, b, alpha):
    """Ridge weights for every column of b at one strength: (G + alpha I)^-1 b by Cholesky."""
    A = G.clone()
    A.diagonal().add_(alpha)
    L = torch.linalg.cholesky(A)
    del A
    return torch.cholesky_solve(b, L)


def scores(pred, Y, I, names, pred_i=None):
    out = {}
    for j, n in enumerate(names):
        m = ~np.isnan(Y[:, j])
        out[n] = float(1 - np.mean((Y[m, j] - pred[m, j]) ** 2) / np.var(Y[m, j]))
    if pred_i is not None:
        m = ~np.isnan(I)
        if m.sum() > 50 and 0 < I[m].mean() < 1:
            out["in_intersection_auc"] = float(roc_auc_score(I[m] > 0.5, pred_i[m]))
    return out


def run(kind, tr, te, feats, tokens, dev, rng):
    road = kind == "road"
    per_clip = 40 if road else 10
    fit = tr.sample(frac=0.8, random_state=0)
    val = tr.drop(fit.index)
    data = {k: pf.collect(d, feats, tokens, per_clip, rng, road) for k, d in (("fit", fit), ("val", val), ("test", te))}
    cols = [0, 1, 2] if road else [3, 4]  # d/W, heading, edge | speed, yaw rate
    names = [pf.TARGETS[c] for c in cols]
    Y = {k: v[1][:, cols] for k, v in data.items()}
    I = {k: v[2] for k, v in data.items()}
    keep = {k: ~np.isnan(Y[k]).any(1) for k in Y}  # rows with every target of this kind
    if road:
        for k in keep:
            keep[k] &= ~np.isnan(I[k])
    # standardize targets on the fit rows; in-intersection is one more (0/1) regression target
    T = {k: np.concatenate([Y[k], I[k][:, None]], 1)[keep[k]] if road else Y[k][keep[k]] for k in Y}
    mu, sd = T["fit"].mean(0), T["fit"].std(0) + 1e-9
    Xf, Xv, Xt = (data[k][0][keep[k]] for k in ("fit", "val", "test"))
    G, b, fmu, ybar = stats(Xf, (T["fit"] - mu) / sd, dev)
    n_fit = len(Xf)
    r2 = []  # (alpha, target) validation R^2; ridge strength chosen per target on the validation clips
    for al in ALPHAS:
        p = apply(Xv, solve(G, b, al * n_fit), fmu, ybar, dev) * sd + mu
        r2.append(1 - ((T["val"] - p) ** 2).mean(0) / T["val"].var(0))
    r2 = np.stack(r2)
    best = [ALPHAS[int(np.argmax(r2[:, j]))] for j in range(r2.shape[1])]
    del G, b
    torch.cuda.empty_cache()
    # refit on all training clips with the chosen strengths, score on the test clips
    Xtr, Ttr = np.concatenate([Xf, Xv]), np.concatenate([T["fit"], T["val"]])
    G, b, fmu, ybar = stats(Xtr, (Ttr - mu) / sd, dev)
    W = torch.cat([solve(G, b[:, j:j + 1], best[j] * len(Xtr)) for j in range(Ttr.shape[1])], 1)  # few distinct strengths
    P = apply(Xt, W, fmu, ybar, dev) * sd + mu
    del G, b, W
    out = scores(P[:, :len(names)], T["test"][:, :len(names)], I["test"][keep["test"]], names,
                 P[:, -1] if road else None)
    out["alpha"] = {n: best[j] for j, n in enumerate(names + (["in_intersection"] if road else []))}
    out["rows"] = {"train": int(len(Xtr)), "test": int(keep["test"].sum())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--tokens", default="pool9,grid33,mid33")
    ap.add_argument("--gpu", type=int, default=-1, help="-1: CPU (default; see the module docstring)")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--tag", default="ridge")
    ap.add_argument("--kin-datasets", default="hdd,bdd,pai,nuplan,av2,nuscenes",
                    help="sources of the kinematic probe rows (nuplan,av2,nuscenes: the clips of the road probe)")
    ap.add_argument("--kinds", default="road,kin", help="probe families to run")
    ap.add_argument("--pai-test", action="store_true",
                    help="hold out 10%% of the PAI clips as test rows (earlier runs; PAI must stay training-only for reported results)")
    a = ap.parse_args()
    dev = torch.device(f"cuda:{a.gpu}") if a.gpu >= 0 else torch.device("cpu")
    torch.set_num_threads(a.threads)
    idx = m1.build_index()
    idx = idx[idx.apply(lambda r: m1.has_features(a.feats, r.dataset, r.clip_id), axis=1)]
    res = {}
    for tokens in a.tokens.split(","):
        res[tokens] = {}
        for kind, dsets, max_clips in (("road", pf.MAP_DS, 100000), ("kin", a.kin_datasets.split(","), 4000)):
            if kind not in a.kinds.split(","):
                continue
            rng = np.random.default_rng(0)
            sub = idx[idx.dataset.isin(dsets)]
            tr, te = sub[sub.split == "train"], sub[sub.split.isin(["val", "test"])]
            if kind == "kin" and a.pai_test:  # earlier runs: 10% of PAI clips as test (no PAI results may be reported)
                pai = tr[tr.dataset == "pai"].sample(frac=0.1, random_state=0)
                tr, te = tr.drop(pai.index), pd.concat([te, pai])
            tr = tr.sample(min(len(tr), max_clips), random_state=0)
            te = te.sample(min(len(te), max_clips // 2), random_state=0)
            r = run(kind, tr, te, a.feats, tokens, dev, rng)
            res[tokens][kind] = r
            print(f"[{tokens}] {kind}:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
            torch.cuda.empty_cache()
    os.makedirs(f"{m1.ROOT}/runs/probes", exist_ok=True)
    json.dump(dict(feats=a.feats, results=res), open(f"{m1.ROOT}/runs/probes/{a.tag}_{a.feats}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
