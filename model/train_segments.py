"""Segment-level benchmark: one lateral + one longitudinal label per event segment
($VEGO_DATA/benchmark/segments_v<k>.csv, --segments; data/build_segments.py). Trains a head on cached step tokens and reports
per-class AP / mAP, top-1 accuracy and macro-F1 on the test pool (val + test splits; HDD test sessions only).

Partial labels: LANE_KEEP (straight or curve, unknown which; ROAD-Waymo) and MOVING (moving, speed change unknown;
ROAD-Waymo) are set-valued targets: the loss is -log of the summed probability of the allowed classes, and in the
metrics they count as negatives for every class outside the set and are left out for the classes inside it.

Head: per step, attention readout over the step's tokens (+ slot embeddings) -> bidirectional temporal transformer
over the segment's steps (padding masked) -> attention pooling over time -> lateral and longitudinal classifiers.

Usage: python model/train_segments.py --feats ft_finalfull_s2 --tag seg_ft [--tokens all] [--seed 0] [--gpu 0]
"""

import argparse
import importlib.util
from egokin.scripts import script_path  # noqa: E402
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score

spec = importlib.util.spec_from_file_location("m1", script_path("train_m1.py"))
m1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m1)
ROOT = m1.ROOT

LAT = ["STRAIGHT", "CURVE", "TURN_LEFT", "TURN_RIGHT", "U_TURN", "LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT",
       "MERGE_LEFT", "MERGE_RIGHT", "BRANCH_LEFT", "BRANCH_RIGHT", "PULL_OVER", "PULL_OUT", "NUDGE_LEFT", "NUDGE_RIGHT",
       "BORROW_LEFT", "NONE"]
# slow-down-then-speed-up (DIP) is no longer a class: split in lane keeping, MOVING inside maneuvers
LONG = ["STATIONARY", "START", "STOP", "ACCELERATE", "DECELERATE", "CRUISE", "REVERSE"]
PARTIAL = {"LANE_KEEP": ["STRAIGHT", "CURVE"], "MOVING": ["ACCELERATE", "DECELERATE", "CRUISE"]}
# Lateral classes each source can label (for the --federated ablation): HDD has no pull-out label; BDD-X captions have
# no merge / branch / obstacle classes; the map rules (AV2, nuScenes, nuPlan) no obstacle classes after the label audit;
# ROAD-Waymo only turns and lane keeping; PAI poses only turns, U-turns and lane keeping. NONE (stationary) everywhere.
_ALL = set(LAT)
SOURCE_LAT = {
    "hdd": _ALL - {"PULL_OUT"},
    "bdd": {"STRAIGHT", "CURVE", "TURN_LEFT", "TURN_RIGHT", "U_TURN", "LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT",
            "PULL_OVER", "PULL_OUT", "NONE"},
    "av2": _ALL - {"NUDGE_LEFT", "NUDGE_RIGHT", "BORROW_LEFT"},
    "road_waymo": {"STRAIGHT", "CURVE", "TURN_LEFT", "TURN_RIGHT", "NONE"},
    "pai": {"STRAIGHT", "CURVE", "TURN_LEFT", "TURN_RIGHT", "U_TURN", "NONE"},
}
SOURCE_LAT["nuscenes"] = SOURCE_LAT["nuplan"] = SOURCE_LAT["av2"]
HZ = 5  # feature steps per second
MIN_TEST = 10  # classes with fewer test segments are reported but left out of the main mAP


def target_sets(labels, classes):
    """(N, C) bool: allowed classes per sample (one-hot, or a set for the partial labels)."""
    out = np.zeros((len(labels), len(classes)), bool)
    for i, l in enumerate(labels):
        for c in PARTIAL.get(l, [l]):
            out[i, classes.index(c)] = True
    return out


class SegHead(nn.Module):
    def __init__(self, n_tok, d_in=1024, d=384, layers=4, max_steps=256):
        super().__init__()
        self.norm, self.proj = nn.LayerNorm(d_in), nn.Linear(d_in, d)
        self.slot = nn.Parameter(torch.zeros(1, n_tok, d))
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, 6, batch_first=True)
        self.pos = nn.Parameter(torch.zeros(1, max_steps, d))
        self.temporal = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 6, 4 * d, 0.1, batch_first=True, norm_first=True), layers)
        self.tq = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.tpool = nn.MultiheadAttention(d, 6, batch_first=True)
        self.lat, self.long = nn.Linear(d, len(LAT)), nn.Linear(d, len(LONG))

    def forward(self, x, valid):  # x (B, T, N, D), valid (B, T)
        B, T, N, _ = x.shape
        t = self.proj(self.norm(x)).view(B * T, N, -1) + self.slot
        h, _ = self.pool(self.q.expand(B * T, -1, -1), t, t)
        h = self.temporal(h.view(B, T, -1) + self.pos[:, :T], src_key_padding_mask=~valid)
        g, _ = self.tpool(self.tq.expand(B, -1, -1), h, h, key_padding_mask=~valid)
        return self.lat(g[:, 0]), self.long(g[:, 0])


def set_loss(logits, allowed):
    lp = logits.float().log_softmax(-1)
    return -(torch.logsumexp(lp.masked_fill(~allowed, -1e4), -1)).mean()


def load_segments(feats, name="segments_v1"):
    d = pd.read_csv(f"{ROOT}/benchmark/{name}.csv", dtype={"clip_id": str, "part": str}, keep_default_na=False)
    d = d[d.kept.astype(str) == "True"].copy()
    if "part" not in d:  # v1: no validation split
        d = d[~((d.dataset == "hdd") & (d.split != "test") & (d.split != "train"))]
        d["part"] = np.where(d.split == "train", "train", "test")
    d = d[d.part.isin(["train", "val", "test"])]
    for c in ("start", "end", "event_start", "event_end"):
        d[c] = d[c].astype(float)
    if feats:  # heads on cached features: segments of clips with features only
        have = {(r.dataset, r.clip_id) for r in d[["dataset", "clip_id"]].drop_duplicates().itertuples()
                if m1.has_features(feats, r.dataset, r.clip_id)}
        d = d[[(a, b) in have for a, b in zip(d.dataset, d.clip_id)]]
    return d.reset_index(drop=True)


class Store:
    """Lazy per-clip token readers (train_m1.TokSel), shared by all segments of a clip."""

    def __init__(self, feats, tokens):
        self.feats, self.idx, self.cache = feats, None if tokens == "all" else m1.RICH_TOKENS[tokens], {}

    def get(self, ds, cid, start, end, max_steps):
        k = (ds, cid)
        if k not in self.cache:
            self.cache[k] = m1.feature_reader(self.feats, ds, cid, self.idx)
        f = self.cache[k]
        s, e = int(np.floor(start * HZ)), int(np.ceil(end * HZ))
        s = min(s, len(f) - 1)
        e = min(e, len(f), s + max_steps)
        return np.asarray(f[s:max(e, s + 1)], np.float32)


def batchify(xs, dev):
    T = max(len(x) for x in xs)
    X = np.zeros((len(xs), T) + xs[0].shape[1:], np.float32)
    V = np.zeros((len(xs), T), bool)
    for i, x in enumerate(xs):
        X[i, :len(x)], V[i, :len(x)] = x, True
    return torch.from_numpy(X).to(dev), torch.from_numpy(V).to(dev)


def ap_table(P, allowed, classes, exact):
    """Per-class AP over segments; samples whose allowed set contains the class but is not exactly it are left out."""
    out, n = {}, {}
    for j, c in enumerate(classes):
        pos = allowed[:, j] & exact
        valid = exact | ~allowed[:, j]
        n[c] = int(pos.sum())
        if pos.sum() and (valid & ~pos).sum():
            out[c] = float(average_precision_score(pos[valid], P[valid, j]))
    return out, n


def evaluate(head, d, store, dev, max_steps):
    head.eval()
    PL, PO = [], []
    with torch.no_grad():
        for i in range(0, len(d), 64):
            sub = d.iloc[i:i + 64]
            x, v = batchify([store.get(r.dataset, r.clip_id, r.start, r.end, max_steps) for r in sub.itertuples()], dev)
            la, lo = head(x, v)
            PL.append(la.float().softmax(-1).cpu().numpy()); PO.append(lo.float().softmax(-1).cpu().numpy())
    head.train()
    return np.concatenate(PL), np.concatenate(PO)


def seg_keys(d):
    """Segment identity across benchmark rebuilds (re-scoring after label corrections: model/rescore_segments.py)."""
    return (d.dataset + "|" + d.clip_id + "|" + d.start.map("{:.1f}".format) + "|" + d.end.map("{:.1f}".format)).to_numpy()


def metrics(d, PL, PO):
    res = {}
    for name, labels, P, classes in (("lateral", d.lat.tolist(), PL, LAT), ("longitudinal", d.long.tolist(), PO, LONG)):
        allowed = target_sets(labels, classes)
        exact = allowed.sum(1) == 1
        ap, n = ap_table(P, allowed, classes, exact)
        main = {c: v for c, v in ap.items() if n[c] >= MIN_TEST}
        y = allowed[exact].argmax(1)
        pred = P[exact].argmax(1)
        res[name] = {"mAP": float(np.mean(list(main.values()))), "mAP_all": float(np.mean(list(ap.values()))),
                     "acc": float((pred == y).mean()), "macro_f1": float(f1_score(y, pred, average="macro")),
                     "AP": ap, "n_test": n}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--tokens", default="all")
    ap.add_argument("--segments", default="segments_v1", help="benchmark version ($VEGO_DATA/benchmark/<name>.csv)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=128, help="segment length cap in 0.2 s steps")
    ap.add_argument("--balance", type=float, default=0.5, help="sample segments ~ (lateral class frequency)^-balance")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-test", type=int, default=0, help="smoke test: first N test segments per dataset")
    ap.add_argument("--predict", default="", help="CSV of extra windows (dataset, clip_id, start, end in s) to score")
    ap.add_argument("--federated", action="store_true",
                    help="ablation: lateral softmax only over the classes the segment's source can label (SOURCE_LAT)")
    ap.add_argument("--drop-partial", action="store_true", help="ablation: train without partially labelled segments")
    a = ap.parse_args()
    dev = torch.device(f"cuda:{a.gpu}") if a.gpu >= 0 else torch.device("cpu")
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    d = load_segments(a.feats, a.segments)
    tr, te = d[d.part == "train"].reset_index(drop=True), d[d.part == "test"].reset_index(drop=True)
    if a.drop_partial:
        tr = tr[~tr.lat.isin(list(PARTIAL)) & ~tr.long.isin(list(PARTIAL))].reset_index(drop=True)
    va = d[d.part == "val"].reset_index(drop=True)
    if a.max_test:
        te = te.groupby("dataset").head(a.max_test).reset_index(drop=True)
        va = va.groupby("dataset").head(a.max_test).reset_index(drop=True)
    print(f"{a.feats}: train {len(tr)} segments ({(tr.end - tr.start).sum() / 3600:.1f} h), val {len(va)}, test {len(te)}", flush=True)
    store = Store(a.feats, a.tokens)
    n_tok = store.get(tr.dataset[0], tr.clip_id[0], tr.start[0], tr.end[0], 4).shape[1]
    head = SegHead(n_tok, max_steps=a.max_steps).to(dev)
    freq = tr.lat.map(tr.lat.value_counts())
    w = (freq ** -a.balance).to_numpy(dtype=np.float64, copy=True); w /= w.sum()
    al_tr = torch.from_numpy(target_sets(tr.lat.tolist(), LAT)).to(dev)
    ao_tr = torch.from_numpy(target_sets(tr.long.tolist(), LONG)).to(dev)
    src_ok = torch.from_numpy(np.array([[c in SOURCE_LAT[ds] for c in LAT] for ds in tr.dataset])).to(dev) | al_tr
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=a.steps, pct_start=0.05)
    for step in range(a.steps):
        ix = rng.choice(len(tr), a.bs, p=w)
        x, v = batchify([store.get(tr.dataset[i], tr.clip_id[i], tr.start[i], tr.end[i], a.max_steps) for i in ix], dev)
        la, lo = head(x, v)
        if a.federated:  # classes the source cannot label give no negative evidence
            la = la.masked_fill(~src_ok[ix], -1e4)
        loss = set_loss(la, al_tr[ix]) + set_loss(lo, ao_tr[ix])
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sched.step()
        if (step + 1) % 1000 == 0:
            print(f"step {step + 1} loss {loss.item():.3f}", flush=True)
    PL, PO = evaluate(head, te, store, dev, a.max_steps)
    res = {"tag": a.tag, "args": vars(a), "long_classes": LONG, "test": metrics(te, PL, PO)}
    if len(va):  # model selection uses the validation split only
        res["val"] = metrics(va, *evaluate(head, va, store, dev, a.max_steps))
    for ds in sorted(te.dataset.unique()):
        m = (te.dataset == ds).to_numpy()
        res[f"{ds}_test"] = metrics(te[m], PL[m], PO[m])
    t = res["test"]
    print(f"[{a.tag} s{a.seed}] lateral mAP {t['lateral']['mAP'] * 100:.1f} acc {t['lateral']['acc'] * 100:.1f} F1 {t['lateral']['macro_f1'] * 100:.1f} | "
          f"longitudinal mAP {t['longitudinal']['mAP'] * 100:.1f} acc {t['longitudinal']['acc'] * 100:.1f} F1 {t['longitudinal']['macro_f1'] * 100:.1f}", flush=True)
    if "val" in res:
        print(f"   val: lateral mAP {res['val']['lateral']['mAP'] * 100:.1f} | longitudinal mAP {res['val']['longitudinal']['mAP'] * 100:.1f}", flush=True)
    print("   lateral AP:", {k: round(v * 100, 1) for k, v in t["lateral"]["AP"].items()}, flush=True)
    out = f"{ROOT}/runs/segments"
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(f"{out}/{a.tag}_s{a.seed}.json", "w"), indent=1)
    np.savez_compressed(f"{out}/{a.tag}_s{a.seed}_pred.npz", lat=PL.astype(np.float16), long=PO.astype(np.float16),
                        key=seg_keys(te), segments=a.segments, long_classes=np.array(LONG))
    torch.save(head.state_dict(), f"{out}/{a.tag}_s{a.seed}_head.pt")
    if a.predict:  # e.g. the PAI pose-only candidates (candidate segments)
        w = pd.read_csv(a.predict, dtype={"clip_id": str})
        w = w[[m1.has_features(a.feats, x, c) for x, c in zip(w.dataset, w.clip_id)]]
        WL, WO = evaluate(head, w, store, dev, a.max_steps)
        name = os.path.splitext(os.path.basename(a.predict))[0]
        np.savez_compressed(f"{out}/{a.tag}_s{a.seed}_{name}.npz", lat=WL, long=WO, row=w.index.to_numpy(),
                            key=(w.cand_id + "/" + w.part).to_numpy() if "cand_id" in w else w.clip_id.to_numpy())
        print(f"scored {len(w)} windows of {a.predict}", flush=True)


if __name__ == "__main__":
    main()
