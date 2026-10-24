"""M1 validation: small temporal model on cached frozen features (5 Hz).

Variants (flags): --feats resnet50|vjepa2, --aux-kin (kinematics regression), --bottleneck
(predicted kinematics fed back into the behavior heads), --logit-adjust (long-tail prior correction).
Reports per-class AP for axis A (longitudinal) and B (lateral) on the test pool, plus HDD-test and night slices.

Usage: python model/train_m1.py --feats vjepa2 --aux-kin --bottleneck --gpu 0 --tag v2_kin_bn
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from egokin.data.index import ROOT
from egokin.data.index import build_index as build_index_all
from egokin.data.scope import filter_index, frame_mask
from egokin.taxonomy import IGNORE, Lat, Long

KIN_SCALE = np.array([10.0, 2.0, 0.3, 0.05], np.float32)  # v, a, yaw_rate, curvature
NO_MANEUVER = -2  # lane keeping known, straight vs curve unknown
LANE_KEEP = [int(Lat.STRAIGHT), int(Lat.CURVE)]


def build_index(datasets=None):
    """Clip index restricted to the data scope (night / work-zone data isolated; VEGO_SCOPE=all for all)."""
    return filter_index(build_index_all(datasets))


RICH_TOKENS = {  # token layout of the vjepa2x features (model/cache_features.py), 74 tokens per step
    "pool9": list(range(0, 9)), "grid33": [0] + list(range(9, 41)), "mid33": list(range(41, 74)),
}
RICH_TOKENS["grid33+mid33"] = RICH_TOKENS["grid33"] + RICH_TOKENS["mid33"]


class TokSel:
    """Lazy view of a cached feature file (T, N, D) keeping only some tokens. A slice is read with one pread of its
    contiguous rows: page-faulting a token subset out of an mmap made the rich-feature heads I/O bound."""

    def __init__(self, path, idx=None, packed=None):
        """packed: (byte offset, T, n_tok, dim) of a clip inside a packed shard file (float16 rows), see feature_reader."""
        if packed is None:
            f = np.load(path, mmap_mode="r")
            assert f.flags.c_contiguous, path
            dtype, offset, shape = f.dtype, f.offset, f.shape
        else:
            offset, T, n_tok, dim = packed
            dtype, shape = np.dtype(np.float16), (T, n_tok, dim)
        self.path, self.idx, self.dtype, self.offset = path, idx, dtype, offset
        self.n_tok, self.dim = (shape[1], shape[2]) if len(shape) == 3 else (1, shape[1])
        self.row = self.n_tok * self.dim * dtype.itemsize
        self.shape = (shape[0], len(idx) if idx is not None else self.n_tok, self.dim)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, sl):
        start, stop, step = sl.indices(self.shape[0])
        n = max(0, stop - start)
        buf = bytearray(n * self.row)
        view, got, pos = memoryview(buf), 0, self.offset + start * self.row
        fd = os.open(self.path, os.O_RDONLY)
        try:
            while got < len(buf):  # one read returns at most ~2 GB on Linux (long HDD sessions are larger)
                k = os.preadv(fd, [view[got:]], pos + got)
                if k <= 0:
                    raise EOFError(f"{self.path}: {got} of {len(buf)} bytes")
                got += k
        finally:
            os.close(fd)
        x = np.frombuffer(buf, self.dtype).reshape(n, self.n_tok, self.dim)[::step]
        return x[:, self.idx] if self.idx is not None else x


_PACKED = {}
# feature caches are read from here; a job may point it at a copy on node-local NVMe (B200 /lscratch), since random
# reads from the shared scratch make head training I/O-bound
FEATS_ROOT = os.environ.get("VEGO_FEATS_ROOT", f"{ROOT}/cache/feats")


def packed_index(feats):
    """{(dataset, clip_id): (shard path, byte offset, T, n_tok, dim)} of a packed feature cache
    ($ROOT/cache/feats/<feats>/packed/shard*.jsonl + .bin, written by model/cache_finetuned.py), {} if none.
    Packed caches avoid creating one file per clip: file creation on the shared scratch is intermittently very slow."""
    if feats not in _PACKED:
        import glob

        idx = {}
        for jl in sorted(glob.glob(f"{FEATS_ROOT}/{feats}/packed/shard*.jsonl")):
            for line in open(jl):
                e = json.loads(line)
                idx[(e["dataset"], e["clip_id"])] = (jl[:-6] + ".bin", e["offset"], e["T"], e["n_tok"], e["dim"])
        _PACKED[feats] = idx
    return _PACKED[feats]


def has_features(feats, dataset, clip_id):
    return (dataset, clip_id) in packed_index(feats) or os.path.exists(f"{FEATS_ROOT}/{feats}/{dataset}/{clip_id}.npy")


def feature_reader(feats, dataset, clip_id, idx=None):
    """TokSel over a clip's cached features: packed shard if present, else the per-clip .npy (float16, read lazily)."""
    p = packed_index(feats).get((dataset, clip_id))
    if p is not None:
        return TokSel(p[0], idx, packed=p[1:])
    return TokSel(f"{FEATS_ROOT}/{feats}/{dataset}/{clip_id}.npy", idx)


def load_clip(r, feats, tokens="all"):
    sel = feature_reader(feats, r.dataset, r.clip_id, None if tokens == "all" else RICH_TOKENS[tokens])
    ya, yb, k = load_labels(r, len(sel))
    if len(ya) < len(sel):
        sel.shape = (len(ya),) + sel.shape[1:]
    return sel, ya, yb, k


def load_labels(r, T=None):
    """5 Hz labels of a clip (step t = frames 2t, 2t+1): long, lat (-2 = no lateral maneuver, partial label),
    kinematics + road state (T, 9). T defaults to all full steps."""
    z = np.load(os.path.join(r.dir, "labels.npz"))
    T = len(z["long"]) // 2 if T is None else min(T, len(z["long"]) // 2)
    idx = np.arange(T) * 2 + 1
    idx = np.minimum(idx, len(z["long"]) - 1)
    kin = z["kin"][: 2 * T].reshape(T, 2, 4).mean(1) / KIN_SCALE  # NaN where unknown (e.g. ROAD-Waymo)
    road = np.full((T, 5), np.nan, np.float32)
    if "road" in z.files:
        rd = z["road"][: 2 * T]
        rd = rd[np.minimum(idx, len(rd) - 1)]
        road = np.stack([np.clip(rd[:, 0] / 2.0, -3, 3), np.clip(rd[:, 1] / 2.0, 0, 3), np.clip(rd[:, 2] / 5.0, 0, 3),
                         np.clip(rd[:, 3] / 0.3, -3, 3), rd[:, 4]], 1).astype(np.float32)
    # u2 = u1 unified boundaries + label-rule fixes and verified test-pool corrections;
    # "lat_u1" / "lat" = earlier versions
    key = os.environ.get("VEGO_LAT_KEY", "lat_u2")
    if key not in z.files:
        raise KeyError(f"{key} missing in {r.dir}/labels.npz; run data/relabel_boundaries.py and data/relabel_u2.py")
    lat = z[key][idx].astype(np.int64)
    # partial label: "no lateral maneuver" (straight or curve, unknown which) -> encoded as -2
    if "lat_neg" in z.files:
        lat[(lat == IGNORE) & z["lat_neg"][idx]] = NO_MANEUVER
    lon, kr = z["long"][idx].astype(np.int64), np.concatenate([np.clip(kin, -5, 5), road], 1)
    out = frame_mask(z, r)[idx]  # frames outside the data scope: excluded from losses and metrics
    lon[out], lat[out], kr[out] = IGNORE, IGNORE, np.nan
    return lon, lat, kr


class Model(nn.Module):
    def __init__(self, d_in, n_tok, aux_kin, bottleneck, d=384, layers=4, n_kin=4, tok_pos=False):
        super().__init__()
        self.norm_in = nn.LayerNorm(d_in)  # backbone features have large, uneven scales
        self.proj = nn.Linear(d_in, d)
        # learned embedding of each token's spatial slot (which grid cell / layer it pools); without it the
        # attention pooling only sees token content
        self.tok_pos = nn.Parameter(torch.zeros(1, n_tok, d)) if tok_pos else None
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, 6, batch_first=True)
        self.pos = nn.Parameter(torch.zeros(1, 1024, d))
        enc = nn.TransformerEncoderLayer(d, 6, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(enc, layers)
        self.aux_kin, self.bottleneck = aux_kin, bottleneck
        # ego kinematics (v, a, yaw_rate, curvature) [+ road state: offset, half width, edge dist, heading dev, intersection]
        self.kin = nn.Linear(d, n_kin)
        self.kin_emb = nn.Sequential(nn.Linear(n_kin, 64), nn.GELU(), nn.Linear(64, 64))
        dh = d + (64 if bottleneck else 0)
        self.head_a = nn.Linear(dh, len(Long))
        self.head_b = nn.Linear(dh, len(Lat))

    def forward(self, x, valid=None):  # x: (B, T, n_tok, d_in); valid: (B, T) bool, False on padding steps
        B, T, N, _ = x.shape
        t = self.proj(self.norm_in(x)).view(B * T, N, -1)
        if self.tok_pos is not None:
            t = t + self.tok_pos
        h, _ = self.pool(self.q.expand(B * T, -1, -1), t, t)
        # padding steps (clips shorter than the window) must not be attended to: at test time a short clip is one
        # unpadded window (the 25.6 s windows hurt the short-clip datasets without this mask)
        h = self.temporal(h.view(B, T, -1) + self.pos[:, :T], src_key_padding_mask=None if valid is None else ~valid)
        kin = self.kin(h)
        hb = torch.cat([h, self.kin_emb(kin)], -1) if self.bottleneck else h
        return self.head_a(hb), self.head_b(hb), kin


def batches(clips, L, bs, rng, weights, by_class=None):
    """Random windows; clips drawn by `weights`. by_class: {class: [(clip, step), ...]} draws a class uniformly and
    centers the window near one of its steps (class-balanced sampling for classifier re-training).
    Yields x, y_a, y_b, kin, valid mask, clip index."""
    classes = sorted(by_class) if by_class else None
    while True:
        xs, ys_a, ys_b, ks, ms, cs = [], [], [], [], [], []
        for _ in range(bs):
            if classes:
                inst = by_class[classes[rng.integers(len(classes))]]
                i, c = inst[rng.integers(len(inst))]
                f = clips[i][0]
                s = int(np.clip(c - rng.integers(0, L), 0, max(0, len(f) - L)))
            else:
                i = rng.choice(len(clips), p=weights)
                f = clips[i][0]
                s = rng.integers(0, max(1, len(f) - L + 1))
            _, a, b, k = clips[i]
            sl = slice(s, s + L)
            x = np.asarray(f[sl], np.float32)
            n = len(x)
            pad = L - n
            xs.append(np.pad(x, ((0, pad), (0, 0), (0, 0))))
            ys_a.append(np.pad(a[s:s + n], (0, pad), constant_values=IGNORE))
            ys_b.append(np.pad(b[s:s + n], (0, pad), constant_values=IGNORE))
            ks.append(np.pad(k[s:s + n], ((0, pad), (0, 0)), constant_values=np.nan))
            ms.append(np.pad(np.ones(n, bool), (0, pad)))
            cs.append(i)
        yield [torch.from_numpy(np.stack(v)) for v in (xs, ys_a, ys_b, ks, ms)] + [torch.tensor(cs)]


def windows(T, L=64):
    """(start, keep_from, keep_to) of bidirectional windows of the training length L (stride L/2).
    Every step is predicted by the window whose center is closest, so keep ranges tile [0, T) exactly."""
    if T <= L:
        return [(0, 0, T)]
    starts = list(range(0, T - L + 1, L // 2))
    if starts[-1] != T - L:
        starts.append(T - L)
    centers = [st + L / 2 for st in starts]
    bounds = [0] + [int(round((a + b) / 2)) for a, b in zip(centers[:-1], centers[1:])] + [T]
    return [(st, bounds[i], bounds[i + 1]) for i, st in enumerate(starts)]


@torch.no_grad()
def predict(model, f, dev, L=64, with_kin=False):
    """Class probabilities per step (and, with_kin, the kinematics / road-state regression outputs)."""
    T = len(f)
    pa, pb = np.zeros((T, len(Long)), np.float32), np.zeros((T, len(Lat)), np.float32)
    pk = np.zeros((T, model.kin.out_features), np.float32)
    for s, k0, k1 in windows(T, L):
        x = torch.from_numpy(np.asarray(f[s:s + L], np.float32)).unsqueeze(0).to(dev)
        la, lb, kin = model(x)[:3]
        pa[k0:k1] = la.softmax(-1)[0, k0 - s:k1 - s].cpu().numpy()
        pb[k0:k1] = lb.softmax(-1)[0, k0 - s:k1 - s].cpu().numpy()
        pk[k0:k1] = kin[0, k0 - s:k1 - s].float().cpu().numpy()
    return (pa, pb, pk) if with_kin else (pa, pb)


KIN_NAMES = ["speed", "accel", "yaw_rate", "curvature", "lane_offset", "lane_half_width", "edge_dist", "heading_dev",
             "in_intersection"]


def regression_scores(P, K):
    """Test-set R^2 of each regression output against its (scaled) target; AUC for in_intersection."""
    out = {}
    for j in range(P.shape[1]):
        y, p = K[:, j], P[:, j]
        m = ~np.isnan(y)
        if m.sum() < 100:
            continue
        if KIN_NAMES[j] == "in_intersection":
            if 0 < y[m].mean() < 1:
                out["in_intersection_auc"] = float(roc_auc_score(y[m] > 0.5, p[m]))
        else:
            out[KIN_NAMES[j]] = float(1 - np.mean((y[m] - p[m]) ** 2) / np.var(y[m]))
    return out


def ap_table(probs, labels, enum, min_pos=20):
    out = {}
    for c in enum:
        y = labels == int(c)
        # NO_MANEUVER frames are valid negatives for maneuver classes, unknown for straight/curve
        valid = (labels >= 0) | ((labels == NO_MANEUVER) & (int(c) not in LANE_KEEP))
        if y[valid].sum() >= min_pos:
            out[c.name] = float(average_precision_score(y[valid], probs[valid, int(c)]))
    return out


def ce_mean(logits, y):
    """Cross-entropy averaged over labeled entries; 0 (graph-connected) when none."""
    n = (y >= 0).sum()
    return F.cross_entropy(logits, y.clamp(min=IGNORE), ignore_index=IGNORE, reduction="sum") / n.clamp(min=1)


def lateral_loss(lb_f, yb_f):
    loss = ce_mean(lb_f, yb_f)
    nm = yb_f == NO_MANEUVER
    if nm.any():  # set-valued target: -log(p_straight + p_curve), weighted by its share of labeled frames
        lk = torch.logsumexp(lb_f[nm].log_softmax(-1)[:, LANE_KEEP], -1)
        loss = loss - lk.mean() * nm.sum() / ((yb_f >= 0).sum() + nm.sum())
    return loss


def known_classes(train, ds_of, n_cls, col):
    """(n_clips, n_cls) bool: classes each clip's dataset ever labels (partial label spaces, --federated)."""
    seen = {}
    for c, d in zip(train, ds_of):
        y = c[col]
        seen.setdefault(d, np.zeros(n_cls, bool))[np.unique(y[y >= 0])] = True
    for d in seen:
        if col == 2:
            seen[d][LANE_KEEP] = True  # NO_MANEUVER frames need straight / curve
    return torch.from_numpy(np.stack([seen[d] for d in ds_of]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="vjepa2")
    ap.add_argument("--tokens", default="all", choices=["all"] + sorted(RICH_TOKENS),
                    help="token subset of vjepa2x features (readout / layer diagnostics)")
    ap.add_argument("--tok-pos", action="store_true", help="learned embedding of each token's spatial slot")
    ap.add_argument("--window", type=int, default=64, help="steps per window (0.2 s each)")
    ap.add_argument("--federated", action="store_true", help="softmax only over classes the sample's dataset labels")
    ap.add_argument("--crt", type=int, default=0, help="steps of class-balanced re-training of the lateral head")
    ap.add_argument("--aux-kin", action="store_true")
    ap.add_argument("--bottleneck", action="store_true")
    ap.add_argument("--logit-adjust", action="store_true")
    ap.add_argument("--road", action="store_true", help="also regress road-relative state (map datasets)")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--datasets", default="", help="comma list (smoke tests); default all")
    ap.add_argument("--max-test", type=int, default=0, help="smoke tests: only the N shortest test clips per dataset")
    a = ap.parse_args()
    dev = torch.device(f"cuda:{a.gpu}") if a.gpu >= 0 else torch.device("cpu")
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    idx = build_index(a.datasets.split(",") if a.datasets else None)
    have = idx.apply(lambda r: has_features(a.feats, r.dataset, r.clip_id), axis=1)
    idx = idx[have]
    tr_idx, te_idx = idx[idx.split == "train"], idx[idx.split.isin(["test", "val"])]
    if a.max_test:
        te_idx = te_idx.sort_values("n_frames").groupby("dataset").head(a.max_test)
    print(f"train clips {len(tr_idx)} ({tr_idx.n_frames.sum() / 36000:.1f} h), test clips {len(te_idx)} ({te_idx.n_frames.sum() / 36000:.1f} h)"
          f"  scope {os.environ.get('VEGO_SCOPE', 'day_nowz')}, labels {os.environ.get('VEGO_LAT_KEY', 'lat_u2')}", flush=True)
    train = [load_clip(r, a.feats, a.tokens) for r in tr_idx.itertuples()]
    ds_of = list(tr_idx.dataset)
    # dataset-balanced clip sampling (sqrt of each dataset's hours), longer clips proportionally more often
    hours = tr_idx.groupby("dataset").n_frames.sum()
    w = tr_idx.apply(lambda r: np.sqrt(hours[r.dataset]) / hours[r.dataset] * r.n_frames, axis=1).to_numpy()
    w = w / w.sum()

    b_all = np.concatenate([c[2] for c in train])
    prior = np.bincount(b_all[b_all >= 0], minlength=len(Lat)) + 1.0
    log_prior = torch.log(torch.tensor(prior / prior.sum(), dtype=torch.float32)).to(dev)
    ka = known_classes(train, ds_of, len(Long), 1).to(dev) if a.federated else None
    kb = known_classes(train, ds_of, len(Lat), 2).to(dev) if a.federated else None

    d_in, n_tok = train[0][0].shape[-1], train[0][0].shape[1]
    n_kin = 9 if a.road else 4
    model = Model(d_in, n_tok, a.aux_kin, a.bottleneck, n_kin=n_kin, tok_pos=a.tok_pos).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=a.steps, pct_start=0.05)
    gen = batches(train, a.window, 32, rng, w)
    t0 = time.time()
    for step in range(a.steps):
        x, ya, yb, k, m, ci = [t.to(dev) for t in next(gen)]
        la, lb, kp = model(x, m)
        if a.logit_adjust:
            lb = lb + log_prior  # train with prior-added logits; predict with raw logits
        if a.federated:
            la = la.masked_fill(~ka[ci][:, None, :], -1e4)
            lb = lb.masked_fill(~kb[ci][:, None, :], -1e4)
        loss = ce_mean(la.flatten(0, 1), ya.flatten()) + lateral_loss(lb.flatten(0, 1), yb.flatten())
        if a.aux_kin:
            k = k[..., :n_kin]
            km = m.unsqueeze(-1) & ~torch.isnan(k)
            if km.any():
                loss = loss + 0.5 * F.smooth_l1_loss(kp[km], k[km])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % 500 == 0:
            print(f"step {step + 1} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)

    if a.crt:  # decoupled classifier re-training: lateral head only, class-balanced windows
        by_class = {}
        for i, c in enumerate(train):
            for cls in np.unique(c[2][c[2] >= 0]):
                by_class.setdefault(int(cls), []).extend((i, int(t)) for t in np.flatnonzero(c[2] == cls))
        by_class = {c: [v[j] for j in rng.choice(len(v), min(len(v), 20000), replace=False)]
                    for c, v in by_class.items() if len(v) >= 20}
        for p in model.parameters():
            p.requires_grad_(False)
        model.head_b.reset_parameters()
        for p in model.head_b.parameters():
            p.requires_grad_(True)
        opt2 = torch.optim.AdamW(model.head_b.parameters(), lr=1e-3, weight_decay=0.05)
        gen2 = batches(train, a.window, 32, rng, w, by_class=by_class)
        for step in range(a.crt):
            x, ya, yb, k, m, ci = [t.to(dev) for t in next(gen2)]
            lb = model(x, m)[1]
            if a.federated:
                lb = lb.masked_fill(~kb[ci][:, None, :], -1e4)
            loss = lateral_loss(lb.flatten(0, 1), yb.flatten())
            opt2.zero_grad(set_to_none=True)
            loss.backward()
            opt2.step()
        print(f"cRT {a.crt} steps over {len(by_class)} classes, last loss {loss.item():.3f}", flush=True)

    model.eval()
    res = {"tag": a.tag, "args": vars(a), "scope": os.environ.get("VEGO_SCOPE", "day_nowz"),
           "lat_key": os.environ.get("VEGO_LAT_KEY", "lat_u2")}
    slices = [("all_test", te_idx)] + [(f"{d}_test", te_idx[te_idx.dataset == d]) for d in sorted(te_idx.dataset.unique())]
    if res["scope"] == "all":  # night / work-zone data are isolated otherwise; HDD test is exempt
        slices += [("night", te_idx[te_idx.night]), ("work_zone", te_idx[te_idx.work_zone])]
    pred_dir = f"{ROOT}/runs/m1/pred_{a.tag}_s{a.seed}"  # per-clip test predictions, for error analysis
    os.makedirs(pred_dir, exist_ok=True)
    preds = {}
    P_k, K_k = [], []
    for r in te_idx.itertuples():  # predict every test clip once; the slices reuse the predictions
        f, ya, yb, k = load_clip(r, a.feats, a.tokens)
        pa, pb, pk = predict(model, f, dev, a.window, with_kin=True)
        preds[(r.dataset, r.clip_id)] = (pa, pb, ya, yb)
        P_k.append(pk); K_k.append(k[:len(pk), :pk.shape[1]])
        np.savez_compressed(f"{pred_dir}/{r.dataset}__{r.clip_id}.npz", pa=pa.astype(np.float16), pb=pb.astype(np.float16))
    for name, sub in slices:
        if not len(sub):
            continue
        P_a, P_b, Y_a, Y_b = map(np.concatenate, zip(*[preds[(r.dataset, r.clip_id)] for r in sub.itertuples()]))
        apa, apb = ap_table(P_a, Y_a, Long), ap_table(P_b, Y_b, Lat)
        if not apb:
            continue
        res[name] = {"mAP_A": float(np.mean(list(apa.values()))), "mAP_B": float(np.mean(list(apb.values()))), "AP_A": apa, "AP_B": apb}
        print(f"[{name}] mAP_A {res[name]['mAP_A']:.3f}  mAP_B {res[name]['mAP_B']:.3f}", flush=True)
        print("   B:", {k: round(v, 3) for k, v in apb.items()}, flush=True)
    if a.aux_kin:  # how well the head regresses kinematics / road state from the features (test clips)
        res["regression"] = regression_scores(np.concatenate(P_k), np.concatenate(K_k))
        print("regression (test R^2 / AUC):", {k: round(v, 3) for k, v in res["regression"].items()}, flush=True)
    os.makedirs(f"{ROOT}/runs/m1", exist_ok=True)
    json.dump(res, open(f"{ROOT}/runs/m1/{a.tag}_s{a.seed}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
