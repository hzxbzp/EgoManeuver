"""HDD official benchmark (TRN/OadTR protocol): 11 goal-oriented classes, per-frame AP over the official
3 fps frames of the 36 test sessions (background excluded from the mean), online (causal) or offline.

The model is the M1 temporal model on cached features plus an HDD head trained on the official labels of
the 100 train sessions (as every published method does). Our own A/B heads can be trained jointly on all
datasets (--multitask) to measure transfer from our data. Official frame i is evaluated at its real time
(official_t, see docs/hdd_alignment.md), i.e. on exactly the frames used by published methods.

Usage: python model/train_hdd_bench.py --feats vjepa2_causal --causal --gpu 0 --tag hdd_online_vj2c
"""

import argparse
import importlib.util
from egokin.scripts import script_path  # noqa: E402
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from egokin.data.scope import frame_mask, official_keep, scope
from sklearn.metrics import average_precision_score

_spec = importlib.util.spec_from_file_location("m1", script_path("train_m1.py"))
m1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m1)
IGNORE = m1.IGNORE
GOAL = ["background", "intersection passing", "left turn", "right turn", "left lane change", "right lane change",
        "left lane branch", "right lane branch", "crosswalk passing", "railroad passing", "merge", "U-turn"]


CTX = ["intersection", "crosswalk", "railroad", "work_zone", "tunnel", "highway", "ramp", "night", "merge_zone", "branch_zone"]


def ctx_targets(r, T):
    """(T, len(CTX)) float targets at 5 Hz from labels.npz ctx_* / hsd_* arrays; NaN where unknown."""
    z = np.load(os.path.join(r.dir, "labels.npz"))
    n = len(z["long"])
    idx = np.minimum(np.arange(T) * 2 + 1, n - 1)
    out = np.full((T, len(CTX)), np.nan, np.float32)
    known = {"hdd": {"intersection", "crosswalk", "railroad", "work_zone", "tunnel", "highway"},
             "hsd": set(CTX), "nuscenes": {"intersection", "crosswalk", "work_zone", "night"},
             "av2": {"intersection", "crosswalk"}, "nuplan": {"intersection", "crosswalk"}}.get(r.dataset, set())
    for j, c in enumerate(CTX):
        if c not in known:
            continue
        if c == "merge_zone":
            v = (z["hsd_merge_goreonleft"] > 0) | (z["hsd_merge_goreonright"] > 0)
        elif c == "branch_zone":
            v = (z["hsd_branch_goreonleft"] > 0) | (z["hsd_branch_goreonright"] > 0)
        elif f"ctx_{c}" in z.files:
            v = z[f"ctx_{c}"]
        else:
            continue
        out[:, j] = np.asarray(v)[idx].astype(np.float32)
    out[frame_mask(z, r)[idx]] = np.nan  # outside the data scope
    return out


class BenchModel(m1.Model):
    def __init__(self, *args, causal=False, n_ctx=len(CTX), **kw):
        super().__init__(*args, **kw)
        dh = self.head_b.in_features
        self.head_hdd = nn.Linear(dh, len(GOAL))
        self.head_ctx = nn.Linear(dh, n_ctx)
        self.causal = causal

    def forward(self, x, valid=None):  # valid: (B, T) bool, False on padding steps (as m1.Model)
        B, T, N, _ = x.shape
        t = self.proj(self.norm_in(x)).view(B * T, N, -1)
        if self.tok_pos is not None:
            t = t + self.tok_pos
        h, _ = self.pool(self.q.expand(B * T, -1, -1), t, t)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device) if self.causal else None
        h = self.temporal(h.view(B, T, -1) + self.pos[:, :T], mask=mask, is_causal=self.causal,
                          src_key_padding_mask=None if valid is None else ~valid)
        kin = self.kin(h)
        hb = torch.cat([h, self.kin_emb(kin)], -1) if self.bottleneck else h
        return self.head_a(hb), self.head_b(hb), kin, self.head_hdd(hb), self.head_ctx(hb)


def hdd_targets(r, T):
    """Official label for each 5 Hz step: nearest official frame in real time (<= 0.2 s away)."""
    z = np.load(os.path.join(r.dir, "labels.npz"))
    if "official_goal" not in z.files:
        return np.full(T, IGNORE, np.int64)
    ot, og = z["official_t"].astype(np.float64), z["official_goal"].astype(np.int64)
    ts = np.arange(T) * 0.2 + 0.1
    j = np.clip(np.searchsorted(ot, ts), 1, len(ot) - 1)
    j = np.where(np.abs(ot[j - 1] - ts) < np.abs(ot[j] - ts), j - 1, j)
    y = np.where(np.abs(ot[j] - ts) <= 0.2, og[j], IGNORE)
    out = frame_mask(z, r)
    y[out[np.minimum(np.arange(T) * 2 + 1, len(out) - 1)]] = IGNORE  # outside the data scope
    return y


@torch.no_grad()
def run(model, f, dev, causal, L=64):
    """Per-step HDD probabilities. Causal: overlapping windows, each kept output has >= L/2 steps of history."""
    T = len(f)
    out = np.zeros((T, len(GOAL)), np.float32)
    if not causal:  # same window length as in training (positions >= L were never trained)
        for s, k0, k1 in m1.windows(T, L):
            x = torch.from_numpy(np.asarray(f[s:s + L], np.float32)).unsqueeze(0).to(dev)
            out[k0:k1] = model(x)[3].softmax(-1)[0, k0 - s:k1 - s].cpu().numpy()
        return out
    for s in range(0, T, L // 2):
        lo = max(0, s - L // 2)
        x = torch.from_numpy(np.asarray(f[lo:s + L // 2], np.float32)).unsqueeze(0).to(dev)
        p = model(x)[3].softmax(-1)[0].cpu().numpy()
        out[s:s + L // 2] = p[s - lo:]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="vjepa2_causal")
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--multitask", action="store_true", help="also train A/B heads on all datasets")
    ap.add_argument("--aux-kin", action="store_true")
    ap.add_argument("--bottleneck", action="store_true")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--window", type=int, default=64, help="training/eval window in 5 Hz steps")
    ap.add_argument("--balance", type=float, default=0.0,
                    help="oversample windows containing rare official classes (0 = off, e.g. 0.5)")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--tok-pos", action="store_true", help="learned embedding of each token's slot")
    ap.add_argument("--ctx", action="store_true", help="scene-context head (HSD places, HDD goal context, map flags)")
    a = ap.parse_args()
    dev = torch.device(f"cuda:{a.gpu}")
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    idx = m1.build_index(None if a.multitask else ["hdd"])
    idx = idx[idx.apply(lambda r: m1.has_features(a.feats, r.dataset, r.clip_id), axis=1)]
    tr_idx = idx[idx.split == "train"]
    te_idx = idx[(idx.dataset == "hdd") & (idx.split == "test")]
    train = []
    for r in tr_idx.itertuples():
        f, ya, yb, k = m1.load_clip(r, a.feats)
        train.append((f, ya, yb, k, hdd_targets(r, len(f)) if r.dataset == "hdd" else np.full(len(f), IGNORE),
                      ctx_targets(r, len(f))))
    print(f"train clips {len(train)}, HDD test sessions {len(te_idx)}", flush=True)
    hours = tr_idx.groupby("dataset").n_frames.sum()
    w = tr_idx.apply(lambda r: np.sqrt(hours[r.dataset]) / hours[r.dataset] * r.n_frames, axis=1).to_numpy()
    w = w / w.sum()

    model = BenchModel(train[0][0].shape[-1], train[0][0].shape[1], a.aux_kin, a.bottleneck, causal=a.causal,
                       layers=a.layers, tok_pos=a.tok_pos).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=a.steps, pct_start=0.05)
    L, bs = a.window, 32
    # rare official classes: merge, crosswalk, railroad, U-turn, branches, lane changes
    rare = {4, 5, 6, 7, 8, 9, 10, 11}
    rare_starts = [np.flatnonzero(np.isin(c[4], list(rare))) for c in train]
    t0 = time.time()
    for step in range(a.steps):
        xs, ya, yb, ks, yh, yc = [], [], [], [], [], []
        for i in rng.choice(len(train), size=bs, p=w):
            f, ca, cb, ck, ch, cc = train[i]
            s = rng.integers(0, max(1, len(f) - L + 1))
            if a.balance > 0 and len(rare_starts[i]) and rng.random() < a.balance:
                c = rng.choice(rare_starts[i])  # center the window on a rare-class step
                s = int(np.clip(c - L // 2 + rng.integers(-L // 4, L // 4 + 1), 0, max(0, len(f) - L)))
            pad = L - len(f[s:s + L])
            xs.append(np.pad(np.asarray(f[s:s + L], np.float32), ((0, pad), (0, 0), (0, 0))))
            ya.append(np.pad(ca[s:s + L], (0, pad), constant_values=IGNORE))
            yb.append(np.pad(cb[s:s + L].clip(min=IGNORE), (0, pad), constant_values=IGNORE))
            ks.append(np.pad(ck[s:s + L], ((0, pad), (0, 0)), constant_values=np.nan))
            yh.append(np.pad(ch[s:s + L], (0, pad), constant_values=IGNORE))
            yc.append(np.pad(cc[s:s + L], ((0, pad), (0, 0)), constant_values=np.nan))
        x, ya, yb, ks, yh, yc = [torch.from_numpy(np.stack(v)).to(dev) for v in (xs, ya, yb, ks, yh, yc)]
        la, lb, kp, lh, lc = model(x)
        loss = F.cross_entropy(lh.flatten(0, 1), yh.flatten(), ignore_index=IGNORE)
        if a.multitask:
            loss = loss + F.cross_entropy(la.flatten(0, 1), ya.flatten(), ignore_index=IGNORE) + \
                F.cross_entropy(lb.flatten(0, 1), yb.flatten(), ignore_index=IGNORE)
        if a.ctx:
            cm = ~torch.isnan(yc)
            if cm.any():
                loss = loss + 0.5 * F.binary_cross_entropy_with_logits(lc[cm], yc[cm])
        if a.aux_kin:
            k = ks[..., :4]
            km = ~torch.isnan(k)
            if km.any():
                loss = loss + 0.5 * F.smooth_l1_loss(kp[km], k[km])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % 1000 == 0:
            print(f"step {step + 1} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)

    model.eval()
    P, Y, K = [], [], []
    for r in te_idx.itertuples():
        f = m1.feature_reader(a.feats, "hdd", r.clip_id)
        z = np.load(os.path.join(r.dir, "labels.npz"))
        prob = run(model, f, dev, a.causal, L)
        step = np.clip(np.round((z["official_t"] - 0.1) / 0.2).astype(int), 0, len(prob) - 1)
        P.append(prob[step]); Y.append(z["official_goal"].astype(int)); K.append(official_keep(z, r))
    P, Y, K = np.concatenate(P), np.concatenate(Y), np.concatenate(K)
    ap_c = {GOAL[c]: float(average_precision_score(Y == c, P[:, c])) for c in range(1, len(GOAL))}
    ap_s = {GOAL[c]: float(average_precision_score(Y[K] == c, P[K, c])) for c in range(1, len(GOAL))}
    res = {"tag": a.tag, "args": vars(a), "scope": scope(), "mAP": float(np.mean(list(ap_c.values()))), "AP": ap_c,
           "mAP_scope": float(np.mean(list(ap_s.values()))), "AP_scope": ap_s, "n_frames": int(len(Y)), "n_frames_scope": int(K.sum())}
    print(f"[HDD official {'online' if a.causal else 'offline'}] mAP {res['mAP'] * 100:.1f}", flush=True)
    print("   ", {k: round(v * 100, 1) for k, v in ap_c.items()}, flush=True)
    os.makedirs(f"{m1.ROOT}/runs/hdd_bench", exist_ok=True)
    json.dump(res, open(f"{m1.ROOT}/runs/hdd_bench/{a.tag}_s{a.seed}.json", "w"), indent=1)
    torch.save(model.state_dict(), f"{m1.ROOT}/runs/hdd_bench/{a.tag}_s{a.seed}.pt")  # warm start for E2E


if __name__ == "__main__":
    main()
