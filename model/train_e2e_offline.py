"""M2 end-to-end, offline (clip curation): V-JEPA 2-L or 2.1-L (--backbone), adapted with LoRA on blocks >=
--first-trainable or fully fine-tuned with layer-wise LR decay (--adapt), on non-overlapping 16-frame windows.
Each window gives 8 steps x the 74-token rich layout of the cached vjepa2x / vjepa21x features
(egokin/models/backbones.py); --tokens selects the readout (pool9 = global + 2x4 of the last layer, as before).
Bidirectional temporal model, heads for our axes A/B with the kinematics (+ road-relative state, --road)
bottleneck, the HDD goal layer and scene context (--ctx). --dense adds the perspective road-structure head on the
encoder's last-layer patch tokens (M2-b: ego lane, other lanes, markings, curb, construction; EPOSH + BDD100K
keyframes, egokin/data/dense.py).

Phase 1 (--warm-steps) trains the head alone on the cached frozen features and evaluates it (the frozen
baseline, same code path); phase 2 fine-tunes LoRA + head end to end on decoded video. Both are evaluated on
the test pool (per dataset, night, work zone) and on HDD (official frames, offline).

torchrun --nproc_per_node 4 model/train_e2e_offline.py --tag e2e_off_v1 [--backbone vjepa21] [--adapt full]
    [--tokens grid33 --tok-pos] [--road] [--ctx]
"""

import argparse
import datetime
import importlib.util
from egokin.scripts import script_path  # noqa: E402
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from egokin.data import dense as dn
from egokin.data.scope import official_keep, scope
from egokin.models.backbones import Backbone
from egokin.models.vjepa_lora import prep
from egokin.taxonomy import Lat, Long


def _load(name):
    spec = importlib.util.spec_from_file_location(name, script_path(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m1 = _load("train_m1")
hb = _load("train_hdd_bench")
e2e = _load("train_e2e")
IGNORE, NO_MANEUVER, LANE_KEEP, GOAL, ROOT = m1.IGNORE, m1.NO_MANEUVER, m1.LANE_KEEP, hb.GOAL, m1.ROOT
FEATS = "vjepa2x_rinterp"  # cached rich-feature dir, set in main() from --backbone / --rope
TOK = list(range(9))  # token indices into the 74-token layout (m1.RICH_TOKENS), set in main() from --tokens
WIN = 64  # steps per window (--L): training windows and the evaluation tiling (m1.windows)


NO_BEHAVIOR = {"hsd", "eposh"}  # context / dense supervision only


def build_clips(splits, datasets, ctx, use_dense=False):
    idx = m1.build_index(datasets)
    idx = idx[idx.split.isin(splits)]
    drop = (set() if ctx else {"hsd"}) | (set() if use_dense else {"eposh"})
    idx = idx[~idx.dataset.isin(drop)]
    clips = []
    for r in idx.itertuples():
        have = m1.has_features(FEATS, r.dataset, r.clip_id)
        feats = (r.dataset, r.clip_id) if have else None  # packed or per-clip cache (m1.feature_reader)
        video = os.path.join(r.dir, "video.mp4")
        dpath = os.path.join(r.dir, "dense.npz")
        if not os.path.exists(video) or not (have or r.dataset in NO_BEHAVIOR):
            continue
        ya, yb, k = m1.load_labels(r, len(m1.feature_reader(FEATS, *feats)) if have else None)
        T = len(ya)
        if T < 8:
            continue
        c = dict(id=r.clip_id, dataset=r.dataset, split=r.split, dir=r.dir, T=T, night=bool(r.night), work_zone=bool(r.work_zone),
                 feats=feats, video=video, y_a=ya, y_b=yb, kin=k.astype(np.float32),
                 y_hdd=hb.hdd_targets(r, T) if r.dataset == "hdd" else np.full(T, IGNORE),
                 ctx=hb.ctx_targets(r, T) if ctx else np.full((T, len(hb.CTX)), np.nan, np.float32))
        if use_dense and os.path.exists(dpath):
            z = np.load(dpath)
            keep = z["f"] // 2 < T
            y = z["y"][keep].astype(np.float16)
            y[:, ~z["ch"]] = np.nan  # channels not annotated in this dataset are unknown
            c.update(dense_f=z["f"][keep], dense_y=y)
        clips.append(c)
    return clips


class Windows(torch.utils.data.Dataset):
    """Random L-step windows aligned with the 16-frame encoder windows; dataset-balanced like M1 (sqrt of each
    dataset's hours, longer clips proportionally more often). mode "feats": cached frozen features; "video":
    the 2L decoded frames."""

    def __init__(self, clips, L, mode, seed, length=10 ** 8, dense_frac=0.0):
        self.clips, self.L, self.mode, self.seed, self.length, self.dense_frac = clips, L, mode, seed, length, dense_frac
        ds = np.array([c["dataset"] for c in clips])
        T = np.array([c["T"] for c in clips], float)
        hours = {d: T[ds == d].sum() for d in set(ds)}
        ok = np.array([d not in NO_BEHAVIOR or (d == "hsd" and not np.isnan(c["ctx"]).all()) for d, c in zip(ds, clips)])
        if mode == "feats":
            ok &= np.array([c["feats"] is not None for c in clips])
        self.w = np.array([np.sqrt(hours[d]) / hours[d] for d in ds]) * T * ok
        self.w /= self.w.sum()
        self.dense_ids = [i for i, c in enumerate(clips) if "dense_f" in c and len(c["dense_f"])]

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i))
        T, L = 0, self.L
        if self.dense_ids and rng.random() < self.dense_frac:  # window around an annotated road-structure frame
            c = self.clips[self.dense_ids[rng.integers(len(self.dense_ids))]]
            T = c["T"]
            t = int(rng.choice(c["dense_f"])) // 2
            s = int(np.clip(8 * ((t - int(rng.integers(0, L - 7))) // 8), 0, 8 * (max(0, T - L) // 8)))  # t inside
        else:
            c = self.clips[rng.choice(len(self.clips), p=self.w)]
            T = c["T"]
            s = 8 * int(rng.integers(0, max(0, T - L) // 8 + 1))
        n = min(L, T - s)
        pad = L - n
        lab = lambda y: np.pad(y[s:s + n], (0, pad), constant_values=IGNORE)
        out = dict(y_a=lab(c["y_a"]), y_b=lab(c["y_b"]), y_hdd=lab(c["y_hdd"]), valid=np.arange(L) < n,
                   kin=np.pad(c["kin"][s:s + n], ((0, pad), (0, 0)), constant_values=np.nan),
                   y_ctx=np.pad(c["ctx"][s:s + n], ((0, pad), (0, 0)), constant_values=np.nan))
        if self.dense_frac > 0:
            yd = np.full((L, len(dn.DENSE)) + dn.GRID, np.nan, np.float16)
            for f, y in zip(c.get("dense_f", []), c.get("dense_y", [])):
                if s <= f // 2 < s + n:
                    yd[f // 2 - s] = y
            out["y_dense"] = yd
        if self.mode == "feats":
            f = np.asarray(m1.feature_reader(FEATS, *c["feats"], TOK)[s:s + n], np.float32)  # one pread per window
            out["x"] = np.pad(f, ((0, pad), (0, 0), (0, 0)))
        else:
            out["frames"] = e2e.decode(c["video"], 2 * s, 2 * s + 2 * L - 1)  # past the end: last frame repeated
        return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in out.items()}


class Encoder(torch.nn.Module):
    """windows -> selected step tokens (B, 8, n_tok, D) (+ road-structure logits per last-layer patch token)."""

    def __init__(self, bb, tok, dense=False):
        super().__init__()
        self.bb, self.tok = bb, list(tok)
        self.dense = torch.nn.Sequential(torch.nn.LayerNorm(1024), torch.nn.Linear(1024, len(dn.DENSE))) if dense else None

    def forward(self, wins):
        rich, last = self.bb(wins)  # (B, 8, 74, D), (B, 8, 14, 28, D)
        x = rich[:, :, self.tok]
        return (x, self.dense(last)) if self.dense is not None else (x, None)


def encode(enc, frames):
    """frames (B, 2L, H, W, 3) uint8 -> (B, L, n_tok, 1024) [+ dense logits (B, L, 14, 28, C)]."""
    B, N = frames.shape[:2]
    x = prep(frames.flatten(0, 1)).view(B * N // 16, 16, 3, 224, 448)
    g, d = enc(x)
    return g.reshape(B, N // 2, *g.shape[2:]), (d.reshape(B, N // 2, *d.shape[2:]) if d is not None else None)


def dense_loss(d, b, dev):
    """BCE with soft targets on annotated (step, cell, channel) entries; 0 but graph-connected otherwise."""
    y = b["y_dense"].to(dev).float().permute(0, 1, 3, 4, 2)  # (B, L, 14, 28, C)
    m = ~torch.isnan(y)
    return e2e.wmean(F.binary_cross_entropy_with_logits(d.float(), torch.nan_to_num(y), reduction="none"), m.float())


def loss_fn(out, b, dev, a):
    la, lb, kp, lh, lc = [o.float() for o in out]
    ya, yb, yh = b["y_a"].to(dev), b["y_b"].to(dev), b["y_hdd"].to(dev)
    ce = lambda lg, y: F.cross_entropy(lg.transpose(1, 2), y.clamp(min=IGNORE), ignore_index=IGNORE, reduction="none")
    wm = e2e.wmean
    loss = wm(ce(la, ya), (ya >= 0).float()) + wm(ce(lh, yh), (yh >= 0).float())
    # axis B with the set-valued "no lateral maneuver" target: -log(p_straight + p_curve), as in M1
    nm = (yb == NO_MANEUVER).float()
    lk = torch.logsumexp(lb.log_softmax(-1)[..., LANE_KEEP], -1)
    loss = loss + ((ce(lb, yb) * (yb >= 0).float()).sum() - (lk * nm).sum()) / ((yb >= 0).float().sum() + nm.sum()).clamp(min=1)
    kt = b["kin"].to(dev)[..., :kp.shape[-1]]
    loss = loss + 0.5 * wm(F.smooth_l1_loss(kp, torch.nan_to_num(kt), reduction="none"), (~torch.isnan(kt)).float())
    yc = b["y_ctx"].to(dev)
    lctx = wm(F.binary_cross_entropy_with_logits(lc, torch.nan_to_num(yc), reduction="none"), (~torch.isnan(yc)).float())
    return loss + (0.5 if a.ctx else 0.0) * lctx


@torch.no_grad()
def fresh_features(enc, video, T, dev, batch=16):
    """Fine-tuned step tokens for a whole clip, decoded in 512-frame chunks (as model/cache_features.py)."""
    cf = _load("cache_features")
    out = []
    for frames in cf.frame_chunks(video, 512):
        n = len(frames)
        if n % 16:
            frames = np.concatenate([frames, np.repeat(frames[-1:], 16 - n % 16, 0)])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = prep(torch.from_numpy(frames).to(dev)).view(-1, 16, 3, 224, 448)
            g = torch.cat([enc(x[i:i + batch])[0] for i in range(0, len(x), batch)])
        out.append(g.flatten(0, 1)[: (n + 1) // 2].float().cpu().numpy())
    G = np.concatenate(out)[:T]
    return np.pad(G, ((0, T - len(G)), (0, 0), (0, 0)), mode="edge") if len(G) < T else G


@torch.no_grad()
def predict(head, G, dev, L=None):
    """Bidirectional windows of the training length (WIN unless given) tiling the clip (m1.windows)."""
    L = L or WIN
    T = len(G)
    pa, pb, ph = np.zeros((T, len(Long)), np.float32), np.zeros((T, len(Lat)), np.float32), np.zeros((T, len(GOAL)), np.float32)
    pk = np.zeros((T, head.kin.out_features), np.float32)  # kinematics (+ road state) regression outputs
    for s, k0, k1 in m1.windows(T, L):
        x = torch.from_numpy(np.asarray(G[s:s + L], np.float32)).unsqueeze(0).to(dev)
        la, lb, kin, lh, _ = head(x)
        for p, lg in ((pa, la), (pb, lb), (ph, lh)):
            p[k0:k1] = lg.float().softmax(-1)[0, k0 - s:k1 - s].cpu().numpy()
        pk[k0:k1] = kin.float()[0, k0 - s:k1 - s].cpu().numpy()
    return pa, pb, ph, pk


@torch.no_grad()
def evaluate(head, clips, dev, rank, world, out_dir, enc=None):
    """Per-clip predictions (sharded across ranks); rank 0 then computes the tables with `summarize`.
    enc None: cached frozen features."""
    head.eval()
    if enc is not None:
        enc.eval()
    os.makedirs(out_dir, exist_ok=True)
    for c in sorted(clips, key=lambda c: -c["T"])[rank::world]:
        G = fresh_features(enc, c["video"], c["T"], dev) if enc is not None else \
            np.asarray(m1.feature_reader(FEATS, *c["feats"], TOK)[0:c["T"]], np.float32)
        pa, pb, ph, pk = predict(head, G, dev)
        extra = {}
        if c["dataset"] == "hdd":
            z = np.load(os.path.join(c["dir"], "labels.npz"))
            step = np.clip(np.round((z["official_t"] - 0.1) / 0.2).astype(int), 0, c["T"] - 1)
            extra = dict(p_hdd=ph[step], y_hdd=z["official_goal"].astype(int), keep_hdd=official_keep(z, SimpleNamespace(dataset=c["dataset"], split=c["split"])))
        np.savez(os.path.join(out_dir, f"{c['dataset']}__{c['id']}.npz"), pa=pa.astype(np.float16), pb=pb.astype(np.float16),
                 pk=pk.astype(np.float32), **extra)
    head.train()
    if enc is not None:
        enc.train()


@torch.no_grad()
def evaluate_dense(enc, clips, dev, max_frames=3000):
    """Per-channel AP of the road-structure head over annotated (frame, cell) pairs of held-out clips."""
    enc.eval()
    P, Y = [], []
    for c in clips:
        for f, y in zip(c["dense_f"], c["dense_y"]):
            w0 = 16 * (int(f) // 16)  # the 16-frame encoder window holding frame f
            frames = e2e.decode(c["video"], w0, w0 + 15)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, d = enc(prep(torch.from_numpy(frames).to(dev)).unsqueeze(0))
            P.append(torch.sigmoid(d[0, (int(f) % 16) // 2].float()).cpu().numpy().reshape(-1, len(dn.DENSE)))
            Y.append(y.astype(np.float32).transpose(1, 2, 0).reshape(-1, len(dn.DENSE)))
            if len(P) >= max_frames:
                break
    enc.train()
    if not P:
        return {}
    P, Y = np.concatenate(P), np.concatenate(Y)
    out = {}
    for k, name in enumerate(dn.DENSE):
        m = ~np.isnan(Y[:, k])
        pos = Y[m, k] > 0.5
        if m.sum() and 0 < pos.sum() < m.sum():
            out[name] = float(average_precision_score(pos, P[m, k]))
    return {"mAP": float(np.mean(list(out.values()))) if out else None, "AP": out, "frames": int(len(P) // (dn.GRID[0] * dn.GRID[1]))}


def summarize(clips, pred_dir):
    res = {}
    slices = [("all_test", clips)] + [(f"{d}_test", [c for c in clips if c["dataset"] == d]) for d in sorted({c["dataset"] for c in clips})] + \
        [("night", [c for c in clips if c["night"]]), ("work_zone", [c for c in clips if c["work_zone"]])]
    preds = {(c["dataset"], c["id"]): np.load(os.path.join(pred_dir, f"{c['dataset']}__{c['id']}.npz")) for c in clips}
    for name, sub in slices:
        if not sub:
            continue
        P_a = np.concatenate([preds[c["dataset"], c["id"]]["pa"] for c in sub]).astype(np.float32)
        P_b = np.concatenate([preds[c["dataset"], c["id"]]["pb"] for c in sub]).astype(np.float32)
        Y_a, Y_b = np.concatenate([c["y_a"] for c in sub]), np.concatenate([c["y_b"] for c in sub])
        apa, apb = m1.ap_table(P_a, Y_a, Long), m1.ap_table(P_b, Y_b, Lat)
        res[name] = {"mAP_A": float(np.mean(list(apa.values()))) if apa else None,
                     "mAP_B": float(np.mean(list(apb.values()))) if apb else None, "AP_A": apa, "AP_B": apb}
    hdd = [preds[c["dataset"], c["id"]] for c in clips if c["dataset"] == "hdd"]
    if hdd:
        P, Y = np.concatenate([z["p_hdd"] for z in hdd]), np.concatenate([z["y_hdd"] for z in hdd])
        K = np.concatenate([z["keep_hdd"] for z in hdd])
        ap_c = {GOAL[k]: float(average_precision_score(Y == k, P[:, k])) for k in range(1, len(GOAL))}
        ap_s = {GOAL[k]: float(average_precision_score(Y[K] == k, P[K, k])) for k in range(1, len(GOAL))}
        res["hdd_official_offline"] = {"mAP": float(np.mean(list(ap_c.values()))), "AP": ap_c,
                                       "mAP_scope": float(np.mean(list(ap_s.values()))), "AP_scope": ap_s}
        res["scope"] = scope()
    pk = [(preds[c["dataset"], c["id"]]["pk"], c["kin"]) for c in clips if "pk" in preds[c["dataset"], c["id"]].files]
    if pk:  # test R^2 of the regression outputs (kinematics; road state with --road), AUC for in-intersection
        P = np.concatenate([p for p, _ in pk])
        K = np.concatenate([k[:len(p), :p.shape[1]] for p, k in pk])
        res["regression"] = m1.regression_scores(P, K)
    return res


def report(res, name):
    line = " ".join(f"{k}:{v['mAP_B']:.3f}" for k, v in res.items() if isinstance(v, dict) and v.get("mAP_B") is not None)
    hdd = res.get("hdd_official_offline", {}).get("mAP")
    if res.get("regression"):
        print(f"[{name}] regression (test R^2 / AUC):", {k: round(v, 3) for k, v in res["regression"].items()}, flush=True)
    print(f"[{name}] mAP_A(all) {res['all_test']['mAP_A']:.3f}  mAP_B {line}" + (f"  HDD offline {hdd * 100:.1f}" if hdd else ""), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--warm-steps", type=int, default=6000, help="phase 1: head on cached frozen features (bs 32 total)")
    ap.add_argument("--steps", type=int, default=10000, help="phase 2: end to end (0: phase 1 only)")
    ap.add_argument("--seed", type=int, default=0, help="head init and window sampling")
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--bs", type=int, default=4, help="per GPU, phase 2")
    ap.add_argument("--backbone", default="vjepa2", choices=["vjepa2", "vjepa21"])
    ap.add_argument("--adapt", default="lora", choices=["lora", "full"],
                    help="lora: LoRA r=16 on blocks >= --first-trainable; full: all weights, layer-wise LR decay")
    ap.add_argument("--first-trainable", type=int, default=12)
    ap.add_argument("--grad-ckpt", action="store_true", help="activation checkpointing in the backbone")
    ap.add_argument("--tokens", default="pool9", choices=sorted(m1.RICH_TOKENS), help="readout (m1.RICH_TOKENS)")
    ap.add_argument("--tok-pos", action="store_true", help="learned embedding of each token's slot in the head")
    ap.add_argument("--lr", type=float, default=1e-4, help="head LR in phase 2")
    ap.add_argument("--lr-lora", type=float, default=1e-4, help="backbone LR (--adapt lora)")
    ap.add_argument("--lr-full", type=float, default=5e-5, help="top-block LR (--adapt full)")
    ap.add_argument("--lr-decay", type=float, default=0.8, help="layer-wise LR decay (--adapt full)")
    ap.add_argument("--road", action="store_true", help="road-relative state in the kinematic bottleneck (9 dims)")
    ap.add_argument("--ctx", action="store_true", help="scene-context head (HSD places etc.)")
    ap.add_argument("--dense", action="store_true", help="perspective road-structure head (EPOSH + BDD100K keyframes)")
    ap.add_argument("--dense-frac", type=float, default=0.15, help="share of windows centered on annotated frames")
    ap.add_argument("--dense-w", type=float, default=1.0)
    ap.add_argument("--datasets", default="", help="comma list; default all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-only", default="", help="checkpoint (.pt) with backbone LoRA + head")
    ap.add_argument("--init-head", default="", help="phase-1 head (head_frozen.pt of a run with the same head config): skip phase 1")
    ap.add_argument("--eval-frozen", action="store_true", help="only evaluate the --init-head head on cached features")
    ap.add_argument("--max-test", type=int, default=0, help="smoke test: only the N shortest test clips per dataset")
    ap.add_argument("--exclude-val", action="store_true",
                    help="train without the benchmark validation groups (build_segments.val_groups)")
    ap.add_argument("--rope", default="interp", choices=["int", "interp"],
                    help="V-JEPA 2 RoPE positions; also selects the matching cached-feature directory")
    a = ap.parse_args()
    global FEATS, TOK, WIN
    from egokin.models.vjepa_rope import feats_name
    FEATS = feats_name("vjepa2x", a.rope) if a.backbone == "vjepa2" else "vjepa21x"
    TOK = m1.RICH_TOKENS[a.tokens]
    WIN = a.L

    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=3))
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    torch.cuda.set_device(dev)
    torch.manual_seed(a.seed)  # identical head init on all ranks
    out_dir = f"{ROOT}/runs/e2e_off/{a.tag}"
    os.makedirs(out_dir, exist_ok=True)
    datasets = a.datasets.split(",") if a.datasets else None
    log = lambda *s: rank == 0 and print(*s, flush=True)

    test = build_clips(["test", "val"], datasets, False)
    if a.max_test:
        test = [c for d in sorted({c["dataset"] for c in test}) for c in sorted([c for c in test if c["dataset"] == d], key=lambda c: c["T"])[:a.max_test]]
    head = hb.BenchModel(1024, len(TOK), True, True, causal=False, n_kin=9 if a.road else 4, tok_pos=a.tok_pos).to(dev)
    backbone = Backbone(a.backbone, a.adapt, a.first_trainable, rope=a.rope, grad_ckpt=a.grad_ckpt).to(dev)
    enc = Encoder(backbone, TOK, a.dense).to(dev)
    results = {"tag": a.tag, "args": vars(a)}

    def run_eval(name, e):
        evaluate(head, test, dev, rank, world, f"{out_dir}/pred_{name}", e)
        dist.barrier()
        if rank == 0:
            results[name] = summarize(test, f"{out_dir}/pred_{name}")
            report(results[name], name)
            json.dump(results, open(f"{out_dir}/result.json", "w"), indent=1)
        dist.barrier()

    if a.eval_frozen:
        head.load_state_dict(torch.load(a.init_head, map_location=dev))
        run_eval("frozen", None)
        dist.destroy_process_group()
        return

    if a.eval_only:
        ck = torch.load(a.eval_only, map_location=dev)
        backbone.load_state_dict(ck["backbone"], strict=False); head.load_state_dict(ck["head"])
        if a.dense and ck.get("dense"):
            enc.dense.load_state_dict(ck["dense"])
        run_eval("e2e", enc)
        dist.destroy_process_group()
        return

    train = build_clips(["train"], datasets, a.ctx, a.dense)
    if a.exclude_val:  # the same groups as the benchmark's validation split
        from egokin.data.scope import filter_index
        bs_ = _load("build_segments")
        idx = filter_index(m1.build_index(["hdd", "bdd", "av2", "nuscenes", "road_waymo", "nuplan", "pai"]))
        val = bs_.val_groups(idx[idx.split.isin(["train", "val", "test"])])
        n0 = len(train)
        train = [c for c in train if (c["dataset"], bs_.val_group(c["dataset"], c["id"])) not in val]
        log(f"validation groups excluded: {n0 - len(train)} of {n0} training clips")
    dense_val = [c for c in build_clips(["val", "test"], datasets, False, True) if "dense_f" in c] if a.dense else []
    log(f"train clips {len(train)} ({sum(c['T'] for c in train) / 18000:.0f} h), test clips {len(test)}, world {world}" +
        (f", dense train clips {sum('dense_f' in c for c in train)}, dense val clips {len(dense_val)}" if a.dense else ""))
    hd = torch.nn.parallel.DistributedDataParallel(head, device_ids=[dev.index])

    # phase 1: head on cached frozen features
    if a.init_head:
        head.load_state_dict(torch.load(a.init_head, map_location=dev))
        log(f"phase 1 skipped, head from {a.init_head}")
    elif a.warm_steps:
        dl = torch.utils.data.DataLoader(Windows(train, a.L, "feats", seed=1000 + rank + 10000 * a.seed), batch_size=32 // world,
                                         num_workers=a.workers, prefetch_factor=4)
        opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=a.warm_steps, pct_start=0.05)
        t0, it = time.time(), iter(dl)
        for step in range(a.warm_steps):
            b = next(it)
            loss = loss_fn(hd(b["x"].to(dev, non_blocking=True), b["valid"].to(dev)), b, dev, a)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step(); sched.step()
            if (step + 1) % 1000 == 0:
                log(f"warm step {step + 1} loss {loss.item():.3f} ({time.time() - t0:.0f}s)")
        del it, dl
        if rank == 0:
            torch.save(head.state_dict(), f"{out_dir}/head_frozen.pt")
        run_eval("frozen", None)

    if not a.steps:
        dist.destroy_process_group()
        return

    # phase 2: backbone (LoRA or full) + head end to end
    bb_groups = backbone.param_groups(a.lr_lora if a.adapt == "lora" else a.lr_full, a.lr_decay)
    bb_params = [p for g in bb_groups for p in g["params"]]
    log(f"phase 2: {a.backbone} {a.adapt}, {sum(p.numel() for p in bb_params) / 1e6:.1f}M trainable backbone parameters")
    enc.train()  # from_pretrained returns eval mode; HF only checkpoints activations in train mode (no dropout either way)
    bb = torch.nn.parallel.DistributedDataParallel(enc, device_ids=[dev.index])
    dl = torch.utils.data.DataLoader(Windows(train, a.L, "video", seed=rank + 10000 * a.seed, dense_frac=a.dense_frac if a.dense else 0.0),
                                     batch_size=a.bs, num_workers=a.workers, persistent_workers=True, prefetch_factor=4)
    extra = list(enc.dense.parameters()) if a.dense else []
    groups = bb_groups + [{"params": list(head.parameters()) + extra, "lr": a.lr, "weight_decay": 0.05}]
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=a.steps, pct_start=max(0.05, 3 / a.steps))
    t0, it = time.time(), iter(dl)
    for step in range(a.steps):
        b = next(it)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            g, d = encode(bb, b["frames"].to(dev, non_blocking=True))
            out = hd(g.float(), b["valid"].to(dev))
        loss = loss_fn(out, b, dev, a)
        if a.dense:
            loss = loss + a.dense_w * dense_loss(d, b, dev)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bb_params + list(head.parameters()) + extra, 1.0)
        opt.step(); sched.step()
        if (step + 1) % 100 == 0:
            log(f"step {step + 1} loss {loss.item():.3f} ({time.time() - t0:.0f}s)")
        if rank == 0 and ((step + 1) % 2000 == 0 or step + 1 == a.steps):
            torch.save({"backbone": backbone.trainable_state(),
                        "head": head.state_dict(), "dense": enc.dense.state_dict() if a.dense else None, "step": step + 1},
                       f"{out_dir}/ckpt.pt")
    del it, dl
    if a.dense and rank == 0:
        results["dense_val"] = evaluate_dense(enc, dense_val, dev)
        log("[dense val]", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in results["dense_val"].items() if k != "AP"},
            {k: round(v, 3) for k, v in results["dense_val"].get("AP", {}).items()})
    run_eval("e2e", enc)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
