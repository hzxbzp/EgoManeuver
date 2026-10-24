"""M2 end-to-end, online (HDD official protocol): LoRA-tuned V-JEPA 2 causal features for the last K steps
of each window + cached frozen causal features for the earlier L-K steps (long-term context), followed by
the causal temporal model with the HDD head (+ our A/B heads). Test-time inference uses exactly the same
composition (blocks of K steps), so training and inference match.

torchrun --nproc_per_node 4 model/train_e2e.py --tag e2e_v1 [--steps 10000]
"""

import argparse
import datetime
import importlib.util
from egokin.scripts import script_path  # noqa: E402
import json
import os
import time
from types import SimpleNamespace

import av
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from egokin.data.scope import official_keep, scope
from egokin.models.vjepa_lora import load_vjepa_lora, pooled_last_tubelet, prep


def _load(name):
    spec = importlib.util.spec_from_file_location(name, script_path(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m1 = _load("train_m1")
hb = _load("train_hdd_bench")
IGNORE, GOAL, ROOT = m1.IGNORE, hb.GOAL, m1.ROOT
FEATS = "vjepa2_causal"  # cached-feature dir; main() appends the RoPE suffix (vjepa_rope.feats_name)
RARE = [4, 5, 6, 7, 8, 9, 10, 11]


def decode(path, f0, f1):
    """Frames f0..f1 (inclusive, 10 fps clip video) as uint8 (n, H, W, 3); indices < 0 repeat frame 0."""
    want = list(range(max(0, f0), f1 + 1))
    out = {}
    with av.open(path) as c:
        st = c.streams.video[0]
        c.seek(int(max(0, f0) / 10.0 / st.time_base), stream=st, backward=True, any_frame=False)
        for fr in c.decode(st):
            k = int(round(float(fr.pts * st.time_base) * 10))
            if k > f1:
                break
            if k >= max(0, f0):
                out[k] = fr.to_ndarray(format="rgb24")
    if not out:  # seek landed past the wanted frames: decode from the start
        with av.open(path) as c:
            for k, fr in enumerate(c.decode(video=0)):
                if k > f1:
                    break
                if k >= max(0, f0):
                    out[k] = fr.to_ndarray(format="rgb24")
    last = None
    frames = []
    for k in want:
        last = out.get(k, last)
        frames.append(last)
    first = next(x for x in frames if x is not None)
    frames = [first if x is None else x for x in frames]
    pad = [frames[0]] * max(0, -f0)
    return np.stack(pad + frames)


class Windows(torch.utils.data.Dataset):
    def __init__(self, clips, L, K, balance, length=10 ** 7, seed=0, hdd_frac=0.0):
        self.clips, self.L, self.K, self.balance, self.length, self.seed = clips, L, K, balance, length, seed
        self.w = np.array([c["T"] for c in clips], float)
        hdd = np.array([c["dataset"] == "hdd" for c in clips])
        if 0 < hdd_frac < 1 and hdd.any() and (~hdd).any():  # fixed HDD share, length-proportional within groups
            self.w = np.where(hdd, hdd_frac * self.w / self.w[hdd].sum(), (1 - hdd_frac) * self.w / self.w[~hdd].sum())
        self.w /= self.w.sum()
        self.rare = [np.flatnonzero(np.isin(c["y_hdd"], RARE)) for c in clips]

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i))
        ci = rng.choice(len(self.clips), p=self.w)
        c = self.clips[ci]
        T, L, K = c["T"], self.L, self.K
        te = int(rng.integers(L - 1, T)) if T >= L else T - 1
        if self.balance > 0 and len(self.rare[ci]) and rng.random() < self.balance:
            te = int(np.clip(rng.choice(self.rare[ci]) + rng.integers(0, K), L - 1 if T >= L else 0, T - 1))
        ts = np.arange(te - L + 1, te + 1)
        valid = ts >= 0
        tsc = np.clip(ts, 0, T - 1)
        feats = np.load(c["feats"], mmap_mode="r")
        feat_ctx = np.asarray(feats[tsc[: L - K]], np.float32)
        frames = decode(c["video"], 2 * (te - K + 1) + 1 - 15, 2 * te + 1)  # 2K + 14 frames
        lab = lambda y: np.where(valid, y[tsc], IGNORE)
        kin = np.where(valid[:, None], c["kin"][tsc], np.nan).astype(np.float32)
        y_ctx = np.where(valid[:, None], c["ctx"][tsc], np.nan).astype(np.float32)
        return dict(feat_ctx=torch.from_numpy(feat_ctx), frames=torch.from_numpy(frames), y_hdd=torch.from_numpy(lab(c["y_hdd"])),
                    kin=torch.from_numpy(kin), y_ctx=torch.from_numpy(y_ctx),
                    y_a=torch.from_numpy(lab(c["y_a"])), y_b=torch.from_numpy(lab(c["y_b"].clip(min=IGNORE))))


class Encoder(torch.nn.Module):
    """windows (N, 16, 3, 224, 448) -> pooled last-tubelet tokens (N, 9, 1024); wrapped in DDP for training."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, wins):
        return pooled_last_tubelet(self.m, wins)


def fresh_features(enc, frames, K):
    """frames: (B, 2K+14, H, W, 3) uint8 -> (B, K, 9, 1024): causal windows ending at frames 15, 17, ..."""
    B = frames.shape[0]
    x = prep(frames.flatten(0, 1)).view(B, frames.shape[1], 3, 224, 448)
    wins = torch.stack([x[:, 2 * k: 2 * k + 16] for k in range(K)], 1).flatten(0, 1)  # (B*K, 16, 3, 224, 448)
    return enc(wins).view(B, K, 9, -1)


def build_clips(split, datasets=("hdd",)):
    idx = m1.build_index(list(datasets))
    idx = idx[(idx.split == split) & idx.apply(lambda r: os.path.exists(f"{ROOT}/cache/feats/{FEATS}/{r.dataset}/{r.clip_id}.npy"), axis=1)]
    clips = []
    for r in idx.itertuples():
        f, ya, yb, k = m1.load_clip(r, FEATS)
        clips.append(dict(id=r.clip_id, dataset=r.dataset, dir=r.dir, T=len(f), feats=f"{ROOT}/cache/feats/{FEATS}/{r.dataset}/{r.clip_id}.npy",
                          video=os.path.join(r.dir, "video.mp4"),
                          y_hdd=hb.hdd_targets(r, len(f)) if r.dataset == "hdd" else np.full(len(f), IGNORE),
                          y_a=ya, y_b=yb, kin=k[:, :4].astype(np.float32), ctx=hb.ctx_targets(r, len(f))))
    return clips


def wmean(l, m):
    """Weighted mean that is 0 (and still connected to the graph, for DDP) when nothing is labeled."""
    return (l * m).sum() / m.sum().clamp(min=1e-6)


def losses(out, b, pw, dev, a):
    """out = (la, lb, kp, lh, lc) over L steps; pw: (L,) per-position weight."""
    la, lb, kp, lh, lc = [o.float() for o in out]
    ce = lambda lg, y: wmean(F.cross_entropy(lg.transpose(1, 2), y, ignore_index=IGNORE, reduction="none"),
                             (y != IGNORE).float() * pw)
    y_hdd, y_a, y_b = b["y_hdd"].to(dev), b["y_a"].to(dev), b["y_b"].to(dev)
    loss = ce(lh, y_hdd) + a.ab_w * (ce(la, y_a) + ce(lb, y_b))
    kt = b["kin"].to(dev)
    loss = loss + 0.5 * wmean(F.smooth_l1_loss(kp, torch.nan_to_num(kt), reduction="none"), (~torch.isnan(kt)).float() * pw[:, None])
    yc = b["y_ctx"].to(dev)
    lctx = wmean(F.binary_cross_entropy_with_logits(lc, torch.nan_to_num(yc), reduction="none"), (~torch.isnan(yc)).float() * pw[:, None])
    return loss + (0.5 if a.ctx else 0.0) * lctx


@torch.no_grad()
def causal_features_stream(backbone, video, T, dev, batch=64):
    """Fine-tuned causal features for every 5 Hz step, decoding the video in 512-frame chunks with a
    14-frame history (memory independent of the session length)."""
    cf = _load("cache_features")
    G = np.zeros((T, 9, 1024), np.float32)
    hist, base = None, 0
    for frames in cf.frame_chunks(video, 512):
        allf = frames if hist is None else np.concatenate([hist, frames])
        off = 0 if hist is None else len(hist)
        t0, n = base // 2, (len(frames) + 1) // 2
        ends = np.minimum(off + 2 * np.arange(n) + 1, len(allf) - 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = prep(torch.from_numpy(allf).to(dev))
            for i in range(0, n, batch):
                idx = torch.stack([torch.arange(e - 15, e + 1) for e in ends[i:i + batch]]).clamp(min=0).to(dev)
                g = pooled_last_tubelet(backbone, x[idx]).float().cpu().numpy()
                lo = t0 + i
                G[lo:min(T, lo + len(g))] = g[: max(0, min(T, lo + len(g)) - lo)]
        hist = allf[-14:]
        base += len(frames)
    return G


@torch.no_grad()
def evaluate(backbone, head, clips, L, K, dev, rank, world, out_dir):
    """Per-session predictions at the official frames; sessions are sharded across ranks."""
    backbone.eval(); head.eval()
    for c in sorted(clips, key=lambda c: -c["T"])[rank::world]:
        T = c["T"]
        cached = np.load(c["feats"], mmap_mode="r")
        G = causal_features_stream(backbone, c["video"], T, dev)
        prob = np.zeros((T, len(GOAL)), np.float32)
        for s in range(0, T, K):
            e = min(T, s + K)
            ctx_ids = np.clip(np.arange(e - L, s), 0, T - 1)
            x = np.concatenate([np.asarray(cached[ctx_ids], np.float32), G[s:e]])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lh = head(torch.from_numpy(x).unsqueeze(0).to(dev))[3]
            prob[s:e] = lh.float().softmax(-1)[0, -(e - s):].cpu().numpy()
        z = np.load(os.path.join(c["dir"], "labels.npz"))
        step = np.clip(np.round((z["official_t"] - 0.1) / 0.2).astype(int), 0, T - 1)
        np.savez(os.path.join(out_dir, f"{c['id']}.npz"), p=prob[step], y=z["official_goal"].astype(int), keep=official_keep(z, SimpleNamespace(dataset=c["dataset"], split="test")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--bs", type=int, default=4, help="per GPU")
    ap.add_argument("--first-trainable", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--balance", type=float, default=0.0, help="rare-class window oversampling (hurt frozen runs)")
    ap.add_argument("--init-head", default="", help="warm start from a frozen-feature BenchModel state_dict (.pt)")
    ap.add_argument("--early-w", type=float, default=1.0, help="loss weight of the L-K context positions (last K = 1)")
    ap.add_argument("--ab-w", type=float, default=0.5, help="weight of our axis A/B losses")
    ap.add_argument("--hdd-frac", type=float, default=0.0, help="share of HDD windows when mixing datasets (0: by length)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--eval-only", default="")
    ap.add_argument("--ctx", action="store_true", help="scene-context head (needs HSD etc. causal features)")
    ap.add_argument("--datasets", default="hdd", help="training datasets with cached causal features")
    ap.add_argument("--max-test", type=int, default=0, help="smoke test: evaluate only the first N test sessions")
    ap.add_argument("--rope", default="interp", choices=["hf", "int", "interp"],
                    help="V-JEPA 2 RoPE positions; also selects the matching cached-feature directory")
    a = ap.parse_args()
    global FEATS
    from egokin.models.vjepa_rope import feats_name
    FEATS = feats_name(FEATS, a.rope)

    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=3))  # eval shards are unbalanced
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    torch.cuda.set_device(dev)
    torch.manual_seed(rank)
    out_dir = f"{ROOT}/runs/e2e/{a.tag}"
    os.makedirs(f"{out_dir}/pred", exist_ok=True)

    backbone = load_vjepa_lora(a.first_trainable, rope=a.rope).to(dev)
    head = hb.BenchModel(1024, 9, True, True, causal=True).to(dev)
    if a.eval_only:
        ck = torch.load(a.eval_only, map_location=dev)
        backbone.load_state_dict(ck["backbone"], strict=False); head.load_state_dict(ck["head"])
    else:
        if a.init_head:
            head.load_state_dict(torch.load(a.init_head, map_location=dev))
        train = build_clips("train", a.datasets.split(","))
        if rank == 0:
            print(f"train sessions {len(train)}, steps {a.steps}, world {world}, init_head {a.init_head or '-'}", flush=True)
        ds = Windows(train, a.L, a.K, a.balance, seed=rank, hdd_frac=a.hdd_frac)
        pw = torch.full((a.L,), a.early_w, device=dev)
        pw[a.L - a.K:] = 1.0
        dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, num_workers=a.workers, persistent_workers=True, prefetch_factor=4)
        lora = [p for p in backbone.parameters() if p.requires_grad]
        bb = torch.nn.parallel.DistributedDataParallel(Encoder(backbone), device_ids=[dev.index])
        hd = torch.nn.parallel.DistributedDataParallel(head, device_ids=[dev.index])
        opt = torch.optim.AdamW([{"params": lora, "lr": a.lr_lora}, {"params": head.parameters(), "lr": a.lr}], weight_decay=0.05)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, [a.lr_lora, a.lr], total_steps=a.steps, pct_start=max(0.05, 3 / a.steps))
        t0 = time.time()
        it = iter(dl)
        for step in range(a.steps):
            b = next(it)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fresh = fresh_features(bb, b["frames"].to(dev, non_blocking=True), a.K)
                x = torch.cat([b["feat_ctx"].to(dev), fresh.float()], 1)
                out = hd(x)
            loss = losses(out, b, pw, dev, a)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora + list(head.parameters()), 1.0)
            opt.step(); sched.step()
            if rank == 0 and (step + 1) % 100 == 0:
                print(f"step {step + 1} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
            if rank == 0 and (step + 1) % 2000 == 0:
                torch.save({"backbone": {k: v for k, v in backbone.state_dict().items() if ".a" in k[-2:] or ".b" in k[-2:]},
                            "head": head.state_dict(), "step": step + 1}, f"{out_dir}/ckpt.pt")
        if rank == 0:
            torch.save({"backbone": {k: v for k, v in backbone.state_dict().items() if k.endswith(".a") or k.endswith(".b")},
                        "head": head.state_dict(), "step": a.steps}, f"{out_dir}/ckpt.pt")
    dist.barrier()
    test = build_clips("test")
    if a.max_test:
        test = sorted(test, key=lambda c: c["T"])[: a.max_test]
    evaluate(backbone, head, test, a.L, a.K, dev, rank, world, f"{out_dir}/pred")
    dist.barrier()
    if rank == 0:
        P, Y, K = [], [], []
        for c in test:
            z = np.load(f"{out_dir}/pred/{c['id']}.npz"); P.append(z["p"]); Y.append(z["y"]); K.append(z["keep"])
        P, Y, K = np.concatenate(P), np.concatenate(Y), np.concatenate(K)
        ap_c = {GOAL[c]: float(average_precision_score(Y == c, P[:, c])) for c in range(1, len(GOAL))}
        ap_s = {GOAL[c]: float(average_precision_score(Y[K] == c, P[K, c])) for c in range(1, len(GOAL))}
        res = {"tag": a.tag, "args": vars(a), "scope": scope(), "mAP": float(np.mean(list(ap_c.values()))), "AP": ap_c,
               "mAP_scope": float(np.mean(list(ap_s.values()))), "AP_scope": ap_s}
        print(f"[HDD official online, E2E] mAP {res['mAP'] * 100:.1f}", {k: round(v * 100, 1) for k, v in ap_c.items()}, flush=True)
        json.dump(res, open(f"{out_dir}/result.json", "w"), indent=1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
