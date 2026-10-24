"""Existing video models on the segment-level benchmark: each model is fine-tuned end to end on the
training segments (Kinetics-pretrained weights) with two heads (lateral 17, longitudinal 8) and evaluated exactly like
model/train_segments.py (same segments, partial labels, metrics).

Models: tsn_r50 (ImageNet ResNet-50 per frame, averaged), i3d_r50, slowfast_r50, x3d_m (pytorchvideo, Kinetics-400),
videomae_b (MCG-NJU/videomae-base-finetuned-kinetics), timesformer_b (facebook/timesformer-base-finetuned-k400).
Input: T frames sampled uniformly over the segment (jittered in training), squashed to S x S; no flips (they would
swap left / right classes).

Usage: torchrun --nproc_per_node 2 model/train_segments_video.py --model videomae_b --tag seg_videomae [--seed 0]
"""

import argparse
import datetime
import importlib.util
from egokin.scripts import script_path  # noqa: E402
import json
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _load(name):
    spec = importlib.util.spec_from_file_location(name, script_path(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


seg = _load("train_segments")
e2e = _load("train_e2e")
ROOT = seg.ROOT
K_MEAN, K_STD = [0.45, 0.45, 0.45], [0.225, 0.225, 0.225]
IN_MEAN, IN_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
# frames, side, normalization
CFG = {"tsn_r50": (8, 224, IN_MEAN, IN_STD), "i3d_r50": (8, 224, K_MEAN, K_STD), "slowfast_r50": (32, 256, K_MEAN, K_STD),
       "x3d_m": (16, 224, K_MEAN, K_STD), "videomae_b": (16, 224, IN_MEAN, IN_STD), "timesformer_b": (8, 224, IN_MEAN, IN_STD)}


class Backbone(nn.Module):
    """Pretrained video network without its classifier: (B, T, 3, S, S) -> (B, dim)."""

    def __init__(self, name):
        super().__init__()
        self.name = name
        if name == "tsn_r50":
            import timm
            self.net, self.dim = timm.create_model("resnet50", pretrained=True, num_classes=0), 2048
        elif name in ("i3d_r50", "slowfast_r50", "x3d_m"):
            from pytorchvideo.models import hub
            self.net = getattr(hub, name)(pretrained=True)
            head = self.net.blocks[-1]
            self.dim = head.proj.in_features
            head.proj = nn.Identity()
            if hasattr(head, "activation"):
                head.activation = None
        elif name == "videomae_b":
            from transformers import VideoMAEModel
            self.net, self.dim = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics"), 768
        elif name == "timesformer_b":
            from transformers import TimesformerModel
            self.net, self.dim = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400"), 768
        else:
            raise ValueError(name)

    def forward(self, x):
        B, T = x.shape[:2]
        if self.name == "tsn_r50":
            return self.net(x.flatten(0, 1)).view(B, T, -1).mean(1)
        if self.name == "slowfast_r50":
            v = x.permute(0, 2, 1, 3, 4)  # (B, 3, T, S, S)
            return self.net([v[:, :, ::4], v]).flatten(1)
        if self.name in ("i3d_r50", "x3d_m"):
            return self.net(x.permute(0, 2, 1, 3, 4)).flatten(1)
        if self.name == "videomae_b":
            return self.net(pixel_values=x).last_hidden_state.mean(1)
        return self.net(pixel_values=x).last_hidden_state[:, 0]


class Model(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.backbone = Backbone(name)
        self.norm = nn.LayerNorm(self.backbone.dim)
        self.lat, self.long = nn.Linear(self.backbone.dim, len(seg.LAT)), nn.Linear(self.backbone.dim, len(seg.LONG))

    def forward(self, x):
        h = self.norm(self.backbone(x).float())
        return self.lat(h), self.long(h)


class Segments(torch.utils.data.Dataset):
    def __init__(self, d, n_frames, side, mean, std, train, seed=0, length=None):
        self.d, self.T, self.side, self.train, self.seed = d.reset_index(drop=True), n_frames, side, train, seed
        self.mean, self.std = torch.tensor(mean).view(1, 3, 1, 1), torch.tensor(std).view(1, 3, 1, 1)
        self.dirs = {}
        self.length = length or len(self.d)

    def __len__(self):
        return self.length

    def frames(self, r, rng):
        f0, f1 = int(round(r.start * 10)), max(int(round(r.start * 10)) + 1, int(round(r.end * 10)) - 1)
        if self.train:  # jittered uniform sampling
            edges = np.linspace(f0, f1 + 1, self.T + 1)
            idx = np.floor(edges[:-1] + rng.random(self.T) * np.diff(edges)).astype(int)
        else:
            idx = np.round(np.linspace(f0, f1, self.T)).astype(int)
        idx = np.clip(idx, f0, f1)
        clip = e2e.decode(os.path.join(r.dir, "video.mp4"), f0, f1)  # (n, H, W, 3) uint8
        x = torch.from_numpy(clip[np.clip(idx - f0, 0, len(clip) - 1)]).permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=(self.side, self.side), mode="bilinear", align_corners=False, antialias=True)
        return (x - self.mean) / self.std

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i))
        j = i if not self.train else int(rng.choice(len(self.d), p=self.w))
        r = self.d.iloc[j]
        return self.frames(r, rng), j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(CFG))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=16, help="per GPU")
    ap.add_argument("--lr", type=float, default=1e-4, help="backbone; heads x10")
    ap.add_argument("--balance", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-test", type=int, default=0, help="smoke test: first N test segments")
    ap.add_argument("--segments", default="segments_v1", help="benchmark version ($VEGO_DATA/benchmark/<name>.csv)")
    a = ap.parse_args()
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=3))
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    torch.cuda.set_device(dev)
    torch.manual_seed(a.seed)
    import pandas as pd
    from egokin.data.index import build_index

    d = seg.load_segments(None, a.segments)  # kept segments with their part (train / val / test)
    idx = build_index(sorted(d.dataset.unique()))
    d = d.merge(idx[["dataset", "clip_id", "dir"]], on=["dataset", "clip_id"])
    tr, te = d[d.part == "train"].reset_index(drop=True), d[d.part == "test"].reset_index(drop=True)
    va = d[d.part == "val"].reset_index(drop=True)
    if a.max_test:
        te = te.groupby("dataset").head(a.max_test).reset_index(drop=True)
        va = va.groupby("dataset").head(a.max_test).reset_index(drop=True)
    n_te = len(te)
    te = pd.concat([te, va], ignore_index=True)  # one evaluation pass over test then val
    T, S, mean, std = CFG[a.model]
    ds = Segments(tr, T, S, mean, std, True, seed=a.seed * 1000 + rank, length=a.steps * a.bs)
    freq = tr.lat.map(tr.lat.value_counts())
    w = (freq ** -a.balance).to_numpy(dtype=np.float64, copy=True)
    ds.w = w / w.sum()
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, num_workers=a.workers, prefetch_factor=4, persistent_workers=True)
    model = Model(a.model).to(dev)
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev.index])
    heads = list(model.norm.parameters()) + list(model.lat.parameters()) + list(model.long.parameters())
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": a.lr}, {"params": heads, "lr": 10 * a.lr}], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [a.lr, 10 * a.lr], total_steps=a.steps, pct_start=0.05)
    al_tr = torch.from_numpy(seg.target_sets(tr.lat.tolist(), seg.LAT)).to(dev)
    ao_tr = torch.from_numpy(seg.target_sets(tr.long.tolist(), seg.LONG)).to(dev)
    log = lambda *x: rank == 0 and print(*x, flush=True)
    log(f"{a.model}: train {len(tr)} segments, test {n_te}, val {len(va)}, world {world}, {T} frames at {S}x{S}")
    import time
    t0 = time.time()
    for step, (x, j) in enumerate(dl):
        x, j = x.to(dev, non_blocking=True), j.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            la, lo = ddp(x)
        loss = seg.set_loss(la, al_tr[j]) + seg.set_loss(lo, ao_tr[j])
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if (step + 1) % 200 == 0:
            log(f"step {step + 1} loss {loss.item():.3f} ({time.time() - t0:.0f}s)")
    # evaluation, sharded over ranks
    model.eval()
    ev = Segments(te, T, S, mean, std, False)
    my = list(range(rank, len(te), world))
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ev, my), batch_size=a.bs, num_workers=a.workers)
    PL, PO, J = [], [], []
    with torch.no_grad():
        for x, j in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                la, lo = model(x.to(dev))
            PL.append(la.float().softmax(-1).cpu()); PO.append(lo.float().softmax(-1).cpu()); J.append(j)
    out = f"{ROOT}/runs/segments"
    os.makedirs(out, exist_ok=True)
    torch.save((torch.cat(PL), torch.cat(PO), torch.cat(J)), f"{out}/{a.tag}_s{a.seed}_rank{rank}.pt")
    dist.barrier()
    if rank == 0:
        parts = [torch.load(f"{out}/{a.tag}_s{a.seed}_rank{r}.pt") for r in range(world)]
        PL, PO, J = (torch.cat([p[k] for p in parts]).numpy() for k in range(3))
        order = np.argsort(J)
        PL, PO = PL[order], PO[order]
        VL, VO, PL, PO, te = PL[n_te:], PO[n_te:], PL[:n_te], PO[:n_te], te.iloc[:n_te]
        res = {"tag": a.tag, "args": vars(a), "long_classes": seg.LONG, "test": seg.metrics(te, PL, PO)}
        if len(va):  # model selection uses the validation split only
            res["val"] = seg.metrics(va, VL, VO)
        for dsn in sorted(te.dataset.unique()):
            m = (te.dataset == dsn).to_numpy()
            res[f"{dsn}_test"] = seg.metrics(te[m], PL[m], PO[m])
        t = res["test"]
        print(f"[{a.tag} s{a.seed}] lateral mAP {t['lateral']['mAP'] * 100:.1f} acc {t['lateral']['acc'] * 100:.1f} F1 {t['lateral']['macro_f1'] * 100:.1f} | "
              f"longitudinal mAP {t['longitudinal']['mAP'] * 100:.1f} acc {t['longitudinal']['acc'] * 100:.1f}", flush=True)
        json.dump(res, open(f"{out}/{a.tag}_s{a.seed}.json", "w"), indent=1)
        np.savez_compressed(f"{out}/{a.tag}_s{a.seed}_pred.npz", lat=PL.astype(np.float16), long=PO.astype(np.float16),
                            key=seg.seg_keys(te), segments=a.segments, long_classes=np.array(seg.LONG))
        for r in range(world):
            os.remove(f"{out}/{a.tag}_s{a.seed}_rank{r}.pt")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
