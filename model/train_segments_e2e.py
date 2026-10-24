"""Control for the baseline's dense supervision: fine-tune the same encoder end to end on SEGMENT labels only, the way the
existing video models are trained (model/train_segments_video.py), with everything else identical to our baseline:
V-JEPA 2-L with interpolated RoPE, 224x448 input at 10 fps (16-frame windows aligned to the clip start), the 4x8 grid
readout (grid33 + slot embeddings), the segment head of model/train_segments.py, class-balanced segment sampling,
set-valued loss for partial labels, full fine-tuning with layer-wise LR decay 0.8 (top block 5e-5, head 1e-4),
OneCycle, bf16, 10k steps.

The head starts from the segment head trained on frozen grid33 features (v2_frozen_grid33_s<seed>_head.pt), the
counterpart of stage 1's head warm-up. Saves ckpt.pt (backbone + head) and result.json in runs/e2e_off/<tag>, the layout
model/cache_finetuned.py reads, so stage 2 (cached features -> segment heads, 3 seeds) is the same as for our baseline.

torchrun --nproc_per_node 2 model/train_segments_e2e.py --tag segonly_full_s0 [--steps 10000] [--bs 4] [--seed 0]
"""

import argparse
import datetime
import importlib.util
from egokin.scripts import script_path  # noqa: E402
import json
import os
import time

import numpy as np
import torch
import torch.distributed as dist

from egokin.models.backbones import Backbone
from egokin.models.vjepa_lora import prep


def _load(name):
    spec = importlib.util.spec_from_file_location(name, script_path(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


seg = _load("train_segments")
e2e = _load("train_e2e")
e2o = _load("train_e2e_offline")
ROOT, HZ = seg.ROOT, seg.HZ


class SegVideos(torch.utils.data.Dataset):
    """Class-balanced random training segments as decoded frames of whole 16-frame windows aligned to the clip start
    (as the cached features), plus the offset and length of the segment's steps inside them."""

    def __init__(self, d, max_steps, balance, seed, length=10 ** 8):
        self.d, self.max_steps, self.seed, self.length = d, max_steps, seed, length
        freq = d.lat.map(d.lat.value_counts())
        w = (freq ** -balance).to_numpy(dtype=np.float64, copy=True)
        self.w = w / w.sum()
        self.al = seg.target_sets(d.lat.tolist(), seg.LAT)
        self.ao = seg.target_sets(d.long.tolist(), seg.LONG)
        self.dirs = {}

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i))
        j = int(rng.choice(len(self.d), p=self.w))
        r = self.d.iloc[j]
        s = int(np.floor(r.start * HZ))
        e = min(int(np.ceil(r.end * HZ)), s + self.max_steps)
        a0 = 8 * (s // 8)
        nw = -(-(e - a0) // 8)
        video = os.path.join(ROOT, "clips", r.dataset, r.clip_id, "video.mp4")
        frames = e2e.decode(video, 2 * a0, 2 * (a0 + 8 * nw) - 1)  # past the end: last frame repeated
        return dict(frames=torch.from_numpy(np.ascontiguousarray(frames)), off=s - a0, n=e - s,
                    al=torch.from_numpy(self.al[j]), ao=torch.from_numpy(self.ao[j]))


def collate(items):
    F = max(x["frames"].shape[0] for x in items)
    frames = torch.stack([torch.cat([x["frames"], x["frames"][-1:].expand(F - x["frames"].shape[0], *x["frames"].shape[1:])])
                          if x["frames"].shape[0] < F else x["frames"] for x in items])
    return dict(frames=frames, off=torch.tensor([x["off"] for x in items]), n=torch.tensor([x["n"] for x in items]),
                al=torch.stack([x["al"] for x in items]), ao=torch.stack([x["ao"] for x in items]))


def forward(enc, head, b, dev):
    frames = b["frames"].to(dev, non_blocking=True)
    B, N = frames.shape[:2]
    x = prep(frames.flatten(0, 1)).view(B * N // 16, 16, 3, 224, 448)
    g, _ = enc(x)  # (B * N / 16, 8, n_tok, D)
    g = g.reshape(B, (N // 16) * 8, *g.shape[2:])
    L = int(b["n"].max())
    xs = torch.zeros(B, L, *g.shape[2:], device=dev, dtype=g.dtype)
    valid = torch.zeros(B, L, dtype=torch.bool, device=dev)
    for k in range(B):
        o, n = int(b["off"][k]), int(b["n"][k])
        xs[k, :n] = g[k, o:o + n]
        valid[k, :n] = True
    return head(xs.float(), valid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--segments", default="segments_v2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=4, help="segments per GPU")
    ap.add_argument("--max-steps", type=int, default=128)
    ap.add_argument("--balance", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-4, help="head LR")
    ap.add_argument("--lr-full", type=float, default=5e-5, help="top-block LR")
    ap.add_argument("--lr-decay", type=float, default=0.8)
    ap.add_argument("--init-head", default="", help="default: runs/segments/v2_frozen_grid33_s<seed>_head.pt")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=3))
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    torch.cuda.set_device(dev)
    torch.manual_seed(a.seed)
    log = lambda *s: rank == 0 and print(*s, flush=True)
    out_dir = f"{ROOT}/runs/e2e_off/{a.tag}"
    os.makedirs(out_dir, exist_ok=True)

    d = seg.load_segments(None, a.segments)
    tr = d[d.part == "train"].reset_index(drop=True)
    log(f"train {len(tr)} segments ({(tr.end - tr.start).sum() / 3600:.1f} h), world {world}")
    tok = e2o.m1.RICH_TOKENS["grid33"]
    head = seg.SegHead(len(tok), max_steps=a.max_steps).to(dev)
    init = a.init_head or f"{ROOT}/runs/segments/v2_frozen_grid33_s{a.seed}_head.pt"
    head.load_state_dict(torch.load(init, map_location=dev))
    log(f"head from {init}")
    backbone = Backbone("vjepa2", "full", 12, rope="interp", grad_ckpt=True).to(dev)
    enc = e2o.Encoder(backbone, tok).to(dev)
    enc.train()
    bb = torch.nn.parallel.DistributedDataParallel(enc, device_ids=[dev.index])
    hd = torch.nn.parallel.DistributedDataParallel(head, device_ids=[dev.index])
    bb_groups = backbone.param_groups(a.lr_full, a.lr_decay)
    bb_params = [p for g in bb_groups for p in g["params"]]
    groups = bb_groups + [{"params": list(head.parameters()), "lr": a.lr, "weight_decay": 0.05}]
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=a.steps,
                                                pct_start=max(0.05, 3 / a.steps))
    dl = torch.utils.data.DataLoader(SegVideos(tr, a.max_steps, a.balance, seed=rank + 10000 * a.seed), batch_size=a.bs,
                                     num_workers=a.workers, persistent_workers=True, prefetch_factor=4, collate_fn=collate)
    # result.json first: model/cache_finetuned.py reads the backbone configuration from it
    args = dict(vars(a), backbone="vjepa2", adapt="full", first_trainable=12, rope="interp", tokens="grid33", L=a.max_steps)
    if rank == 0:
        json.dump({"tag": a.tag, "args": args, "supervision": "segment labels only (control)"},
                  open(f"{out_dir}/result.json", "w"), indent=1)
    t0, it = time.time(), iter(dl)
    for step in range(a.steps):
        b = next(it)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            la, lo = forward(bb, hd, b, dev)
        loss = seg.set_loss(la, b["al"].to(dev)) + seg.set_loss(lo, b["ao"].to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bb_params + list(head.parameters()), 1.0)
        opt.step(); sched.step()
        if (step + 1) % 100 == 0:
            log(f"step {step + 1} loss {loss.item():.3f} ({time.time() - t0:.0f}s)")
        if rank == 0 and ((step + 1) % 2000 == 0 or step + 1 == a.steps):
            torch.save({"backbone": backbone.trainable_state(), "head": head.state_dict(), "step": step + 1},
                       f"{out_dir}/ckpt.pt")
    log(f"done {a.steps} steps ({time.time() - t0:.0f}s)")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
