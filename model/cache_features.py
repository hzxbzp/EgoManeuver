"""Cache frozen backbone features for M1 (5 Hz feature steps).

vjepa2:   V-JEPA 2 ViT-L/16, 16-frame windows (1.6 s @10 fps) at 224x448 -> 8 steps x (2x4 grid + global) x 1024
resnet50: ImageNet ResNet-50 per frame (every 2nd frame -> 5 Hz), 224x448 -> global pooled 2048 (frame-level baseline)

Usage: python model/cache_features.py --model vjepa2 --gpu 0 [--datasets hdd,av2] [--splits test] [--batch 8]
Output: $VEGO_DATA/cache/feats/<model>/<dataset>/<clip_id>.npy (float16)
"""

import argparse
import os
import time

import av
import numpy as np
import torch
import torch.nn.functional as F

from egokin.data.index import ROOT, build_index

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def frame_chunks(path, chunk=512):
    """Decode a video in chunks of `chunk` frames (a multiple of 16) so memory does not grow with
    video length (HDD sessions reach 60k frames)."""
    with av.open(path) as c:
        c.streams.video[0].thread_type = "AUTO"
        buf = []
        for f in c.decode(video=0):
            buf.append(f.to_ndarray(format="rgb24"))
            if len(buf) == chunk:
                yield np.stack(buf)
                buf = []
        if buf:
            yield np.stack(buf)


def prep(frames, dev):  # (T,H,W,3) uint8 -> (T,3,224,448) normalized half
    x = torch.from_numpy(frames).to(dev).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=(224, 448), mode="bilinear", align_corners=False, antialias=True)
    return ((x - MEAN.to(dev)) / STD.to(dev)).half()


class VJEPA2:
    def __init__(self, dev, rope="interp"):
        from transformers import AutoModel

        from egokin.models.vjepa_rope import patch_rope

        os.environ.setdefault("HF_HOME", f"{ROOT}/cache/hf")
        self.m = patch_rope(AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256", dtype=torch.float16), rope)
        self.m = self.m.to(dev).eval()
        self.dev = dev

    @torch.no_grad()
    def __call__(self, frames, batch):
        T = len(frames)
        pad = (-T) % 16
        if pad:
            frames = np.concatenate([frames, np.repeat(frames[-1:], pad, 0)])
        wins = prep(frames, self.dev).view(-1, 16, 3, 224, 448)
        out = []
        for i in range(0, len(wins), batch):
            tok = self.m.get_vision_features(wins[i:i + batch])  # (B, 8*14*28, 1024)
            B = tok.shape[0]
            tok = tok.view(B, 8, 14, 28, -1).permute(0, 1, 4, 2, 3).reshape(B * 8, -1, 14, 28)
            grid = F.adaptive_avg_pool2d(tok.float(), (2, 4)).flatten(2)  # (B*8, D, 8)
            glob = tok.float().mean((2, 3)).unsqueeze(-1)  # (B*8, D, 1)
            out.append(torch.cat([glob, grid], -1).permute(0, 2, 1).reshape(B * 8, 9, -1).half().cpu())
        feats = torch.cat(out).numpy()
        return feats[: (T + 1) // 2]  # 5 Hz steps covering the real frames


MID_LAYER = 15  # 0-based; ~65% depth of the 24-layer ViT-L (best features are mid-depth, research/literature/02)
RICH_TOKENS = {  # token layout of the vjepa2x features, (T, 74, 1024) per clip
    "pool9": list(range(0, 9)),  # last layer: global + 2x4 grid (= the vjepa2 features)
    "grid33": [0] + list(range(9, 41)),  # last layer: global + 4x8 grid
    "mid33": list(range(41, 74)),  # layer MID_LAYER: global + 4x8 grid
}


def pool_step_tokens(tok, grids):
    """tok (N, D, 14, 28) -> (N, 1 + sum(h*w), D): global mean followed by each adaptive-average grid."""
    parts = [tok.float().mean((2, 3)).unsqueeze(-1)] + [F.adaptive_avg_pool2d(tok.float(), g).flatten(2) for g in grids]
    return torch.cat(parts, -1).permute(0, 2, 1)


class VJEPA2Rich(VJEPA2):
    """Last layer: global + 2x4 + 4x8 grids (41 tokens); layer MID_LAYER: global + 4x8 (33 tokens). 74 tokens
    per 5 Hz step, for the A2 readout / layer / probe diagnostics (one forward pass)."""

    def __init__(self, dev, rope="interp"):
        super().__init__(dev, rope)
        self._mid = None
        self.m.encoder.layer[MID_LAYER].register_forward_hook(
            lambda mod, args, out: setattr(self, "_mid", out[0] if isinstance(out, tuple) else out))

    @torch.no_grad()
    def __call__(self, frames, batch):
        T = len(frames)
        pad = (-T) % 16
        if pad:
            frames = np.concatenate([frames, np.repeat(frames[-1:], pad, 0)])
        wins = prep(frames, self.dev).view(-1, 16, 3, 224, 448)
        out = []
        for i in range(0, len(wins), batch):
            last = self.m.get_vision_features(wins[i:i + batch])  # (B, 8*14*28, D), final LayerNorm applied
            B = last.shape[0]
            grid = lambda t: t.view(B, 8, 14, 28, -1).permute(0, 1, 4, 2, 3).reshape(B * 8, -1, 14, 28)
            a = pool_step_tokens(grid(last), [(2, 4), (4, 8)])
            b = pool_step_tokens(grid(self._mid), [(4, 8)])
            out.append(torch.cat([a, b], 1).half().cpu())
        return torch.cat(out).numpy()[: (T + 1) // 2]


class VJEPA21Rich(VJEPA2Rich):
    """Same 74-token layout as VJEPA2Rich, from V-JEPA 2.1 ViT-L (official code; egokin/models/vjepa21.py)."""

    def __init__(self, dev, rope=None):
        from egokin.models.vjepa21 import load_vjepa21_large

        self.dev, self._mid = dev, None
        self.enc = load_vjepa21_large(dev)  # fp32 weights, bf16 autocast in get_vision_features
        self.enc.blocks[MID_LAYER].register_forward_hook(
            lambda mod, args, out: setattr(self, "_mid", out[0] if isinstance(out, tuple) else out))
        self.m = self  # get_vision_features below

    def get_vision_features(self, wins):  # (B, 16, 3, H, W) -> (B, N, D), final normalized tokens
        from egokin.models.vjepa21 import encode

        return encode(self.enc, wins.permute(0, 2, 1, 3, 4))


class VJEPA2Causal(VJEPA2):
    """Strictly online features: step t (frames 2t, 2t+1) uses only the 16 frames ending at frame 2t+1;
    the last tubelet token of that window is kept. 8x the compute of the non-overlapping windows."""

    @torch.no_grad()
    def __call__(self, frames, batch, history=None):
        # history: the previous 14 frames (uint8) so windows at the start of a chunk see the past
        allf = frames if history is None else np.concatenate([history, frames])
        off = 0 if history is None else len(history)
        if len(allf) < 16:
            allf = np.concatenate([np.repeat(allf[:1], 16 - len(allf), 0), allf])
            off += 16 - len(np.concatenate([history, frames]) if history is not None else frames)
        x = prep(allf, self.dev)  # (N,3,224,448)
        ends = [off + 2 * t + 1 for t in range((len(frames) + 1) // 2)]
        ends = [min(e, len(allf) - 1) for e in ends]
        out = []
        for i in range(0, len(ends), batch):
            idx = torch.stack([torch.arange(e - 15, e + 1) for e in ends[i:i + batch]]).clamp(min=0).to(self.dev)
            tok = self.m.get_vision_features(x[idx])  # (B, 8*14*28, D)
            B = tok.shape[0]
            last = tok.view(B, 8, 14, 28, -1)[:, -1].permute(0, 3, 1, 2)  # (B, D, 14, 28)
            grid = F.adaptive_avg_pool2d(last.float(), (2, 4)).flatten(2)
            glob = last.float().mean((2, 3)).unsqueeze(-1)
            out.append(torch.cat([glob, grid], -1).permute(0, 2, 1).half().cpu())
        return torch.cat(out).numpy()


class ResNet50:
    def __init__(self, dev):
        import torchvision

        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = torch.nn.Identity()
        self.m = m.to(dev).half().eval()
        self.dev = dev

    @torch.no_grad()
    def __call__(self, frames, batch):
        frames = frames[1::2]  # 5 Hz, centered in each 0.2 s step
        if len(frames) == 0:  # a trailing chunk with a single frame
            return np.zeros((0, 2048), np.float16)
        out = [self.m(prep(frames[i:i + batch * 8], self.dev)).cpu() for i in range(0, len(frames), batch * 8)]
        return torch.cat(out).numpy().astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["vjepa2", "vjepa2x", "vjepa21x", "vjepa2_causal", "resnet50"], required=True)
    ap.add_argument("--in-scope", action="store_true", help="only clips inside the data scope")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--datasets", default="")
    ap.add_argument("--splits", default="")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-hours", type=float, default=0, help="cap per dataset/split (0 = all)")
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th clip starting at i (multi-GPU)")
    ap.add_argument("--rope", default="interp", choices=["hf", "int", "interp"],
                    help="V-JEPA 2 RoPE positions for non-square inputs; hf = legacy, cached under the old names")
    a = ap.parse_args()
    dev = torch.device(f"cuda:{a.gpu}")
    idx = build_index(a.datasets.split(",") if a.datasets else None)
    if a.in_scope:
        from egokin.data.scope import filter_index
        idx = filter_index(idx)
    if a.splits:
        idx = idx[idx.split.isin(a.splits.split(","))]
    if a.max_hours:
        idx = idx.sample(frac=1.0, random_state=0)
        idx = idx[idx.groupby(["dataset", "split"]).n_frames.cumsum() <= a.max_hours * 36000]
    i_shard, n_shard = map(int, a.shard.split("/"))
    idx = idx.sort_values(["dataset", "clip_id"]).iloc[i_shard::n_shard]
    from egokin.models.vjepa_rope import feats_name

    name = feats_name(a.model, a.rope) if a.model in ("vjepa2", "vjepa2x", "vjepa2_causal") else a.model
    model = ResNet50(dev) if a.model == "resnet50" else \
        {"vjepa2": VJEPA2, "vjepa2x": VJEPA2Rich, "vjepa21x": VJEPA21Rich, "vjepa2_causal": VJEPA2Causal}[a.model](dev, a.rope)
    print(f"model {a.model}, rope {a.rope} -> cache/feats/{name}", flush=True)
    t0, frames_done = time.time(), 0
    for i, r in enumerate(idx.itertuples()):
        out = f"{ROOT}/cache/feats/{name}/{r.dataset}/{r.clip_id}.npy"
        if os.path.exists(out):
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        parts, hist = [], None
        for frames in frame_chunks(os.path.join(r.dir, "video.mp4")):
            if a.model == "vjepa2_causal":
                parts.append(model(frames, a.batch, hist))
                hist = frames[-14:] if hist is None or len(frames) >= 14 else np.concatenate([hist, frames])[-14:]
            else:
                parts.append(model(frames, a.batch))  # 512-frame chunks = whole 16-frame windows
            frames_done += len(frames)
        np.save(out + ".tmp.npy", np.concatenate(parts))
        os.replace(out + ".tmp.npy", out)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(idx)} clips, {frames_done / (time.time() - t0):.0f} frames/s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
