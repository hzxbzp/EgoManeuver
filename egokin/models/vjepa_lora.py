"""V-JEPA 2 encoder with LoRA adapters, producing the same pooled per-step tokens as the cached
causal features (model/cache_features.py, model vjepa2_causal): for step t, the 16 frames ending at
frame 2t+1 are encoded and the last tubelet's tokens are pooled to a 2x4 grid + a global token (9 x 1024).
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.zeros(r, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.a.t().to(x.dtype)) @ self.b.t().to(x.dtype) * self.scale


def load_vjepa_lora(first_trainable: int = 12, r: int = 16, names=("query", "value", "proj", "fc1", "fc2"),
                    rope: str = "interp"):
    """Frozen V-JEPA 2 ViT-L; LoRA on layers >= first_trainable. Earlier layers get no gradient, so their
    activations are not kept for backward. rope: RoPE position mode for non-square inputs (vjepa_rope.py;
    "hf" = legacy scrambled positions)."""
    from transformers import AutoModel

    from egokin.models.vjepa_rope import patch_rope

    os.environ.setdefault("HF_HOME", os.path.join(os.environ.get("VEGO_DATA", "."), "cache", "hf"))
    m = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256", dtype=torch.float32)
    del m.predictor
    patch_rope(m, rope)
    for p in m.parameters():
        p.requires_grad_(False)
    for i, layer in enumerate(m.encoder.layer):
        if i < first_trainable:
            continue
        for parent in (layer.attention, layer.mlp):
            for n in names:
                if hasattr(parent, n) and isinstance(getattr(parent, n), nn.Linear):
                    setattr(parent, n, LoRALinear(getattr(parent, n), r=r))
    return m


def prep(frames_u8: torch.Tensor) -> torch.Tensor:
    """(N, H, W, 3) uint8 on device -> (N, 3, 224, 448) normalized float."""
    x = frames_u8.permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=(224, 448), mode="bilinear", align_corners=False, antialias=True)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def pooled_last_tubelet(model, windows: torch.Tensor) -> torch.Tensor:
    """windows: (B, 16, 3, 224, 448) -> (B, 9, 1024): pooled tokens of the last tubelet of each window."""
    tok = model.get_vision_features(windows)  # (B, 8*14*28, D)
    B = tok.shape[0]
    last = tok.view(B, 8, 14, 28, -1)[:, -1].permute(0, 3, 1, 2)  # (B, D, 14, 28)
    grid = F.adaptive_avg_pool2d(last, (2, 4)).flatten(2)
    glob = last.mean((2, 3)).unsqueeze(-1)
    return torch.cat([glob, grid], -1).permute(0, 2, 1)


def pooled_all_tubelets(model, windows: torch.Tensor) -> torch.Tensor:
    """windows: (B, 16, 3, 224, 448) -> (B, 8, 9, 1024): every tubelet step pooled to global + 2x4 grid, the same
    tokens as the cached non-causal `vjepa2` features (model/cache_features.py)."""
    tok = model.get_vision_features(windows)  # (B, 8*14*28, D)
    return pool_tokens(tok.view(tok.shape[0], 8, 14, 28, -1))


def pool_tokens(t: torch.Tensor) -> torch.Tensor:
    """(B, 8, 14, 28, D) patch tokens -> (B, 8, 9, D): global mean + 2x4 grid per tubelet step."""
    B = t.shape[0]
    x = t.permute(0, 1, 4, 2, 3).flatten(0, 1)  # (B*8, D, 14, 28)
    grid = F.adaptive_avg_pool2d(x, (2, 4)).flatten(2)
    glob = x.mean((2, 3)).unsqueeze(-1)
    return torch.cat([glob, grid], -1).permute(0, 2, 1).reshape(B, 8, 9, -1)
