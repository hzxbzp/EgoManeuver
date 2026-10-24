"""V-JEPA 2.1 ViT-L/16 (distilled from ViT-G, 384 px) through the official Meta code (third_party/vjepa2), for the
A3 backbone comparison. The official RoPE handles non-square inputs (interpolate_rope=True), so no patch is needed.

Checkpoint: https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt (code MIT / Apache-2.0).
Input layout of the official encoder: (B, C, T, H, W). Run it like the official evals: fp32 weights under bf16
autocast. With fp16 weights and no autocast the RoPE returns fp32 q/k next to an fp16 v and SDPA fails.
"""

import contextlib
import os
import sys

import torch

REPO = os.environ.get("VJEPA2_REPO", "./third_party/vjepa2")


def no_cudnn_sdpa():
    """Use the flash / memory-efficient SDPA kernels instead of cuDNN: with torch 2.11 + cuDNN 9.19 the cuDNN
    attention backward returns NaN for V-JEPA 2.1 under bf16 autocast (first at block 2; the forward agrees with the
    other kernels to bf16 precision). The official attention re-enables every backend inside the deprecated
    torch.backends.cuda.sdp_kernel() context, so that context becomes a no-op too. Process-wide."""
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.sdp_kernel = lambda *args, **kwargs: contextlib.nullcontext()


def load_vjepa21_large(dev, ckpt=None):
    ckpt = ckpt or os.path.join(os.environ.get("VEGO_DATA", "."), "cache", "vjepa2_1", "vjepa2_1_vitl_dist_vitG_384.pt")
    no_cudnn_sdpa()
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from src.hub.backbones import _clean_backbone_key, _make_vjepa2_1_model, vjepa2_1_teacher_embed_dim

    enc, pred = _make_vjepa2_1_model(model_name="vjepa2_1_vit_large_384", checkpoint_key="ema_encoder", img_size=384,
                                     predictor_depth=12, predictor_num_mask_tokens=8, n_output_distillation=1,
                                     return_all_tokens=True, teacher_embed_dim=vjepa2_1_teacher_embed_dim,
                                     pretrained=False)
    del pred
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    enc.load_state_dict(_clean_backbone_key(sd["ema_encoder"]), strict=True)
    del sd
    return enc.to(dev).eval()


def encode(enc, x):
    """x (B, C, T, H, W) normalized video -> (B, N, D) final normalized tokens (fp32)."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return enc(x).float()
