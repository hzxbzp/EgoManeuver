"""Correct 3D-RoPE positions of the Hugging Face V-JEPA 2 encoder for non-square inputs.

HF `VJEPA2RopeAttention` derives (frame, row, column) of every token from its flat index assuming the square
pre-training grid (crop_size // patch_size = 16 per side). For our 224x448 inputs (14x28 tokens per tubelet) that
scrambles all vertical and temporal relative positions. `patch_rope` records the real token grid from the input
shape (forward pre-hook on the patch embedding) and replaces the position computation of every attention layer:

- "int":    true integer positions (the official Meta code with handle_nonsquare_inputs=True); widths beyond the
            pre-training grid are extrapolated (28 columns vs 16 seen in pre-training);
- "interp": rows/columns that exceed the pre-training grid are rescaled into [0, grid - 1] (fractional positions);
            the default (HDD offline mAP 50.3 vs 48.3 for "int" and 44.7 for "hf", 3 seeds);
- "hf":     leave the model unchanged (legacy behaviour, features under the old cache names).

For square inputs at the pre-training size every mode gives exactly the original positions.
"""

import types

import torch

MODES = ("hf", "int", "interp")


def feats_name(base: str, mode: str) -> str:
    """Cache directory name for features computed with a given RoPE mode (legacy names for "hf")."""
    return base if mode == "hf" else f"{base}_r{mode}"


def patch_rope(model, mode: str = "interp"):
    """model: transformers VJEPA2Model. Returns the same model, patched in place."""
    if mode not in MODES:
        raise ValueError(f"rope mode {mode!r} not in {MODES}")
    if mode == "hf":
        return model
    from transformers.models.vjepa2.modeling_vjepa2 import VJEPA2RopeAttention

    cfg = model.config
    grid0 = cfg.crop_size // cfg.patch_size
    attns = [m for m in model.encoder.modules() if isinstance(m, VJEPA2RopeAttention)]

    def set_grid(_module, args, kwargs):
        x = args[0] if args else kwargs["pixel_values_videos"]  # (B, T, C, H, W)
        grid = (max(x.shape[1], cfg.tubelet_size) // cfg.tubelet_size, x.shape[-2] // cfg.patch_size,
                x.shape[-1] // cfg.patch_size)
        for a in attns:
            a._vego_grid = grid

    def get_position_ids(self, x, masks=None):
        grid = getattr(self, "_vego_grid", None)
        if masks is not None or grid is None:
            return type(self).get_position_ids(self, x, masks)
        T, H, W = grid
        n = x.size(1)
        if n != T * H * W:
            raise RuntimeError(f"token count {n} does not match grid {grid}")
        ids = torch.arange(n, device=x.device)
        t, h, w = ids // (H * W), (ids % (H * W)) // W, ids % W
        if mode == "interp":  # fractional positions in the activation dtype (fp32 positions would upcast q/k only)
            h = h * ((grid0 - 1) / (H - 1)) if H > grid0 else h
            w = w * ((grid0 - 1) / (W - 1)) if W > grid0 else w
            t, h, w = (v.to(x.dtype) for v in (t, h, w))
        return t, h, w

    model.encoder.embeddings.register_forward_pre_hook(set_grid, with_kwargs=True)
    for a in attns:
        a.get_position_ids = types.MethodType(get_position_ids, a)
    model._vego_rope = mode
    return model
