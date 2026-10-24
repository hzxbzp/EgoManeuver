"""Tests for the V-JEPA 2 RoPE position fix (egokin/models/vjepa_rope.py), on a tiny randomly initialised model.

Run: python -m pytest tests/test_vjepa_rope.py -q
"""

import copy

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import VJEPA2Config, VJEPA2Model  # noqa: E402

from egokin.models.vjepa_rope import feats_name, patch_rope  # noqa: E402

PATCH, TUBELET, CROP = 8, 2, 32  # pre-training grid 4x4


def tiny():
    torch.manual_seed(0)
    cfg = VJEPA2Config(hidden_size=48, num_hidden_layers=2, num_attention_heads=4, mlp_ratio=2.0, patch_size=PATCH,
                       tubelet_size=TUBELET, crop_size=CROP, frames_per_clip=8, pred_hidden_size=24,
                       pred_num_hidden_layers=1, pred_num_attention_heads=4, pred_num_mask_tokens=2)
    return VJEPA2Model(cfg).eval()


def video(h, w, frames=4):
    torch.manual_seed(1)
    return torch.randn(1, frames, 3, h, w)


def attn_layers(m):
    from transformers.models.vjepa2.modeling_vjepa2 import VJEPA2RopeAttention
    return [x for x in m.encoder.modules() if isinstance(x, VJEPA2RopeAttention)]


@pytest.mark.parametrize("mode", ["int", "interp"])
def test_square_input_unchanged(mode):
    base = tiny()
    fixed = patch_rope(copy.deepcopy(base), mode)
    x = video(CROP, CROP)
    with torch.no_grad():
        assert torch.equal(base.get_vision_features(x), fixed.get_vision_features(x))


def test_positions_match_true_grid():
    m = patch_rope(tiny(), "int")
    x = video(24, 48)  # 3 x 6 tokens per tubelet, 2 tubelets
    with torch.no_grad():
        m.get_vision_features(x)
    a = attn_layers(m)[0]
    assert a._vego_grid == (2, 3, 6)
    t, h, w = a.get_position_ids(torch.zeros(1, 36, 48))
    tt, hh, ww = torch.meshgrid(torch.arange(2), torch.arange(3), torch.arange(6), indexing="ij")
    assert torch.equal(t, tt.flatten()) and torch.equal(h, hh.flatten()) and torch.equal(w, ww.flatten())


def test_interp_keeps_positions_in_pretraining_range():
    m = patch_rope(tiny(), "interp")
    with torch.no_grad():
        m.get_vision_features(video(24, 64))  # 3 x 8 tokens; width exceeds the 4-token pre-training grid
    t, h, w = attn_layers(m)[0].get_position_ids(torch.zeros(1, 48, 48))
    assert float(w.max()) == pytest.approx(CROP // PATCH - 1) and float(w.min()) == 0.0
    assert torch.equal(h, h.round())  # 3 rows fit in the grid: unchanged integers


def test_nonsquare_differs_from_hf():
    base = tiny()
    fixed = patch_rope(copy.deepcopy(base), "int")
    x = video(24, 48)
    with torch.no_grad():
        assert not torch.allclose(base.get_vision_features(x), fixed.get_vision_features(x))


def test_feats_name():
    assert feats_name("vjepa2", "hf") == "vjepa2"
    assert feats_name("vjepa2_causal", "int") == "vjepa2_causal_rint"


def test_interp_positions_follow_activation_dtype():
    """fp16 inference: float32 fractional positions would make q/k float32 while v stays fp16 (SDPA error)."""
    m = patch_rope(tiny(), "interp")
    with torch.no_grad():
        m.get_vision_features(video(24, 64))
    a = attn_layers(m)[0]
    for dt in (torch.float16, torch.bfloat16, torch.float32):
        t, h, w = a.get_position_ids(torch.zeros(1, 48, 48, dtype=dt))
        assert t.dtype == h.dtype == w.dtype == dt
