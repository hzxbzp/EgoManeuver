"""The end-to-end encoder (egokin/models/backbones.pool_rich) must give the same 74-token layout as the cached
rich features (model/cache_features.py, vjepa2x / vjepa21x), so heads trained on the cache transfer."""

import importlib.util
from egokin.scripts import script_path  # noqa: E402
import os

import torch

from egokin.models.backbones import pool_rich

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, script_path(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pool_rich_matches_cache_layout():
    cf = _load("cache_features")
    torch.manual_seed(0)
    B, D = 2, 16
    last, mid = torch.randn(B, 8, 14, 28, D), torch.randn(B, 8, 14, 28, D)
    grid = lambda t: t.permute(0, 1, 4, 2, 3).reshape(B * 8, D, 14, 28)
    ref = torch.cat([cf.pool_step_tokens(grid(last), [(2, 4), (4, 8)]), cf.pool_step_tokens(grid(mid), [(4, 8)])], 1)
    out = pool_rich(last, mid)
    assert out.shape == (B, 8, 74, D)
    torch.testing.assert_close(out.reshape(B * 8, 74, D), ref.float())


def test_rich_token_sets():
    m1 = _load("train_m1")
    t = m1.RICH_TOKENS
    assert t["pool9"] == list(range(9)) and len(t["grid33"]) == 33 and len(t["mid33"]) == 33
    assert t["grid33"][0] == 0 and t["mid33"][0] == 41 and max(t["grid33+mid33"]) == 73
    # pool9 of the rich layout = global + 2x4 of the last layer (the legacy 9-token features)
    torch.manual_seed(1)
    last = torch.randn(1, 8, 14, 28, 4)
    out = pool_rich(last, last)[0, :, :9]
    x = last[0].permute(0, 3, 1, 2)
    legacy = torch.cat([x.mean((2, 3)).unsqueeze(-1), torch.nn.functional.adaptive_avg_pool2d(x, (2, 4)).flatten(2)], -1)
    torch.testing.assert_close(out, legacy.permute(0, 2, 1))
