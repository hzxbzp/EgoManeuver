"""Trainable video backbones for the offline end-to-end model (Route A, A4/A5): V-JEPA 2-L (HF, RoPE fixed,
egokin/models/vjepa_rope.py) or V-JEPA 2.1-L (official code, egokin/models/vjepa21.py), adapted with LoRA on
blocks >= first_trainable or fully fine-tuned with layer-wise LR decay. The output per 16-frame window is the
74-token rich layout of the cached features (model/cache_features.py, vjepa2x / vjepa21x), so a head trained on
cached features can be fine-tuned end to end without changing its input:
  last layer: global + 2x4 + 4x8 (41 tokens); layer MID_LAYER: global + 4x8 (33 tokens).
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from egokin.models.vjepa_lora import LoRALinear

MID_LAYER = 15  # 0-based, as scripts/cache_features.MID_LAYER
GRID = (14, 28)  # patch grid of a 224x448 frame


def pool_rich(last, mid):
    """last, mid: (B, 8, 14, 28, D) -> (B, 8, 74, D) in the order of the cached rich features."""
    def pool(t, grids):
        x = t.permute(0, 1, 4, 2, 3).flatten(0, 1)  # (B*8, D, 14, 28)
        parts = [x.mean((2, 3)).unsqueeze(-1)] + [F.adaptive_avg_pool2d(x, g).flatten(2) for g in grids]
        return torch.cat(parts, -1).permute(0, 2, 1)  # (B*8, n, D)

    B = last.shape[0]
    return torch.cat([pool(last, [(2, 4), (4, 8)]), pool(mid, [(4, 8)])], 1).reshape(B, 8, 74, -1)


class Backbone(nn.Module):
    """kind: vjepa2 | vjepa21. adapt: lora (blocks >= first_trainable) | full. forward(wins (B, 16, 3, 224, 448))
    -> (rich tokens (B, 8, 74, D), last-layer patch tokens (B, 8, 14, 28, D))."""

    def __init__(self, kind="vjepa2", adapt="lora", first_trainable=12, r=16, rope="interp", grad_ckpt=False):
        super().__init__()
        from egokin.models.vjepa21 import no_cudnn_sdpa

        no_cudnn_sdpa()  # cuDNN attention backward gives NaN under bf16 (see vjepa21.no_cudnn_sdpa)
        self.kind, self.adapt, self._mid = kind, adapt, None
        if kind == "vjepa2":
            from transformers import AutoModel

            from egokin.models.vjepa_rope import patch_rope

            os.environ.setdefault("HF_HOME", os.path.join(os.environ.get("VEGO_DATA", "."), "cache", "hf"))
            m = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256", dtype=torch.float32)
            del m.predictor
            patch_rope(m, rope)
            if grad_ckpt:  # non-reentrant, so the mid-layer hook output stays in the graph
                m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.m, blocks = m, m.encoder.layer
            lora_parents = lambda b: [(b.attention, ("query", "value", "proj")), (b.mlp, ("fc1", "fc2"))]
            self.stem, self.top = [m.encoder.embeddings], [m.encoder.layernorm]
        elif kind == "vjepa21":
            from egokin.models.vjepa21 import load_vjepa21_large

            m = load_vjepa21_large("cpu")
            m.use_activation_checkpointing = grad_ckpt  # official forward uses non-reentrant checkpointing
            self.m, blocks = m, m.blocks
            lora_parents = lambda b: [(b.attn, ("qkv", "proj")), (b.mlp, ("fc1", "fc2"))]
            self.stem = [mod for n, mod in m.named_children() if n.startswith("patch_embed")]
            self.top = [m.norms_block]
        else:
            raise ValueError(kind)
        self.blocks = blocks
        blocks[MID_LAYER].register_forward_hook(
            lambda mod, args, out: setattr(self, "_mid", out[0] if isinstance(out, tuple) else out))
        for p in m.parameters():
            p.requires_grad_(adapt == "full")
        if kind == "vjepa21":  # not used by the video forward (image stem, hierarchical norms of other layers)
            unused = [getattr(m, "patch_embed_img", None), *list(m.norms_block)[:-1]]
            for mod in unused:
                if mod is not None:
                    mod.requires_grad_(False)
            if getattr(m, "img_mod_embed", None) is not None:
                m.img_mod_embed.requires_grad_(False)
        if adapt == "lora":
            for i, b in enumerate(blocks):
                if i < first_trainable:
                    continue
                for parent, names in lora_parents(b):
                    for n in names:
                        if isinstance(getattr(parent, n, None), nn.Linear):
                            setattr(parent, n, LoRALinear(getattr(parent, n), r=r))
        elif adapt != "full":
            raise ValueError(adapt)

    def forward(self, wins):
        B = wins.shape[0]
        if self.kind == "vjepa2":
            last = self.m.get_vision_features(wins)
        else:
            last = self.m(wins.permute(0, 2, 1, 3, 4))  # official layout (B, C, T, H, W); final norm applied
        last = last.view(B, 8, *GRID, -1)
        mid = self._mid.view(B, 8, *GRID, -1)
        self._mid = None
        return pool_rich(last, mid), last

    def param_groups(self, lr, lr_decay=0.75, weight_decay=0.05):
        """Trainable parameters as AdamW groups. LoRA: one group at lr. Full: layer-wise decay; the final norms and
        the top block get lr, block i lr * lr_decay ** (depth - 1 - i), the stem lr * lr_decay ** depth."""
        if self.adapt == "lora":
            return [{"params": [p for p in self.parameters() if p.requires_grad], "lr": lr, "weight_decay": weight_decay}]
        depth = len(self.blocks)
        tr = lambda ps: [p for p in ps if p.requires_grad]
        groups = [{"params": tr(p for mod in self.stem for p in mod.parameters()), "lr": lr * lr_decay ** depth}]
        groups += [{"params": tr(b.parameters()), "lr": lr * lr_decay ** (depth - 1 - i)} for i, b in enumerate(self.blocks)]
        groups += [{"params": tr(p for mod in self.top for p in mod.parameters()), "lr": lr}]
        seen = {id(p) for g in groups for p in g["params"]}
        rest = [p for p in self.parameters() if p.requires_grad and id(p) not in seen]
        if rest:  # e.g. modality embeddings of V-JEPA 2.1: treat as stem
            groups[0]["params"] += rest
        for g in groups:
            g["weight_decay"] = weight_decay
        return [g for g in groups if g["params"]]

    def trainable_state(self):
        """What a checkpoint has to keep: LoRA factors, or the whole backbone when fully fine-tuned."""
        sd = self.state_dict()
        if self.adapt == "full":
            return sd
        return {k: v for k, v in sd.items() if k.endswith(".a") or k.endswith(".b")}
