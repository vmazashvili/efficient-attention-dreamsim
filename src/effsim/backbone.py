"""DINO ViT-B/16 loading and CLS embedding (SPEC §5.1)."""

import timm
import torch

MODEL_NAME = "vit_base_patch16_224.dino"


def load_dino(pretrained=True, name=MODEL_NAME):
    return timm.create_model(name, pretrained=pretrained).eval()


def embed(model, x, all_blocks=False):
    """CLS token after the last block, before the final norm. all_blocks=True: (B, n_blocks, D)."""
    x = model.norm_pre(model.patch_drop(model._pos_embed(model.patch_embed(x))))
    cls = []
    for blk in model.blocks:
        x = blk(x)
        if all_blocks:
            cls.append(x[:, 0])
    return torch.stack(cls, dim=1) if all_blocks else x[:, 0]
