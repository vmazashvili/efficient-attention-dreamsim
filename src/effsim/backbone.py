"""DINO ViT-B/16 loading and CLS embedding (SPEC §5.1)."""

import timm
import torch
from torch.utils.checkpoint import checkpoint

MODEL_NAME = "vit_base_patch16_224.dino"


def load_dino(pretrained=True, name=MODEL_NAME):
    return timm.create_model(name, pretrained=pretrained).eval()


def embed(model, x, all_blocks=False, grad_checkpoint=False):
    """CLS token after the last block, before the final norm. all_blocks=True: (B, n_blocks, D).

    grad_checkpoint recomputes each block's activations in backward to save memory.
    """
    x = model.norm_pre(model.patch_drop(model._pos_embed(model.patch_embed(x))))
    ckpt = grad_checkpoint and torch.is_grad_enabled()
    cls = []
    for blk in model.blocks:
        x = checkpoint(blk, x, use_reentrant=False) if ckpt else blk(x)
        if all_blocks:
            cls.append(x[:, 0])
    return torch.stack(cls, dim=1) if all_blocks else x[:, 0]
