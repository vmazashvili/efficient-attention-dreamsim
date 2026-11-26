"""2AFC scoring with DreamSim's convention, embedding export, run log."""

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def distance(a, b):
    return 1 - F.cosine_similarity(a, b, dim=-1)


def decide(d0, d1, target):
    """DreamSim training/train.py: decision = d1 < d0 (right closer); correct = (target >= 0.5) == decision."""
    return (target >= 0.5) == (d1 < d0)


@torch.inference_mode()
def score_2afc(embed_fn, loader, device, keep_embeddings=False):
    out = {"correct": [], "d0": [], "d1": [], "ids": [], "emb": []}
    for ref, left, right, target, ids in loader:
        n = ref.shape[0]
        e = embed_fn(torch.cat([ref, left, right]).to(device)).float()
        e_ref, e_left, e_right = e[:n], e[n:2 * n], e[2 * n:]
        d0, d1 = distance(e_ref, e_left).cpu(), distance(e_ref, e_right).cpu()
        out["correct"].append(decide(d0, d1, target))
        out["d0"].append(d0)
        out["d1"].append(d1)
        out["ids"].append(ids)
        if keep_embeddings:
            out["emb"].append(torch.stack([e_ref, e_left, e_right], dim=1).cpu().half())
    return {k: torch.cat(v) for k, v in out.items() if v}


def summarise(result, is_imagenet):
    c = result["correct"].float()
    return {
        "all": c.mean().item(),
        "imagenet": c[is_imagenet].mean().item(),
        "no_imagenet": c[~is_imagenet].mean().item(),
        "n": len(c),
    }


def save_embeddings(path, result):
    """Writes <path>.npy (N, 3, D) fp16 as ref/left/right, and <path>_ids.npy."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix(".npy"), result["emb"].numpy())
    np.save(path.with_name(path.stem + "_ids.npy"), result["ids"].numpy())


def append_run(csv_path, **row):
    with open(csv_path, newline="") as f:
        header = next(csv.reader(f))
    unknown = set(row) - set(header)
    if unknown:
        raise KeyError(f"not in {csv_path} header: {sorted(unknown)}")
    with open(csv_path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=header).writerow(row)
