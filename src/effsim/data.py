"""NIGHTS triplets from data.csv and the 224 px copy."""

import csv
import shutil
import tarfile
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

MIN_VOTES = 6
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def extract_shards(drive_dir, out):
    """Unpack the tar shards written by notebooks/prepare_nights.ipynb, plus data.csv."""
    drive_dir, out = Path(drive_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    local = out / "_shard.tar"
    shards = sorted((drive_dir / "shards").glob("*.tar"))
    for shard in shards:
        shutil.copyfile(shard, local)
        with tarfile.open(local) as tar:
            tar.extractall(out, filter="data")
    local.unlink(missing_ok=True)
    shutil.copyfile(drive_dir / "data.csv", out / "data.csv")
    return len(shards)


def load_triplets(csv_path, split, ids_file=None):
    """Rows of `split` with votes >= 6, optionally restricted to the ids in a data/splits/ file."""
    with open(csv_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(r["votes"]) >= MIN_VOTES and r["split"] == split]
    if ids_file is not None:
        with open(ids_file, newline="") as f:
            keep = {r["id"] for r in csv.DictReader(f)}
        rows = [r for r in rows if r["id"] in keep]
        if len(rows) != len(keep):
            raise ValueError(f"{ids_file}: {len(keep) - len(rows)} ids not in split '{split}'")
    return rows


class TripletDataset(Dataset):
    """Returns (ref, left, right, target, id); target = right_vote, so 1 means right is closer."""

    def __init__(self, root, rows, normalize=True):
        self.root = Path(root)
        self.rows = rows
        steps = [transforms.ToTensor()]
        if normalize:
            steps.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
        self.transform = transforms.Compose(steps)
        self.is_imagenet = torch.tensor([r["is_imagenet"].upper() == "TRUE" for r in rows])

    def __len__(self):
        return len(self.rows)

    def _load(self, rel):
        with Image.open(self.root / rel) as im:
            return self.transform(im.convert("RGB"))

    def __getitem__(self, i):
        r = self.rows[i]
        return (self._load(r["ref_path"]), self._load(r["left_path"]), self._load(r["right_path"]),
                torch.tensor(float(r["right_vote"])), int(r["id"]))
