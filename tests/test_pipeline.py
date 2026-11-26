import csv
import importlib.util

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from effsim.backbone import embed, load_dino
from effsim.data import TripletDataset, extract_shards, load_triplets
from effsim.evaluate import append_run, decide, distance, save_embeddings, score_2afc, summarise

from conftest import ROOT

CSV = ROOT / "data" / "data.csv"
SMOKE_VAL = ROOT / "data" / "splits" / "smoke_val.csv"
needs_csv = pytest.mark.skipif(not CSV.exists(), reason="data/data.csv missing: run scripts/fetch_data_csv.py")


def test_decide_known_answers():
    ref = torch.tensor([[1.0, 0.0]] * 4)
    near, far = torch.tensor([[1.0, 0.1]]), torch.tensor([[0.0, 1.0]])
    left = torch.cat([near, far, near, far])
    right = torch.cat([far, near, far, near])
    target = torch.tensor([0.0, 1.0, 1.0, 0.0])  # 0: humans picked left, 1: right
    d0, d1 = distance(ref, left), distance(ref, right)
    assert decide(d0, d1, target).tolist() == [True, True, False, False]


def test_decide_tie_counts_as_left():
    d = torch.tensor([0.3])
    assert decide(d, d, torch.tensor([0.0])).item() and not decide(d, d, torch.tensor([1.0])).item()


class _Fixed(torch.utils.data.Dataset):
    def __init__(self, embs, targets):
        self.embs, self.targets = embs, targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        ref, left, right = self.embs[i]
        return ref, left, right, self.targets[i], i


def test_score_2afc_on_handmade_embeddings():
    embs = [(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.1]), torch.tensor([0.0, 1.0])),
            (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.2])),
            (torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0]))]
    targets = torch.tensor([0.0, 1.0, 1.0])
    res = score_2afc(lambda x: x, DataLoader(_Fixed(embs, targets), batch_size=2), "cpu", keep_embeddings=True)
    assert res["correct"].tolist() == [True, True, False]
    s = summarise(res, torch.tensor([True, True, False]))
    assert s["all"] == pytest.approx(2 / 3) and s["imagenet"] == 1.0 and s["no_imagenet"] == 0.0
    assert res["emb"].shape == (3, 3, 2)


@pytest.fixture(scope="module")
def smoke8(tmp_path_factory):
    rows = load_triplets(CSV, "val", SMOKE_VAL)[:8]
    root = tmp_path_factory.mktemp("nights224")
    rng = np.random.default_rng(0)
    for r in rows:
        for c in ("ref_path", "left_path", "right_path"):
            p = root / r[c]
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)).save(p)
    return root, rows


@needs_csv
def test_load_triplets_subset():
    rows = load_triplets(CSV, "val", SMOKE_VAL)
    assert len(rows) == 128 and all(int(r["votes"]) >= 6 and r["split"] == "val" for r in rows)


@needs_csv
def test_dataset_items(smoke8):
    root, rows = smoke8
    ref, left, right, target, i = TripletDataset(root, rows)[0]
    assert ref.shape == left.shape == right.shape == (3, 224, 224)
    assert target.item() == float(rows[0]["right_vote"]) and i == int(rows[0]["id"])
    raw = TripletDataset(root, rows, normalize=False)[0][0]
    assert raw.min() >= 0 and raw.max() <= 1 and ref.min() < 0


@needs_csv
def test_b0_path_on_8_smoke_val_triplets(smoke8, tmp_path):
    root, rows = smoke8
    model = load_dino(pretrained=False)
    ds = TripletDataset(root, rows)
    res = score_2afc(lambda x: embed(model, x), DataLoader(ds, batch_size=4), "cpu", keep_embeddings=True)
    assert res["correct"].shape == (8,) and res["emb"].shape == (8, 3, 768)
    assert res["ids"].tolist() == [int(r["id"]) for r in rows]
    s = summarise(res, ds.is_imagenet)
    assert 0 <= s["all"] <= 1 and s["n"] == 8

    save_embeddings(tmp_path / "b0_smoke", res)
    assert np.load(tmp_path / "b0_smoke.npy").shape == (8, 3, 768)
    assert np.load(tmp_path / "b0_smoke_ids.npy").tolist() == res["ids"].tolist()


def test_embed_is_block12_cls_before_final_norm():
    model = load_dino(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        cls = embed(model, x)
        per_block = embed(model, x, all_blocks=True)
        normed = model.forward_features(x)[:, 0]
        renormed = model.norm(cls)
    assert cls.shape == (2, 768) and per_block.shape == (2, 12, 768)
    assert torch.allclose(per_block[:, -1], cls)
    assert torch.allclose(renormed, normed, atol=1e-5)
    assert not torch.allclose(cls, normed, atol=1e-3)


def test_extract_shards(tmp_path):
    import tarfile

    drive = tmp_path / "drive"
    (drive / "shards").mkdir(parents=True)
    (drive / "data.csv").write_text("id\n")
    src = tmp_path / "ref" / "000"
    src.mkdir(parents=True)
    (src / "001.png").write_bytes(b"x")
    with tarfile.open(drive / "shards" / "ref_000.tar", "w") as tar:
        tar.add(src / "001.png", arcname="ref/000/001.png")
    out = tmp_path / "out"
    assert extract_shards(drive, out) == 1
    assert (out / "ref/000/001.png").read_bytes() == b"x" and (out / "data.csv").exists()
    assert not (out / "_shard.tar").exists()


def test_append_run(tmp_path):
    p = tmp_path / "runs.csv"
    p.write_text((ROOT / "results" / "runs.csv").read_text().splitlines()[0] + "\n")
    append_run(p, run_id="B0", stage="zero-shot", test_2afc=0.9)
    rows = list(csv.DictReader(open(p)))
    assert rows == [{**{k: "" for k in rows[0]}, "run_id": "B0", "stage": "zero-shot", "test_2afc": "0.9"}]
    with pytest.raises(KeyError):
        append_run(p, nonsense=1)


@pytest.mark.skipif(importlib.util.find_spec("dreamsim") is None, reason="dreamsim not installed (B-rel runs on Colab)")
def test_brel_embed_shape(tmp_path):
    from effsim.brel import load_brel

    embed_fn = load_brel("cpu", cache_dir=tmp_path)
    with torch.inference_mode():
        assert embed_fn(torch.rand(2, 3, 224, 224)).shape == (2, 768)
