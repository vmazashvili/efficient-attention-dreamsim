import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from effsim.train import (build_b1, hinge_loss, load_checkpoint, make_optimizer, save_checkpoint,
                          train_b1, train_step)

CFG = {"lr": 3e-4, "betas": [0.5, 0.999], "weight_decay": 0.0, "batch_size": 4, "epochs": 2,
       "margin": 0.05, "amp": True,
       "lora": {"target_modules": ["qkv"], "r": 16, "alpha": 32, "dropout": 0.2}}


TINY = "vit_tiny_patch16_224"  # same block/qkv layout as ViT-B, small enough for laptop CPU tests


def tiny_b1(seed=0):
    return build_b1(CFG, seed=seed, pretrained=False, name=TINY)


def fake_triplets(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    imgs = torch.randn(3, n, 3, 224, 224, generator=g)
    target = (torch.arange(n) % 2).float()
    return TensorDataset(imgs[0], imgs[1], imgs[2], target, torch.arange(n))


def lora_params(model):
    return {k: v.detach().clone() for k, v in model.named_parameters() if "lora_" in k}


def test_hinge_loss_matches_dreamsim_formula():
    d0, d1 = torch.tensor([0.5, 0.2, 0.3]), torch.tensor([0.2, 0.5, 0.3])
    target = torch.tensor([1.0, 1.0, 0.0])
    # logit = d0 - d1 = [0.3, -0.3, 0]; y = [1, 1, -1]; max(0, 0.05 - logit*y) = [0, 0.35, 0.05]
    assert hinge_loss(d0, d1, target, 0.05).item() == pytest.approx(0.40 / 3)


def test_only_lora_is_trainable():
    model = tiny_b1(0)
    trainable = [k for k, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in k and "qkv" in k for k in trainable)
    assert len(trainable) == 12 * 2


def test_few_steps_reduce_loss():
    model = tiny_b1(0)
    model.train()
    opt = make_optimizer(model, {**CFG, "lr": 1e-3})
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    batch = next(iter(DataLoader(fake_triplets(4), batch_size=4)))
    losses = [train_step(model, batch, opt, scaler, CFG["margin"], "cpu", amp=False) for _ in range(6)]
    assert losses[-1] < losses[0]


def test_checkpoint_resume_round_trip(tmp_path):
    model = tiny_b1(0)
    base = {k: v.clone() for k, v in model.named_parameters() if "lora_" not in k}
    opt = make_optimizer(model, CFG)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    batch = next(iter(DataLoader(fake_triplets(4), batch_size=4)))
    model.train()
    train_step(model, batch, opt, scaler, CFG["margin"], "cpu", amp=False)
    save_checkpoint(tmp_path / "last.pt", model, opt, scaler, epoch=3)
    saved = lora_params(model)

    fresh = tiny_b1(1)
    fresh.load_state_dict({k: v for k, v in model.state_dict().items() if "lora_" not in k}, strict=False)
    assert any(not torch.equal(saved[k], v) for k, v in lora_params(fresh).items())
    opt2 = make_optimizer(fresh, CFG)
    ck = load_checkpoint(tmp_path / "last.pt", fresh, opt2, scaler)
    assert ck["epoch"] == 3
    restored = lora_params(fresh)
    assert saved.keys() == restored.keys() and all(torch.equal(saved[k], restored[k]) for k in saved)
    s1, s2 = opt.state_dict()["state"], opt2.state_dict()["state"]
    assert s1.keys() == s2.keys() and all(torch.equal(s1[i]["exp_avg"], s2[i]["exp_avg"]) for i in s1)
    assert all(torch.equal(base[k], v) for k, v in model.named_parameters() if "lora_" not in k)


def test_train_b1_resumes_and_keeps_best(tmp_path):
    ds = fake_triplets(8)
    val = DataLoader(fake_triplets(4, seed=1), batch_size=4)
    model = tiny_b1(0)
    r1 = train_b1(model, ds, val, tmp_path, CFG, seed=0, device="cpu", epochs=1, num_workers=0, log=lambda *_: None)
    assert [h["epoch"] for h in r1["history"]] == [0, 1]

    model2 = tiny_b1(0)
    r2 = train_b1(model2, ds, val, tmp_path, CFG, seed=0, device="cpu", epochs=2, num_workers=0,
                  log=lambda *_: None)
    assert [h["epoch"] for h in r2["history"]] == [0, 1, 2]
    best = torch.load(tmp_path / "best.pt", weights_only=False)
    assert best["epoch"] == r2["best_epoch"]
    assert r2["best_val"] == max(h["val_2afc"] for h in r2["history"])
