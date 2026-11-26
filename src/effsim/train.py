"""Stage B: DreamSim hinge-loss training with LoRA on qkv (SPEC §5.1)."""

import os
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader

from effsim.backbone import MODEL_NAME, embed, load_dino
from effsim.evaluate import distance, score_2afc


def build_b1(train_cfg, seed, pretrained=True, name=MODEL_NAME):
    torch.manual_seed(seed)
    lora = train_cfg["lora"]
    config = LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"], bias="none",
                        target_modules=list(lora["target_modules"]))
    return get_peft_model(load_dino(pretrained, name), config)


def vit_of(model):
    return model.base_model.model


def hinge_loss(d0, d1, target, margin):
    """DreamSim HingeLoss on logit = d0 - d1, summed, divided by batch size."""
    y = 2 * torch.round(target) - 1
    return torch.clamp(margin - (d0 - d1) * y, min=0).sum() / target.shape[0]


def make_optimizer(model, train_cfg):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=train_cfg["lr"], betas=tuple(train_cfg["betas"]),
                            weight_decay=train_cfg["weight_decay"])


def train_step(model, batch, opt, scaler, margin, device, amp):
    ref, left, right, target, _ = (t.to(device, non_blocking=True) for t in batch)
    b = ref.shape[0]
    # ref is embedded twice, as in DreamSim's two forward calls, so it gets independent LoRA dropout
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        e = embed(vit_of(model), torch.cat([ref, left, ref, right]))
    e = e.float()
    loss = hinge_loss(distance(e[:b], e[b:2 * b]), distance(e[2 * b:3 * b], e[3 * b:]), target, margin)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    return loss.item()


def save_checkpoint(path, model, opt, scaler, **state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"lora": get_peft_model_state_dict(model), "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(), **state}, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, opt=None, scaler=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    set_peft_model_state_dict(model, ck["lora"])
    if opt is not None:
        opt.load_state_dict(ck["optimizer"])
    if scaler is not None:
        scaler.load_state_dict(ck["scaler"])
    return ck


def val_2afc(model, loader, device):
    model.eval()
    return score_2afc(lambda x: embed(vit_of(model), x), loader, device)["correct"].float().mean().item()


def train_b1(model, train_ds, val_loader, ckpt_dir, train_cfg, seed, device, epochs=None,
             train_eval_loader=None, num_workers=2, log=print):
    """Trains, scoring val each epoch; resumes from ckpt_dir/last.pt; returns with the best weights loaded."""
    epochs = epochs or train_cfg["epochs"]
    amp = bool(train_cfg["amp"]) and device == "cuda"
    margin = train_cfg["margin"]
    ckpt_dir = Path(ckpt_dir)
    last, best = ckpt_dir / "last.pt", ckpt_dir / "best.pt"
    model.to(device)
    opt = make_optimizer(model, train_cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    if last.exists():
        ck = load_checkpoint(last, model, opt, scaler)
        history, best_val, best_epoch = ck["history"], ck["best_val"], ck["best_epoch"]
        log(f"resumed after epoch {ck['epoch']} (best val {best_val:.4f} at epoch {best_epoch})")
        start = ck["epoch"] + 1
    else:
        v = val_2afc(model, val_loader, device)
        history, best_val, best_epoch = [{"epoch": 0, "val_2afc": v}], v, 0
        save_checkpoint(best, model, opt, scaler, epoch=0)
        save_checkpoint(last, model, opt, scaler, epoch=0, history=history, best_val=v, best_epoch=0)
        log(f"epoch 0: val {v:.4f}")
        start = 1

    for epoch in range(start, epochs + 1):
        t0 = time.time()
        model.train()
        loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=num_workers,
                            pin_memory=device == "cuda", generator=torch.Generator().manual_seed(seed + epoch))
        losses = [train_step(model, batch, opt, scaler, margin, device, amp) for batch in loader]
        minutes = (time.time() - t0) / 60
        entry = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "minutes": minutes,
                 "val_2afc": val_2afc(model, val_loader, device)}
        if train_eval_loader is not None:
            entry["train_2afc"] = val_2afc(model, train_eval_loader, device)
        history.append(entry)
        if entry["val_2afc"] > best_val:
            best_val, best_epoch = entry["val_2afc"], epoch
            save_checkpoint(best, model, opt, scaler, epoch=epoch)
        save_checkpoint(last, model, opt, scaler, epoch=epoch, history=history, best_val=best_val,
                        best_epoch=best_epoch)
        log(" ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in entry.items()))

    load_checkpoint(best, model)
    model.eval()
    timed = [h["minutes"] for h in history if "minutes" in h]
    return {"best_val": best_val, "best_epoch": best_epoch, "history": history,
            "min_per_epoch": sum(timed) / len(timed) if timed else None}
