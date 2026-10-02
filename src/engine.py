"""Training / inference loops shared by all experiments."""
import copy
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import SEED

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights(labels, n_cls):
    """Inverse-frequency weights: n_total / (n_cls * n_c)."""
    counts = np.bincount(labels, minlength=n_cls).astype(np.float32)
    return torch.tensor(counts.sum() / (n_cls * counts), dtype=torch.float32)


@torch.inference_mode()
def predict_proba(model, dataset, batch_size=64, tta_hflip=False):
    """Softmax probabilities for every sample (optionally averaged with a h-flip view)."""
    model.eval().to(DEVICE)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    out = []
    for x, _ in loader:
        x = x.to(DEVICE)
        p = torch.softmax(model(x), 1)
        if tta_hflip:
            p = (p + torch.softmax(model(torch.flip(x, dims=[3])), 1)) / 2
        out.append(p.cpu())
    return torch.cat(out).numpy()


def fit(model, train_ds, val_ds=None, *, epochs, lr, batch_size=64, optimizer="adam",
        weight_decay=0.0, scheduler=None, warmup_epochs=0, class_w=None,
        label_smoothing=0.0, select_best=False, param_groups=None, name="model",
        log_file=None):
    """Generic supervised training loop.

    select_best=True keeps the weights of the epoch with the best accuracy on `val_ds`
    (the protocol used in the original competition notebook). With select_best=False
    the final-epoch weights are returned, so `val_ds` is only monitored.
    """
    model = model.to(DEVICE)
    params = param_groups if param_groups is not None else model.parameters()
    if optimizer == "adam":
        opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    steps_per_epoch = len(loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    base_lrs = [g["lr"] for g in opt.param_groups]

    def lr_factor(step):
        if scheduler != "cosine":
            return 1.0
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    crit = nn.CrossEntropyLoss(
        weight=None if class_w is None else class_w.to(DEVICE),
        label_smoothing=label_smoothing)

    history, best = [], {"val_acc": -1.0, "epoch": 0, "state": None}
    step = 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum = correct = n = 0
        for x, y in loader:
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * lr_factor(step)
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            step += 1
            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += y.size(0)
        rec = {"epoch": ep, "train_loss": loss_sum / n, "train_acc": correct / n,
               "lr": opt.param_groups[-1]["lr"], "sec": round(time.time() - t0, 1)}

        if val_ds is not None:
            proba = predict_proba(model, val_ds)
            y_val = np.array([int(l) for l in val_ds.labels])
            rec["val_acc"] = float((proba.argmax(1) == y_val).mean())
            if select_best and rec["val_acc"] > best["val_acc"]:
                best = {"val_acc": rec["val_acc"], "epoch": ep,
                        "state": copy.deepcopy(model.state_dict())}
        history.append(rec)

        msg = (f"[{name}] ep {ep:2d}/{epochs} loss {rec['train_loss']:.3f} "
               f"train_acc {rec['train_acc']:.3f}")
        if "val_acc" in rec:
            msg += f" val_acc {rec['val_acc']:.3f}"
        msg += f" ({rec['sec']}s)"
        print(msg, flush=True)
        if log_file is not None:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"name": name, **rec}) + "\n")

    if select_best and best["state"] is not None:
        model.load_state_dict(best["state"])
        print(f"[{name}] restored best epoch {best['epoch']} (val_acc {best['val_acc']:.3f})",
              flush=True)
    return model, {"history": history, "best_epoch": best["epoch"] if select_best else epochs}
