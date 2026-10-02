"""Experiment 1 - Re-run the original competition journey on a fixed hold-out split.

During the competition every step was judged on a single random 85/15 split, whose
noise made it hard to tell real improvements from luck. Here each step is trained with
the same protocol as the original notebook (Adam 1e-3, 30 epochs, best epoch picked on
an inner 15% validation split of the dev set), then scored once on the untouched
hold-out split, so the steps are directly comparable.

    python -m experiments.01_baselines [--steps 0 1 2 3 4 5 6]
"""
import argparse

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import RESULTS_DIR, SEED
from src.data import (ArrayDataset, baseline_transform, holdout_split, load_train,
                      resize_images)
from src.engine import class_weights, fit, predict_proba, seed_everything
from src.metrics import save_proba, save_result, summarize
from src.models import MLP, DeepCNN, SimpleCNN

EPOCHS, LR, BATCH = 30, 1e-3, 64

# step -> (description, model factory, img size, augment, class weights + cosine)
STEPS = {
    0: ("MLP (starter code)", lambda: MLP(64), 64, False, False),
    1: ("CNN 3 blocks", lambda: SimpleCNN(bn=False), 64, False, False),
    2: ("+ augmentation (flip, rotate 15)", lambda: SimpleCNN(bn=False), 64, True, False),
    3: ("+ BatchNorm", lambda: SimpleCNN(bn=True), 64, True, False),
    4: ("+ class weights + cosine LR", lambda: SimpleCNN(bn=True), 64, True, True),
    5: ("DeepCNN 4 blocks @96px", DeepCNN, 96, True, True),
}


def run_single(step, x, y, dev, hold, log_file):
    desc, make_model, size, augment, tricks = STEPS[step]
    x = resize_images(x, size)
    seed_everything()
    tr, va = train_test_split(dev, test_size=0.15, stratify=y[dev], random_state=SEED)
    train_ds = ArrayDataset(x[tr], y[tr], baseline_transform(size, augment, True))
    val_ds = ArrayDataset(x[va], y[va], baseline_transform(size, pre_resized=True))
    model, info = fit(make_model(), train_ds, val_ds, epochs=EPOCHS, lr=LR, batch_size=BATCH,
                      scheduler="cosine" if tricks else None,
                      class_w=class_weights(y[tr], 6) if tricks else None,
                      select_best=True, name=f"step{step}", log_file=log_file)
    proba = predict_proba(model, ArrayDataset(x[hold], y[hold],
                                              baseline_transform(size, pre_resized=True)))
    save_result("baselines", f"step{step}", summarize(y[hold], proba), description=desc,
                best_epoch=info["best_epoch"], inner_val_acc=max(
                    h["val_acc"] for h in info["history"]))
    save_proba(f"baseline_step{step}_holdout", proba)


def run_kfold_ensemble(x, y, dev, hold, log_file, k=5):
    """Step 6 - the final competition model: 5-fold DeepCNN ensemble + h-flip TTA."""
    x = resize_images(x, 96)
    seed_everything()
    hold_ds = ArrayDataset(x[hold], y[hold], baseline_transform(96, pre_resized=True))
    skf = StratifiedKFold(k, shuffle=True, random_state=SEED)
    proba_sum = np.zeros((len(hold), 6))
    for fold, (tr_i, va_i) in enumerate(skf.split(dev, y[dev]), start=1):
        tr, va = dev[tr_i], dev[va_i]
        model, _ = fit(DeepCNN(), ArrayDataset(x[tr], y[tr], baseline_transform(96, True, True)),
                       ArrayDataset(x[va], y[va], baseline_transform(96, pre_resized=True)),
                       epochs=EPOCHS, lr=LR, batch_size=BATCH, scheduler="cosine",
                       class_w=class_weights(y[tr], 6), select_best=True,
                       name=f"step6-fold{fold}", log_file=log_file)
        proba_sum += predict_proba(model, hold_ds, tta_hflip=True)
    proba = proba_sum / k
    save_result("baselines", "step6", summarize(y[hold], proba),
                description="DeepCNN 5-fold ensemble + TTA (final competition model)")
    save_proba("baseline_step6_holdout", proba)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6])
    args = ap.parse_args()

    x, y, _ = load_train()
    dev, hold = holdout_split(y)
    log_file = RESULTS_DIR / "baselines_history.jsonl"
    if set(args.steps) == set(STEPS) | {6}:
        log_file.unlink(missing_ok=True)
    for s in args.steps:
        if s == 6:
            run_kfold_ensemble(x, y, dev, hold, log_file)
        else:
            run_single(s, x, y, dev, hold, log_file)


if __name__ == "__main__":
    main()
