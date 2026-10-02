"""Experiment 2 - Which pretrained backbone represents recycling images best?

Every backbone is frozen and used as a feature extractor (penultimate-layer embeddings).
A logistic-regression head is tuned with 5-fold CV on the dev split and evaluated once
on the hold-out split. This is a cheap, fair way to compare backbones on a CPU before
spending hours fine-tuning one of them.

    python -m experiments.02_backbone_probe [--models resnet50 convnext_tiny ...]
"""
import argparse
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CACHE_DIR, SEED
from src.data import ArrayDataset, holdout_split, load_test, load_train, pretrained_transform
from src.engine import DEVICE, seed_everything
from src.metrics import save_proba, save_result, summarize
from src.models import BACKBONES, build_transfer_model


@torch.inference_mode()
def extract(model, images, batch_size=32):
    ds = ArrayDataset(images, transform=pretrained_transform(224, augment=False))
    feats = []
    for i in range(0, len(ds), batch_size):
        x = torch.stack([ds[j][0] for j in range(i, min(i + batch_size, len(ds)))])
        feats.append(model(x.to(DEVICE)).cpu())
    return torch.cat(feats).numpy()


def get_features(name, x_train, x_test):
    path = CACHE_DIR / f"feats_{name}.npz"
    if path.exists():
        f = np.load(path)
        return f["train"], f["test"]
    model = build_transfer_model(name, feature_only=True).eval().to(DEVICE)
    t0 = time.time()
    f_train, f_test = extract(model, x_train), extract(model, x_test)
    print(f"[{name}] features {f_train.shape} extracted in {time.time() - t0:.0f}s", flush=True)
    np.savez(path, train=f_train, test=f_test)
    return f_train, f_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(BACKBONES))
    args = ap.parse_args()

    seed_everything()
    x, y, _ = load_train()
    x_test, _ = load_test()
    dev, hold = holdout_split(y)
    print(f"dev {len(dev)} / hold-out {len(hold)} / test {len(x_test)}", flush=True)

    for name in args.models:
        f_train, f_test = get_features(name, x, x_test)
        clf = GridSearchCV(
            make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=3000, random_state=SEED)),
            param_grid={"logisticregression__C": [0.01, 0.03, 0.1, 0.3, 1.0],
                        "logisticregression__class_weight": [None, "balanced"]},
            cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
            scoring="accuracy", n_jobs=1)
        clf.fit(f_train[dev], y[dev])
        proba = clf.predict_proba(f_train[hold])
        m = summarize(y[hold], proba)
        save_result("probe", name, m, cv_acc=float(clf.best_score_),
                    best_params={k.split("__")[1]: v for k, v in clf.best_params_.items()},
                    feat_dim=int(f_train.shape[1]))
        save_proba(f"probe_{name}_holdout", proba)
        save_proba(f"probe_{name}_test", clf.predict_proba(f_test))


if __name__ == "__main__":
    main()
