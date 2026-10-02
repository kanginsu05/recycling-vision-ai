"""Experiment 5 - Kaggle submission file from a model's saved test-set probabilities.

    python -m experiments.05_make_submission --final ft_efficientnet_b0
"""
import argparse

import numpy as np
import pandas as pd

from src.config import CLASSES, ROOT, RESULTS_DIR
from src.data import load_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default="ft_efficientnet_b0")
    args = ap.parse_args()

    _, names = load_test()
    proba = np.load(RESULTS_DIR / "proba" / f"{args.final}_test.npy")
    sub = pd.DataFrame({"ImageId": names, "Label": [CLASSES[i] for i in proba.argmax(1)]})
    out = ROOT / "outputs" / f"submission_{args.final}.csv"
    out.parent.mkdir(exist_ok=True)
    sub.to_csv(out, index=False)
    print(f"{out} ({len(sub)} rows)")
    print(sub["Label"].value_counts().to_string())


if __name__ == "__main__":
    main()
