"""Evaluation helpers and result bookkeeping."""
import json

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)

from .config import CLASSES, RESULTS_DIR


def summarize(y_true, proba):
    y_pred = proba.argmax(1)
    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4,
                                   output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class": {c: {k: float(report[c][k]) for k in ("precision", "recall", "f1-score")}
                      for c in CLASSES},
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def save_result(experiment, name, metrics, **extra):
    """Append/replace one model's hold-out metrics in results/<experiment>.json."""
    path = RESULTS_DIR / f"{experiment}.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[name] = {**extra, **metrics}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {name}: hold-out acc {metrics['accuracy']:.4f} | "
          f"macro-F1 {metrics['macro_f1']:.4f}  (saved to {path.name})", flush=True)


def save_proba(name, proba):
    np.save(RESULTS_DIR / "proba" / f"{name}.npy", proba)


(RESULTS_DIR / "proba").mkdir(exist_ok=True)
