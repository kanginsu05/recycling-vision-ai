"""Experiment 4 - Figures and error analysis for the README.

    python -m experiments.04_analyze --final ft_efficientnet_b0
"""
import argparse
import json
import shutil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

from src.config import ASSETS_DIR, CLASSES, CLASSES_KO, RESULTS_DIR, TRAIN_DIR
from src.data import holdout_split, load_train
from src.inference import Predictor
from src import plotting as P

PUBLIC_LB = {  # Kaggle public leaderboard scores recorded during the competition
    "step0": 0.5373, "step1": 0.6607, "step2": 0.6875, "step3": 0.7500,
    "step4": 0.7500, "step6": 0.7500,
}
STEP_LABELS = {
    "step0": "MLP (기본 코드)", "step1": "CNN 3블록", "step2": "+ 데이터 증강",
    "step3": "+ BatchNorm", "step4": "+ 클래스 가중치·Cosine LR",
    "step5": "DeepCNN 4블록 @96px", "step6": "DeepCNN 5-Fold 앙상블+TTA",
}
BACKBONE_LABELS = {
    "mobilenet_v3_large": "MobileNetV3-L", "efficientnet_b0": "EfficientNet-B0",
    "efficientnet_v2_s": "EfficientNetV2-S", "resnet50": "ResNet-50",
    "convnext_tiny": "ConvNeXt-T", "swin_t": "Swin-T",
}


def load(name):
    p = RESULTS_DIR / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# ---------------------------------------------------------------- figures
def fig_class_distribution(y):
    counts = np.bincount(y, minlength=6)
    order = np.argsort(counts)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    labels = [f"{CLASSES_KO[CLASSES[i]]} ({CLASSES[i]})" for i in order]
    ax.barh(labels, counts[order], color=P.BLUE, height=0.55)
    for i, v in enumerate(counts[order]):
        ax.text(v + 6, i, f"{v} ({v / counts.sum():.1%})", va="center", color=P.INK_2,
                fontsize=9.5)
    ax.set_xlim(0, counts.max() * 1.22)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("학습 이미지 수")
    ax.set_title(f"클래스 분포 — 학습 이미지 {counts.sum():,}장, trash는 5%뿐")
    P.save(fig, "class_distribution.png")


def fig_journey(base, probe, ft, final):
    rows = [(STEP_LABELS[k], base[k]["accuracy"], P.GRAY) for k in STEP_LABELS if k in base]
    if probe:
        best = max(probe, key=lambda k: probe[k]["cv_acc"])
        rows.append((f"전이학습: {BACKBONE_LABELS[best]} 특징 + 로지스틱 회귀",
                     probe[best]["accuracy"], P.BLUE))
    if final in ft:
        name = BACKBONE_LABELS[final.removeprefix("ft_")]
        rows.append((f"전이학습: {name} 파인튜닝 (최종)", ft[final]["accuracy"], P.BLUE))
    labels, accs, colors = zip(*rows)
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(rows) + 1.2))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, accs, color=colors, height=0.6)
    for yi, a in zip(y, accs):
        ax.text(a + 0.006, yi, f"{a:.3f}", va="center", color=P.INK, fontsize=9.5)
    ax.set_yticks(y, labels)
    ax.set_xlim(0.4, 1.0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Hold-out 정확도 (431장, 학습·모델 선택에 쓰지 않은 데이터)")
    ax.set_title("개선 과정 — 같은 Hold-out 데이터로 다시 평가")
    P.save(fig, "journey.png")


def fig_probe(probe):
    lat = json.loads((RESULTS_DIR / "cpu_latency.json").read_text())
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, v in probe.items():
        ax.scatter(lat[k], v["accuracy"], s=70, color=P.BLUE, edgecolor=P.SURFACE,
                   linewidth=2, zorder=3)
        ax.annotate(BACKBONE_LABELS[k], (lat[k], v["accuracy"]), xytext=(7, -3),
                    textcoords="offset points", color=P.INK_2, fontsize=9.5)
    ax.set_xlabel("CPU 추론 시간 (ms/이미지, i5-1035G7)")
    ax.set_ylabel("Hold-out 정확도")
    ax.set_xlim(0, max(lat.values()) * 1.3)
    ax.set_title("사전학습 백본 비교 — 가중치 고정 + 로지스틱 회귀")
    P.save(fig, "backbone_probe.png")


def fig_confusion(before, after, before_title, after_title):
    cmap = LinearSegmentedColormap.from_list("blue", P.BLUE_RAMP)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, res, title in zip(axes, (before, after), (before_title, after_title)):
        cm = np.array(res["confusion_matrix"], dtype=float)
        cm = cm / cm.sum(1, keepdims=True)
        ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
        ax.grid(False)
        ax.set_xticks(range(6), CLASSES, rotation=40, ha="right")
        ax.set_yticks(range(6), CLASSES)
        ax.set_xlabel("예측"); ax.set_ylabel("정답")
        for i in range(6):
            for j in range(6):
                if cm[i, j] >= 0.005:
                    ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                            color="white" if cm[i, j] > 0.55 else P.INK)
        ax.set_title(f"{title}\n정확도 {res['accuracy']:.3f}", fontsize=11.5)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("혼동 행렬 (행 기준 정규화)", x=0.02, ha="left", fontsize=13,
                 fontweight="bold", color=P.INK)
    fig.tight_layout()
    P.save(fig, "confusion_matrix.png")


def fig_per_class(before, after, before_label, after_label):
    f_b = [before["per_class"][c]["f1-score"] for c in CLASSES]
    f_a = [after["per_class"][c]["f1-score"] for c in CLASSES]
    x = np.arange(6)
    w = 0.36
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x - w / 2 - 0.01, f_b, w, color=P.GRAY, label=before_label)
    ax.bar(x + w / 2 + 0.01, f_a, w, color=P.BLUE, label=after_label)
    for xi, a in zip(x, f_a):
        ax.text(xi + w / 2 + 0.01, a + 0.015, f"{a:.2f}", ha="center", color=P.INK,
                fontsize=8.5)
    ax.set_xticks(x, [f"{CLASSES_KO[c]}\n{c}" for c in CLASSES])
    ax.set_ylim(0, 1.08)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("F1-score")
    ax.legend(loc="lower left", ncols=2, bbox_to_anchor=(0, 1.0), fontsize=9.5)
    ax.set_title("클래스별 F1-score", pad=30)
    P.save(fig, "per_class_f1.png")


def fig_learning_curve(final):
    path = RESULTS_DIR / f"{final}_history.jsonl"
    if not path.exists():
        return
    h = [json.loads(l) for l in path.read_text().splitlines()]
    ep = [r["epoch"] for r in h]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(ep, [r["train_acc"] for r in h], color=P.GRAY, lw=2, marker="o", ms=5,
            label="학습 정확도 (증강 적용)")
    ax.plot(ep, [r["val_acc"] for r in h], color=P.BLUE, lw=2, marker="o", ms=5,
            label="Hold-out 정확도 (모니터링만)")
    ax.annotate(f"{h[-1]['val_acc']:.3f}", (ep[-1], h[-1]["val_acc"]), xytext=(6, 4),
                textcoords="offset points", color=P.INK, fontsize=9.5)
    ax.set_xticks(ep)
    ax.set_xlabel("Epoch"); ax.set_ylabel("정확도")
    ax.legend(loc="lower right")
    ax.set_title("파인튜닝 학습 곡선")
    P.save(fig, "learning_curve.png")


def _grid(items, predictor, title, fname, ncols=4):
    """items: list of (image uint8, true label, pred label, prob)."""
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(ncols * 3.6, nrows * 2.0))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for k, (img, t, p, prob) in enumerate(items):
        r, c = divmod(k, ncols)
        _, cam_img = predictor.explain(img)
        a0, a1 = axes[r, 2 * c], axes[r, 2 * c + 1]
        a0.imshow(img); a1.imshow(cam_img)
        ok = t == p
        a0.set_title(f"정답 {t}", fontsize=9, color=P.INK_2, loc="left")
        a1.set_title(f"예측 {p} {prob:.0%}", fontsize=9, loc="left",
                     color=P.INK if ok else "#d03b3b")
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=P.INK)
    fig.tight_layout()
    P.save(fig, fname)


def fig_gradcam(x, y, hold, proba, predictor):
    pred = proba.argmax(1)
    rng = np.random.default_rng(0)
    correct, wrong = [], []
    for ci in range(6):
        idx = [i for i, h in enumerate(hold) if y[h] == ci and pred[i] == ci]
        for i in rng.choice(idx, 2, replace=False):
            correct.append((x[hold[i]], CLASSES[ci], CLASSES[pred[i]], proba[i].max()))
    for i in np.argsort(-proba.max(1)):
        if pred[i] != y[hold[i]]:
            wrong.append((x[hold[i]], CLASSES[y[hold[i]]], CLASSES[pred[i]], proba[i].max()))
    _grid(correct, predictor, "Grad-CAM — 맞힌 예시 (클래스별 2장)", "gradcam_correct.png")
    if wrong:
        _grid(wrong[:8], predictor, "Grad-CAM — 틀린 예시 (확신이 높았던 순)",
              "gradcam_errors.png")


def export_examples(paths, y, hold, proba):
    """Copy one correctly classified hold-out image per class for the demo app."""
    out = ASSETS_DIR / "examples"
    out.mkdir(exist_ok=True)
    pred = proba.argmax(1)
    for ci, c in enumerate(CLASSES):
        i = next(i for i, h in enumerate(hold) if y[h] == ci and pred[i] == ci)
        img = Image.open(TRAIN_DIR / paths[hold[i]]).convert("RGB")
        img.resize((384, 288)).save(out / f"{c}.jpg", quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default="ft_efficientnet_b0")
    args = ap.parse_args()
    P.setup()

    x, y, paths = load_train()
    dev, hold = holdout_split(y)
    base, probe, ft = load("baselines"), load("probe"), load("finetune")

    fig_class_distribution(y)
    if base:
        fig_journey(base, probe, ft, args.final)
    if probe:
        fig_probe(probe)
    if args.final in ft:
        final = ft[args.final]
        fig_learning_curve(args.final)
        if "step6" in base:
            fig_confusion(base["step6"], final, "대회 최종 모델 (DeepCNN 앙상블)",
                          "전이학습 파인튜닝 (최종)")
            fig_per_class(base["step6"], final, "대회 최종 모델 (DeepCNN 앙상블)",
                          "전이학습 파인튜닝 (최종)")
        proba = np.load(RESULTS_DIR / "proba" / f"{args.final}_holdout.npy")
        ckpt = ASSETS_DIR.parent / "models" / f"{args.final}.pt"
        shutil.copy(ckpt, ASSETS_DIR.parent / "models" / "recycle_classifier.pt")
        predictor = Predictor()
        fig_gradcam(x, y, hold, proba, predictor)
        export_examples(paths, y, hold, proba)


if __name__ == "__main__":
    main()
