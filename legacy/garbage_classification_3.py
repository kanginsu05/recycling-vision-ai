"""
2026-ai-34 캐글 챌린지 — 점수 개선용 "최종 모델" 스크립트 (구조적 변화 버전)

garbage_classification.py 는 보고서용 5단계 실험(MLP -> CNN -> ... 한 번에
하나씩 바꾸기)을 위한 것이고, 이 스크립트는 그 실험들과는 별도로 실제
리더보드 점수를 더 올리기 위한 "최종 제출용" 모델입니다. 기존 실험 결과를
건드리지 않기 위해 파일을 분리했습니다.

이번에 적용한 구조적 변화 2가지:
1) 해상도 확대 (64x64 -> 96x96) + conv 블록 추가 (128 -> 256채널, 4블록)
   - 지금까지 쓰던 64x64 는 원본(512x384)보다 정보 손실이 커서, glass/plastic,
     paper/cardboard 처럼 표면 질감으로 구분해야 하는 클래스에서 한계가 있었음.
     해상도를 높이고 그만큼 conv 블록도 하나 더 늘려서 표현력을 맞춤.
2) 5-Fold 교차검증 앙상블
   - 지금까지는 15%(322장)만 validation 으로 썼는데, 데이터가 2,151장뿐이라
     이 하나의 분리 방식에 결과가 크게 좌우됐음(로그에서 val acc가 epoch마다
     크게 출렁였던 것도 이 때문). 데이터를 5등분해서 5개 모델을 학습하고,
     제출할 땐 5개 모델의 예측을 평균(앙상블)내서 특정 분리 방식에 대한 운의
     영향을 줄이고 데이터를 더 알뜰하게 씀.

기존에 효과가 확인된 것들(클래스 가중치, Cosine LR 스케줄러, TTA)도 그대로
가져와서 함께 적용했습니다.

실행 시간 안내: 해상도가 올라가고(64x64 -> 96x96, 픽셀 수 2.25배) 블록도
늘었고 5-fold 이므로, 이전 스크립트보다 훨씬 오래 걸립니다(수 배 이상).
너무 오래 걸리면 아래 K(fold 수)나 EPOCHS 를 줄여서 실행하세요.
"""

# %% [1] 설정 ----------------------------------------------------------------
import random, copy
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

DATA = Path("/kaggle/input/garbage-classification-challenge")
if not DATA.exists():
    here = Path(__file__).resolve().parent
    for cand in [here, here / "kaggle_upload"]:
        if (cand / "train").exists():
            DATA = cand
            break
    else:
        raise FileNotFoundError("train/ 폴더를 찾지 못했습니다.")
print("데이터 경로:", DATA)

TRAIN_DIR = DATA / "train"
TEST_DIR  = DATA / "test"

BATCH = 64
K = 5              # fold 개수 (너무 오래 걸리면 3으로 줄여도 됨)
EPOCHS = 30         # fold 당 epoch 수 (너무 오래 걸리면 20으로 줄여도 됨)
IMG_SIZE = 96       # 개선 1: 64 -> 96

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

def savefig(name):
    plt.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    print(f"  [그림 저장] {FIG_DIR / name}")


# %% [2] 데이터 준비 ----------------------------------------------------------
eval_tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])
aug_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(15),
    T.ToTensor(),
])

base = ImageFolder(TRAIN_DIR, transform=eval_tf)
classes = base.classes
all_labels = np.array([y for _, y in base.samples])
print("클래스:", classes)
print("전체 학습 이미지:", len(base), "장 (96x96 로 리사이즈)")


def make_fold_loaders(tr_idx, va_idx, train_transform):
    tr_base = ImageFolder(TRAIN_DIR, transform=train_transform)
    va_base = ImageFolder(TRAIN_DIR, transform=eval_tf)
    tr = DataLoader(Subset(tr_base, tr_idx), batch_size=BATCH,
                     shuffle=True, num_workers=2, drop_last=True)
    va = DataLoader(Subset(va_base, va_idx), batch_size=BATCH,
                     shuffle=False, num_workers=2)
    return tr, va


# %% [3] 학습 함수 (기존 스크립트와 동일 - class_weights, scheduler 지원) ------
def train_model(model, train_loader, val_loader, epochs=30, lr=1e-3, name="model",
                 class_weights=None, use_scheduler=True):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
                 if use_scheduler else None)
    w = class_weights.to(device) if class_weights is not None else None
    crit = nn.CrossEntropyLoss(weight=w)
    best_val, best_state, best_ep = -1.0, None, 0
    train_acc_hist, val_acc_hist = [], []

    for ep in range(1, epochs + 1):
        model.train(); s_correct = n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward(); opt.step()
            s_correct += (out.argmax(1) == y).sum().item(); n += y.size(0)
        train_acc = s_correct / n
        if scheduler is not None:
            scheduler.step()

        model.eval(); s_correct = n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                s_correct += (out.argmax(1) == y).sum().item(); n += y.size(0)
        val_acc = s_correct / n
        train_acc_hist.append(train_acc); val_acc_hist.append(val_acc)

        if val_acc > best_val:
            best_val = val_acc; best_state = copy.deepcopy(model.state_dict()); best_ep = ep

        print(f"[{name}] epoch {ep:2d}  train_acc {train_acc:.3f}  val_acc {val_acc:.3f}"
              f"{'  <- best' if best_ep == ep else ''}")

    model.load_state_dict(best_state)
    print(f">>> [{name}] best val accuracy: {best_val:.4f} (epoch {best_ep})\n")
    return model, {"best_val": best_val, "best_ep": best_ep,
                   "train_acc": train_acc_hist, "val_acc": val_acc_hist}


# %% [4] 모델 정의 — 개선 1: conv 블록 4개 (128 -> 256채널 블록 추가) ----------
class DeepCNN(nn.Module):
    def __init__(self, n_cls=6, p_drop=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),    # 96->48
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),    # 48->24
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),  # 24->12
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2), # 12->6 (새 블록)
        )
        C_LAST = 256
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(p_drop),
            nn.Linear(C_LAST, n_cls),
        )
    def forward(self, x):
        return self.head(self.features(x))


# %% [5] 개선 2: 5-Fold 교차검증 -----------------------------------------------
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)

fold_models = []
oof_pred = np.full(len(base), -1)   # out-of-fold 예측 (전체 학습 데이터 기준)

for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels), start=1):
    print(f"\n===== Fold {fold}/{K}  (train {len(tr_idx)}장 / val {len(va_idx)}장) =====")

    tr_loader, va_loader = make_fold_loaders(tr_idx.tolist(), va_idx.tolist(), aug_tf)

    # 이 fold의 학습 데이터 기준으로 클래스 가중치 계산 (trash 보정)
    fold_labels = all_labels[tr_idx]
    fold_n = np.array([Counter(fold_labels)[i] for i in range(len(classes))], dtype=np.float32)
    fold_weights = torch.tensor(fold_n.sum() / (len(classes) * fold_n), dtype=torch.float32)

    model = DeepCNN()
    model, hist = train_model(model, tr_loader, va_loader, epochs=EPOCHS,
                                name=f"Fold{fold}", class_weights=fold_weights,
                                use_scheduler=True)
    fold_models.append(model)

    # 이 fold의 validation 부분 예측을 저장해서, 나중에 전체 데이터에 대한
    # out-of-fold(OOF) 정확도를 계산한다 (322장이 아니라 2,151장 전체 기준!)
    model.eval()
    preds = []
    with torch.no_grad():
        for x, y in va_loader:
            preds.append(model(x.to(device)).argmax(1).cpu())
    oof_pred[va_idx] = torch.cat(preds).numpy()

oof_acc = (oof_pred == all_labels).mean()
print("=" * 70)
print(f"5-Fold 앙상블 OOF(out-of-fold) 정확도: {oof_acc:.4f}  "
      f"(학습 데이터 {len(all_labels)}장 전체 기준)")
print("참고: 이전 단일 분리(322장) 기준 최고 기록은 0.7391 이었음")
print("=" * 70)

print("\nOOF 기준 클래스별 성능:")
print(classification_report(all_labels, oof_pred, target_names=classes, digits=3))

cm = confusion_matrix(all_labels, oof_pred)
cm_norm = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(6)); ax.set_xticklabels(classes, rotation=45, ha="right")
ax.set_yticks(range(6)); ax.set_yticklabels(classes)
ax.set_xlabel("모델의 예측"); ax.set_ylabel("실제 정답")
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                 color="white" if cm_norm[i, j] > .5 else "black", fontsize=9)
plt.colorbar(im); plt.title("Confusion Matrix - 5-Fold OOF (행 기준 정규화)")
plt.tight_layout(); savefig("09_ensemble_confusion_matrix.png"); plt.show()


# %% [6] 제출 파일 생성 — 5개 fold 모델 앙상블 + TTA(좌우반전) ------------------
class TestImages(Dataset):
    def __init__(self, root, transform):
        self.paths = sorted(Path(root).glob("*.jpg"))
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.paths[i].name

test_loader = DataLoader(TestImages(TEST_DIR, eval_tf), batch_size=BATCH,
                           shuffle=False, num_workers=2)

for m in fold_models:
    m.eval()

rows = []
with torch.no_grad():
    for x, ids in test_loader:
        x = x.to(device)
        prob_sum = torch.zeros(x.size(0), len(classes), device=device)
        for m in fold_models:
            prob_sum += torch.softmax(m(x), dim=1)                              # 원본
            prob_sum += torch.softmax(m(torch.flip(x, dims=[3])), dim=1)        # 좌우반전 (TTA)
        prob_avg = prob_sum / (2 * len(fold_models))   # 5개 모델 x 2(원본+반전) 평균
        pred = prob_avg.argmax(1).cpu()
        rows += [(i, classes[int(c)]) for i, c in zip(ids, pred)]

sub = pd.DataFrame(rows, columns=["ImageId", "Label"])
sub.to_csv("submission_ensemble.csv", index=False)

print(f"\n제출 파일: submission_ensemble.csv (5-fold 앙상블 + TTA)")
print(sub.head(), "\n")
print("총", len(sub), "줄")
print("\n예측된 클래스 분포:")
print(sub["Label"].value_counts())
print("\n기존 submission.csv(단일 모델) 와 submission_ensemble.csv(5-fold 앙상블) 를")
print("둘 다 캐글에 제출해 보고, Public 점수가 더 높은 쪽을 최종 제출로 선택하세요.")
