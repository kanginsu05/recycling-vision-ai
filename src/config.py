"""Project-wide paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
CACHE_DIR = DATA_DIR / "cache"

RESULTS_DIR = ROOT / "results"
ASSETS_DIR = ROOT / "assets"
MODELS_DIR = ROOT / "models"

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASSES_KO = {
    "cardboard": "골판지",
    "glass": "유리",
    "metal": "금속(캔)",
    "paper": "종이",
    "plastic": "플라스틱",
    "trash": "일반쓰레기",
}

SEED = 42
HOLDOUT_FRAC = 0.2      # fixed, stratified hold-out split used by every experiment
CACHE_SIZE = 256        # images are decoded once and cached as 256x256 uint8

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

for d in (RESULTS_DIR, ASSETS_DIR, MODELS_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)
