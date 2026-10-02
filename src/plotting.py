"""Shared matplotlib style for README figures (validated reference palette)."""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

from .config import ASSETS_DIR

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"        # categorical slot 1
ORANGE = "#eb6834"      # categorical slot 2
GRAY = "#c3c2b7"        # de-emphasised marks
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]


def setup():
    names = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in ("Malgun Gothic", "AppleGothic", "NanumGothic",
                               "Noto Sans CJK KR") if f in names), "DejaVu Sans")
    mpl.rcParams.update({
        "font.family": family,
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
        "font.size": 10.5,
    })


def save(fig, name, dpi=160):
    path = ASSETS_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] {path.relative_to(ASSETS_DIR.parent)}")
