"""Render docs/results.png from the committed summary.json files.

The figure in the README is generated, not drawn — every bar comes from
`results/<adapter>/summary.json`, scored raw by `eval/run_eval.py`.

    python results/plot_summary.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "docs", "results.png")

BASELINE = "stock-qwen25vl-3b"
CHAMPION = "eyoun-soup61"

LABELS = {
    "arocrbench_patsocr": "patsocr",
    "arocrbench_arabicocr": "arabicocr",
    "arocrbench_synthesizear": "synthesizear",
    "arocrbench_isippt": "isippt",
    "arocrbench_hindawi": "hindawi",
    "nakba_test": "nakba (newspaper)",
    "sedra_handwritten": "sedra (handwriting)",
    "arocrbench_historyar": "historyar",
    "misraj_dococr": "misraj_dococr",
    "arocrbench_khattparagraph": "khatt (handwritten ¶)",
}

INK, MUTED, GRID = "#1c1917", "#78716c", "#e7e5e4"
STOCK, WIN, REGRESSION = "#d6d3d1", "#0f766e", "#b45309"


def load(name):
    with open(os.path.join(HERE, name, "summary.json"), encoding="utf-8") as fh:
        return json.load(fh)


def main():
    stock, tuned = load(BASELINE), load(CHAMPION)
    suites = sorted(LABELS, key=lambda s: -tuned[s]["cer"])

    y = np.arange(len(suites))
    h = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.barh(y + h / 2, [stock[s]["cer"] for s in suites], h,
            color=STOCK, zorder=3)
    ax.barh(y - h / 2, [tuned[s]["cer"] for s in suites], h,
            color=[REGRESSION if tuned[s]["cer"] > stock[s]["cer"] else WIN for s in suites],
            zorder=3)

    for i, s in enumerate(suites):
        ax.text(tuned[s]["cer"] * 1.12, i - h / 2, f"{tuned[s]['cer']:.3f}",
                va="center", fontsize=8, color=INK, zorder=4)
        ax.text(stock[s]["cer"] * 1.12, i + h / 2, f"{stock[s]['cer']:.3f}",
                va="center", fontsize=8, color=MUTED, zorder=4)

    ax.set_xscale("log")
    ax.set_xlim(0.02, 22)
    ax.set_yticks(y, [LABELS[s] for s in suites], fontsize=9.5, color=INK)
    ax.set_xlabel("Character Error Rate — log scale, lower is better", fontsize=9.5, color=MUTED)
    ax.set_title("Eyoun-3B vs. the un-finetuned base model, ten Arabic OCR suites",
                 fontsize=12, color=INK, pad=14, loc="left")
    ax.annotate("khatt is the documented regression — see the model card",
                xy=(0.0, -0.135), xycoords="axes fraction", fontsize=8.5, color=REGRESSION)
    ax.legend(handles=[Patch(color=STOCK, label="base model"),
                       Patch(color=WIN, label="Eyoun-3B"),
                       Patch(color=REGRESSION, label="Eyoun-3B, regression")],
              loc="upper right", frameon=False, fontsize=9.5)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, colors=MUTED)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, facecolor="white", bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
