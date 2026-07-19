#!/usr/bin/env python3
"""Erzeugt die Stage-2-Verarbeitungskette als Abbildung (PDF + PNG).

Nutzung:
    python make_stage2_pipeline.py --out figures/stage2_pipeline.pdf
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE = "#4C72B0"
ORANGE = "#DD8452"
GREY = "#8C8C8C"
LIGHT = "#EAEFF5"


def box(ax, x, y, w, h, title, sub="", fc=LIGHT, ec=BLUE, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.2, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h * (0.62 if sub else 0.5), title,
            ha="center", va="center", fontsize=fs, weight="bold", color="#1a1a1a")
    if sub:
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                fontsize=fs - 1.4, color="#444444")


def arrow(ax, x1, y1, x2, y2, color=GREY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=11,
                                 linewidth=1.1, color=color,
                                 shrinkA=0, shrinkB=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="stage2_pipeline.pdf", type=Path)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # --- Reihe 1: gemeinsame Vorverarbeitung -------------------------------
    ax.text(0.05, 4.72, "Gemeinsame Vorverarbeitung", fontsize=8.5,
            weight="bold", color=BLUE)
    y1, h, w = 3.75, 0.78, 2.05
    box(ax, 0.05, y1, w, h, "Stage-1-Crop",
        "Spindel freigestellt\n5472$\\times$3648 px")
    box(ax, 2.55, y1, w, h, "Skalierung", "512$\\times$512 px")
    box(ax, 5.05, y1, w, h, "Tiling",
        "256 px Kacheln\n25 % Überlappung")
    box(ax, 7.55, y1, w, h, "Filter",
        "weiße Hintergrund-\nkacheln verworfen")
    for x in (2.10, 4.60, 7.10):
        arrow(ax, x, y1 + h / 2, x + 0.45, y1 + h / 2)

    # Umbruchpfeil nach unten
    arrow(ax, 9.60, y1, 9.60, 2.95, color=GREY)
    arrow(ax, 9.60, 2.95, 0.60, 2.95, color=GREY)
    arrow(ax, 0.60, 2.95, 0.60, 2.62, color=GREY)

    # --- Reihe 2: modellspezifische Kachel-Bewertung -----------------------
    ax.text(0.05, 2.42, "Kachel-Bewertung (modellspezifisch)", fontsize=8.5,
            weight="bold", color=ORANGE)
    y2, h2 = 1.30, 0.95
    box(ax, 0.05, y2, 2.30, h2, "PatchCore",
        "Distanz zum nächsten\nGut-Patch  ·  256 px",
        fc="#FDF1E7", ec=ORANGE)
    box(ax, 2.55, y2, 2.10, h2, "PaDiM",
        "Mahalanobis-\nDistanz  ·  256 px",
        fc="#FDF1E7", ec=ORANGE)
    box(ax, 4.85, y2, 2.10, h2, "ConvAE",
        "Rekonstruktions-\nfehler  ·  128 px",
        fc="#FDF1E7", ec=ORANGE)
    box(ax, 7.15, y2, 2.45, h2, "U-Net (supervised)",
        "Sigmoid-Wahrschein-\nlichkeit  ·  256 px",
        fc="#FDF1E7", ec=ORANGE)
    ax.text(4.82, 1.06, "je Kachel: Anomaliekarte $+$ Score",
            ha="center", fontsize=7.4, color="#555555", style="italic")

    # Pfeile nach unten in die Zusammenführung
    for x in (1.20, 3.60, 5.90, 8.37):
        arrow(ax, x, y2 - 0.02, x, 0.80)

    # --- Reihe 3: gemeinsame Zusammenführung -------------------------------
    y3, h3 = 0.05, 0.70
    box(ax, 1.30, y3, 7.10, h3, "Zusammenführung  (für alle Verfahren identisch)",
        "Maximum je Pixel über überlappende Kacheln  $\\rightarrow$  "
        "Gesamtkarte;   Bild-Score $=$ 0{,}98-Quantil",
        fc="#EAF0F6", ec=BLUE, fs=8.2)

    fig.tight_layout(pad=0.4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=300)
    fig.savefig(args.out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"OK: {args.out} und {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
