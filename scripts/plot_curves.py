#!/usr/bin/env python
"""Trainingskurven plotten — je ein repraesentativer Lauf pro Modell/Strategie.

Liest die normierten CSVs aus curves_out/ (von extract_curves.py) und erzeugt
fuer ausgewaehlte Laeufe je eine Abbildung mit zwei y-Achsen:
  links  : train_loss und val_loss (Overfitting-Diagnose)
  rechts : val_map50 (Generalisierungsleistung)

Zusaetzlich eine Vergleichsgrafik der val_map50-Kurven aller ausgewaehlten Laeufe.

Nutzung:
    python scripts/plot_curves.py
    python scripts/plot_curves.py --runs yolo_augon_realonly_20260715_223330 \
                                          yolo_augon_synthpretrain_20260715_003555
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent

# repraesentative finale Laeufe je Konfiguration (sha=deda..., volle 100 Ep. wo moeglich).
# Bei Bedarf via --runs ueberschreiben.
_DEFAULT_RUNS = [
    "yolo_augon_realonly_20260715_223330",       # YOLO11 real-only (B)
    "yolo_augon_synthpretrain_20260715_003555",  # YOLO11 synth->real (C)
    "yolo26_augon_20260716_042439",              # YOLO26 real-only
    "yolo26_augon_synthpretrain_20260720_054954",# YOLO26 synth->real
]


def _read(csv_path: Path) -> dict[str, list[float]]:
    cols: dict[str, list] = {}
    for r in csv.DictReader(open(csv_path)):
        for k, v in r.items():
            try:
                cols.setdefault(k, []).append(float(v))
            except (TypeError, ValueError):
                cols.setdefault(k, []).append(None)
    return cols


def _plot_single(name: str, d: dict, out: Path) -> None:
    ep = d.get("epoch", [])
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel("Epoche")
    ax1.set_ylabel("Loss")
    for key, lab, style in [("train_loss", "train seg-loss", "-"),
                            ("val_loss", "val seg-loss", "--")]:
        y = d.get(key, [])
        pts = [(e, v) for e, v in zip(ep, y) if v is not None]
        if pts:
            xs, ys = zip(*pts)
            ax1.plot(xs, ys, style, label=lab)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("val mAP@50 (Maske)")
    y = d.get("val_map50", [])
    pts = [(e, v) for e, v in zip(ep, y) if v is not None]
    if pts:
        xs, ys = zip(*pts)
        ax2.plot(xs, ys, "-", color="tab:green", label="val mAP@50")
        bi = max(range(len(ys)), key=lambda i: ys[i])
        ax2.scatter([xs[bi]], [ys[bi]], color="tab:green", zorder=5)
        ax2.annotate(f"best {ys[bi]:.3f}\\n@ep {int(xs[bi])}",
                     (xs[bi], ys[bi]), textcoords="offset points",
                     xytext=(6, -18), fontsize=8, color="tab:green")
    ax2.legend(loc="lower right", fontsize=9)

    plt.title(name, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"curve_{name}.png", dpi=130)
    plt.close(fig)


def _plot_compare(runs: dict[str, dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, d in runs.items():
        ep, y = d.get("epoch", []), d.get("val_map50", [])
        pts = [(e, v) for e, v in zip(ep, y) if v is not None]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, label=name, alpha=0.85)
    ax.set_xlabel("Epoche")
    ax.set_ylabel("val mAP@50 (Maske)")
    ax.set_title("Validierungs-mAP über das Training — Vergleich")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "curve_compare_valmap.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", default=str(_ROOT / "curves_out"))
    ap.add_argument("--out", default=str(_ROOT / "figures" / "curves"))
    ap.add_argument("--runs", nargs="*", default=_DEFAULT_RUNS)
    args = ap.parse_args()

    cdir = Path(args.curves)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    loaded = {}
    for name in args.runs:
        p = cdir / f"{name}.csv"
        if not p.exists():
            print(f"  fehlt: {p}  (mit extract_curves.py erzeugt?)")
            continue
        d = _read(p)
        loaded[name] = d
        _plot_single(name, d, out)
        print(f"  geplottet: curve_{name}.png")

    if len(loaded) > 1:
        _plot_compare(loaded, out)
        print("  geplottet: curve_compare_valmap.png")

    print(f"\nAbbildungen in: {out}")


if __name__ == "__main__":
    main()