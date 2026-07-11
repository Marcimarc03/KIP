#!/usr/bin/env python
"""Normalisierte Konfusionsmatrix aus einer COCO-Predictions-JSON + Test-GT.

Box-IoU-Matching (>= --iou) zwischen Vorhersagen (Score >= --score-thr) und
Ground Truth; erfasst Fehlklassifikationen sowie verfehlte GT (-> \"background\"-
Zeile) und Falsch-Positive (-> \"background\"-Spalte). Nur die im Test real
vorhandenen Klassen. Kein Neutraining noetig.

Beispiel:
    python scripts/render_confusion.py \
      --pred results/component_benchmark/<run>/predictions/coco_predictions.json \
      --gt data/coco_converted/test.json \
      --score-thr 0.3 --out figures/confusion_yolo.png
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _iou(a, b):  # boxes [x, y, w, h]
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2])
    y2 = min(a[1] + a[3], b[1] + b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 0 else 0.0


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--title", default="Confusion Matrix (normalisiert)")
    return p.parse_args()


def main():
    a = _parse()
    gt = json.load(open(a.gt))
    catname = {c["id"]: c["name"] for c in gt["categories"]}

    # GT je Bild + real vorhandene Klassen
    gt_by_img = {}
    real = sorted({ann["category_id"] for ann in gt["annotations"]})
    for ann in gt["annotations"]:
        gt_by_img.setdefault(ann["image_id"], []).append((ann["category_id"], ann["bbox"]))

    names = [catname[c] for c in real] + ["background"]
    idx = {c: i for i, c in enumerate(real)}
    BG = len(real)
    n = len(names)
    M = np.zeros((n, n))  # M[predicted, true]

    preds_by_img = {}
    for pr in json.load(open(a.pred)):
        if pr.get("score", 1.0) >= a.score_thr:
            preds_by_img.setdefault(pr["image_id"], []).append((pr["category_id"], pr["bbox"], pr.get("score", 1.0)))

    for iid, gts in gt_by_img.items():
        preds = sorted(preds_by_img.get(iid, []), key=lambda x: -x[2])
        used = set()
        for pcat, pbox, _ in preds:
            best, bj = a.iou, -1
            for j, (gcat, gbox) in enumerate(gts):
                if j in used:
                    continue
                v = _iou(pbox, gbox)
                if v >= best:
                    best, bj = v, j
            r = idx.get(pcat, BG)
            if bj >= 0:
                used.add(bj)
                M[r, idx[gts[bj][0]]] += 1
            else:
                if pcat in idx:
                    M[idx[pcat], BG] += 1  # Falsch-Positiv
        for j, (gcat, gbox) in enumerate(gts):
            if j not in used:
                M[BG, idx[gcat]] += 1      # verfehlt (background vorhergesagt)

    col = M.sum(axis=0, keepdims=True)
    Mn = np.divide(M, col, out=np.zeros_like(M), where=col > 0)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("True (Ground Truth)"); ax.set_ylabel("Predicted")
    ax.set_title(a.title, fontsize=11)
    for i in range(n):
        for j in range(n):
            v = Mn[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if v > 0.5 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(a.out, dpi=200)
    print(f"gespeichert: {a.out}  (Klassen: {names})")


if __name__ == "__main__":
    main()
