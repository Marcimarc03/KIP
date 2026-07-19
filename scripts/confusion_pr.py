#!/usr/bin/env python
"""Konfusionsmatrix + echte Precision/Recall/F1/IoU je Klasse aus COCO-Predictions.

Matcht Predictions <-> GT per Box-IoU (>= --iou) bei Konfidenz >= --conf
(Standard-Detektions-Konfusionsmatrix). Ersetzt die AR-Proxies des mAP-Evaluators
durch echte, schwellwertbasierte P/R/F1 + mittlere Box-IoU der Treffer.

Usage:
  python scripts/confusion_pr.py <run_dir> [--conf 0.25] [--iou 0.5] [--out figures/confusion_<run>.png]
    run_dir enthaelt predictions/coco_predictions.json ; GT = data/coco_converted/test.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GT_JSON = ROOT / "data" / "coco_converted" / "test.json"


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gt = json.load(open(GT_JSON))
    cats = {c["id"]: c["name"] for c in gt["categories"]}
    present = sorted({a["category_id"] for a in gt["annotations"]})   # nur real getestete Klassen
    names = [cats[c] for c in present]
    idx = {c: i for i, c in enumerate(present)}
    K = len(present)

    gt_by_img = defaultdict(list)
    for a in gt["annotations"]:
        gt_by_img[a["image_id"]].append((a["category_id"], a["bbox"]))

    preds = json.loads((Path(args.run_dir) / "predictions" / "coco_predictions.json").read_text())
    pred_by_img = defaultdict(list)
    for p in preds:
        if p["score"] >= args.conf:
            pred_by_img[p["image_id"]].append((p["category_id"], p["bbox"], p["score"]))

    # Konfusionsmatrix (K+1)x(K+1); letzte Zeile/Spalte = Hintergrund (FP / FN)
    M = np.zeros((K + 1, K + 1), dtype=int)
    ious = []
    for img in gt["images"]:
        gts = gt_by_img.get(img["id"], [])
        prs = sorted(pred_by_img.get(img["id"], []), key=lambda x: -x[2])
        matched_gt = set()
        for pc, pb, _ps in prs:
            best_iou, best_gi = 0.0, -1
            for gi, (gc, gb) in enumerate(gts):
                if gi in matched_gt:
                    continue
                v = box_iou(pb, gb)
                if v > best_iou:
                    best_iou, best_gi = v, gi
            if best_gi >= 0 and best_iou >= args.iou:
                gc = gts[best_gi][0]
                M[idx[gc], idx[pc]] += 1
                matched_gt.add(best_gi)
                if pc == gc:
                    ious.append(best_iou)
            else:
                M[K, idx[pc]] += 1          # False Positive
        for gi, (gc, gb) in enumerate(gts):
            if gi not in matched_gt:
                M[idx[gc], K] += 1          # False Negative (verpasstes GT)

    # P/R/F1 je Klasse aus der Matrix
    print(f"\nRun: {Path(args.run_dir).name}  (conf>={args.conf}, IoU>={args.iou})")
    print(f"{'Klasse':22}{'P':>8}{'R':>8}{'F1':>8}")
    print("-" * 46)
    ps, rs, fs = [], [], []
    for c in present:
        i = idx[c]
        tp = M[i, i]
        fp = M[:, i].sum() - tp
        fn = M[i, :].sum() - tp
        P = tp / (tp + fp) if tp + fp > 0 else 0.0
        R = tp / (tp + fn) if tp + fn > 0 else 0.0
        F1 = 2 * P * R / (P + R) if P + R > 0 else 0.0
        ps.append(P); rs.append(R); fs.append(F1)
        print(f"{cats[c]:22}{P:8.3f}{R:8.3f}{F1:8.3f}")
    print("-" * 46)
    print(f"{'Mittel (macro)':22}{np.mean(ps):8.3f}{np.mean(rs):8.3f}{np.mean(fs):8.3f}")
    print(f"mittlere Box-IoU der Treffer: {np.mean(ious):.3f}" if ious else "keine Treffer")

    # Konfusionsmatrix als PNG
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = names + ["Hintergrund"]
    fig, ax = plt.subplots(figsize=(1.05 * len(labels) + 2, 0.95 * len(labels) + 2))
    ax.imshow(M, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Vorhergesagt"); ax.set_ylabel("Ground Truth")
    thr = M.max() / 2 if M.max() > 0 else 1
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, M[i, j], ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > thr else "black")
    ax.set_title(f"Konfusionsmatrix (conf$\\geq${args.conf}, IoU$\\geq${args.iou})", fontsize=10)
    plt.tight_layout()
    out = args.out or f"figures/confusion_{Path(args.run_dir).name}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("gespeichert:", out)


if __name__ == "__main__":
    main()
