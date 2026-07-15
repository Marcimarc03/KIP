#!/usr/bin/env python
"""Bootstrap 95%-CI fuer Stage-1 segm mAP50 ueber die Testbilder (kein Retraining).

Quantifiziert die EVAL-Unsicherheit (kleines Testset), unabhaengig von der
Trainings-/Seed-Varianz. Resamplet die Testbilder MIT ZURUECKLEGEN; jedes gezogene
Bild bekommt eine frische image_id (sonst dedupliziert pycocotools Duplikate),
rechnet je Resample die mAP neu, und berichtet Punktwert + 95%-CI. Bei >=2 Runs
zusaetzlich der GEPAARTE Modellvergleich (gleiche Resamples -> CI der Differenz).

Usage:
  python scripts/bootstrap_stage1.py <run_dir> [<run_dir> ...] [--n-boot 1000] [--seed 0]
    run_dir enthaelt predictions/coco_predictions.json ; GT = data/coco_converted/test.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[1]
GT_JSON = ROOT / "data" / "coco_converted" / "test.json"


def _map50(gt_dict: dict, dt_anns: list) -> float:
    """In-memory COCO-Eval; gibt segm mAP50 zurueck (stdout unterdrueckt)."""
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = gt_dict
        gt.createIndex()
        if not dt_anns:
            return 0.0
        dt = gt.loadRes(dt_anns)
        ev = COCOeval(gt, dt, iouType="segm")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return float(ev.stats[1])  # stats[1] = AP @ IoU=0.50 (segm)


def _resample_dicts(draw, images, gt_by_img, dt_by_img, categories):
    """Baue GT-Dict + je-Modell dt-Listen fuer die gezogenen Bilder (frische IDs)."""
    new_images, new_gt = [], []
    new_dt = {m: [] for m in dt_by_img}
    ann_id = 1
    for k, oid in enumerate(draw):
        nid = k + 1
        im = dict(images[oid])
        im["id"] = nid
        new_images.append(im)
        for a in gt_by_img.get(oid, []):
            na = dict(a)
            na["image_id"] = nid
            na["id"] = ann_id
            ann_id += 1
            new_gt.append(na)
        for m, preds in dt_by_img.items():
            for p in preds.get(oid, []):
                q = dict(p)
                q["image_id"] = nid
                new_dt[m].append(q)
    gt_dict = {"images": new_images, "annotations": new_gt, "categories": categories}
    return gt_dict, new_dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", help="Run-Ordner mit predictions/coco_predictions.json")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gt = COCO(str(GT_JSON))
    images = {im["id"]: im for im in gt.dataset["images"]}
    categories = gt.dataset["categories"]
    img_ids = list(images.keys())
    gt_by_img = defaultdict(list)
    for a in gt.dataset["annotations"]:
        gt_by_img[a["image_id"]].append(a)

    # Predictions je Modell laden, nach Bild gruppieren
    models, dt_by_img = [], {}
    for rd in args.run_dirs:
        pj = Path(rd) / "predictions" / "coco_predictions.json"
        if not pj.exists():
            print(f"[warn] keine Predictions: {pj} -> uebersprungen")
            continue
        name = Path(rd).name
        models.append(name)
        by_img = defaultdict(list)
        for p in json.loads(pj.read_text()):
            by_img[p["image_id"]].append(p)
        dt_by_img[name] = by_img
    if not models:
        print("Keine gueltigen Runs.")
        return

    # Punktwerte auf dem vollen Testset
    full_gt = {"images": list(images.values()),
               "annotations": [a for anns in gt_by_img.values() for a in anns],
               "categories": categories}
    point = {m: _map50(full_gt, [p for ps in dt_by_img[m].values() for p in ps]) for m in models}

    # Bootstrap
    rng = np.random.default_rng(args.seed)
    boot = {m: np.empty(args.n_boot) for m in models}
    n = len(img_ids)
    idx_arr = np.array(img_ids)
    print(f"Bootstrap: {args.n_boot} Resamples ueber {n} Testbilder, {len(models)} Modell(e) ...")
    for b in range(args.n_boot):
        draw = idx_arr[rng.integers(0, n, n)]
        gt_dict, new_dt = _resample_dicts(draw, images, gt_by_img, dt_by_img, categories)
        for m in models:
            boot[m][b] = _map50(gt_dict, new_dt[m])
        if (b + 1) % 100 == 0:
            print(f"  ... {b + 1}/{args.n_boot}")

    print("\n=== segm mAP50: Punktwert + 95%-CI (Bootstrap ueber Testbilder) ===")
    for m in models:
        lo, hi = np.percentile(boot[m], [2.5, 97.5])
        print(f"  {m:48s} {point[m]:.4f}  95%-CI [{lo:.4f}, {hi:.4f}]")

    # Gepaarte Modellvergleiche (gleiche Resamples -> Differenz-CI)
    if len(models) >= 2:
        print("\n=== Gepaarte Differenzen (A - B), 95%-CI ===")
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a, bm = models[i], models[j]
                diff = boot[a] - boot[bm]
                lo, hi = np.percentile(diff, [2.5, 97.5])
                sig = "signifikant (CI schliesst 0 aus)" if (lo > 0 or hi < 0) else "NICHT unterscheidbar (CI enthaelt 0)"
                print(f"  {a}  -  {bm}:  {np.mean(diff):+.4f}  95%-CI [{lo:+.4f}, {hi:+.4f}]  -> {sig}")


if __name__ == "__main__":
    main()
