#!/usr/bin/env python
"""Stage-2 metrics.json aus gespeicherten Predictions rekonstruieren.

Anwendungsfall: Ein Stage-2-Lauf hat scores.csv und amaps.npz geschrieben,
die metrics.json fehlt aber. Statt den Lauf zu wiederholen, werden hier
dieselben Metrik-Funktionen (kip.metrics.anomaly) auf die gespeicherten
Daten angewendet -- exakt die Rechnung aus kip/stage2/runner.py.

Rekonstruiert werden gepoolte UND per-fold Metriken; config.yaml wird
ebenfalls neu geschrieben (aus dem Run-Namen abgeleitet), damit der Lauf
schema-konform ist.

Nutzung:
    python scripts/eval_stage2_from_predictions.py <run_dir> [--dry-run]
    python scripts/eval_stage2_from_predictions.py results/defect_detection/*_20260719_*
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kip.metrics.anomaly import aupro, best_f1, dice_iou, image_auroc, pixel_auroc

_RUN_RE = re.compile(r"^(?P<method>\w+?)_(?P<split>loto|fixed|gkf)_"
                     r"aug(?P<aug>on|off)_seed(?P<seed>\d+)_")


def _meta_from_name(name: str) -> dict:
    m = _RUN_RE.match(name)
    if not m:
        raise ValueError(f"Run-Name nicht parsebar: {name}")
    return {"method": m["method"], "split": m["split"],
            "augmentation": m["aug"] == "on", "seed": int(m["seed"])}


def rebuild(run_dir: Path, dry_run: bool = False, force: bool = False) -> dict | None:
    pred = run_dir / "predictions"
    scores_csv, amaps_npz = pred / "scores.csv", pred / "amaps.npz"
    out = run_dir / "metrics.json"

    if not (scores_csv.exists() and amaps_npz.exists()):
        print(f"  SKIP {run_dir.name}: predictions unvollstaendig")
        return None
    if out.exists() and not (force or dry_run):
        print(f"  SKIP {run_dir.name}: metrics.json existiert (--force zum Ueberschreiben)")
        return None

    meta = _meta_from_name(run_dir.name)
    df = pd.read_csv(scores_csv)
    npz = np.load(amaps_npz)
    amaps, gts = npz["amaps"], npz["gts"]

    labels = df["label"].to_numpy()
    scores = df["norm_score"].to_numpy()

    # --- per-fold ---
    per_fold = []
    for fold in df["fold"].unique():
        idx = np.where(df["fold"].to_numpy() == fold)[0]
        f_lab, f_sc = labels[idx], scores[idx]
        f_gts, f_amaps = [gts[i] for i in idx], [amaps[i] for i in idx]
        f_dp = [(g, a) for g, a in zip(f_gts, f_amaps) if g.max() > 0]
        per_fold.append({
            "fold": str(fold),
            "held_out_tools": sorted(set(df["tool"].to_numpy()[idx].tolist())),
            "n_test": int(len(idx)),
            "n_defect": int(f_lab.sum()),
            "image_auroc": image_auroc(f_lab, f_sc),
            "pixel_auroc": (pixel_auroc(np.concatenate([g.ravel() for g in f_gts]),
                                        np.concatenate([a.ravel() for a in f_amaps]))
                            if f_gts else None),
            "pixel_aupro": (aupro([g for g, _ in f_dp], [a for _, a in f_dp])
                            if f_dp else None),
        })

    # --- pooled (identisch zu runner.py) ---
    img_auroc = image_auroc(labels, scores)
    f1, thr = best_f1(labels, scores)
    pix_auroc = pixel_auroc(np.concatenate([g.ravel() for g in gts]),
                            np.concatenate([a.ravel() for a in amaps]))
    defect_pairs = [(g, a) for g, a in zip(gts, amaps) if g.max() > 0]
    pro = aupro([g for g, _ in defect_pairs], [a for _, a in defect_pairs]) if defect_pairs else None

    dice = iou = None
    if meta["method"] == "unet" and defect_pairs:
        dis = [dice_iou(g, (a > 0.5).astype(np.uint8)) for g, a in defect_pairs]
        dice = float(np.mean([d["dice"] for d in dis]))
        iou = float(np.mean([d["iou"] for d in dis]))

    pooled = {"image_auroc": img_auroc, "image_f1": f1, "image_f1_threshold": thr,
              "pixel_auroc": pix_auroc, "pixel_aupro": pro, "pixel_f1": None,
              "dice": dice, "iou": iou}

    payload = {
        "schema_version": "1.0",
        "run_id": run_dir.name,
        "stage": 2,
        "model": meta["method"],
        "augmentation": meta["augmentation"],
        "seed": meta["seed"],
        "smoke": False,
        "device": "cuda:0",
        "split": {"scheme": meta["split"], "n_folds": len(per_fold),
                  "test_set": "bgad_pooled"},
        "dataset": {"n_images": int(len(df)),
                    "n_good": int((labels == 0).sum()),
                    "n_defect": int((labels == 1).sum())},
        "metrics": {"pooled": pooled, "per_fold": per_fold},
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "note": ("metrics.json aus gespeicherten Predictions rekonstruiert "
                 "(scores.csv + amaps.npz); identische Metrik-Funktionen wie im Lauf."),
    }

    d = f"{pooled['dice']:.4f}" if pooled["dice"] is not None else "-"
    print(f"  {run_dir.name}")
    print(f"    img_auroc={img_auroc:.4f}  pix_auroc={pix_auroc:.4f}  "
          f"aupro={pro:.4f}  dice={d}")

    if not dry_run:
        out.write_text(json.dumps(payload, indent=2))
        cfg_out = run_dir / "config.yaml"
        if not cfg_out.exists() or force:
            cfg_out.write_text(yaml.dump({
                "method": meta["method"], "split": meta["split"],
                "augmentation": meta["augmentation"], "seed": meta["seed"],
                "epochs": 200 if meta["method"] == "ae" else (300 if meta["method"] == "unet" else None),
                "tile_size": 256, "device": "cuda:0", "smoke": False,
                "fg_quantile": 0.98, "coreset_ratio": 0.1,
                "rebuilt_from_predictions": True,
            }, default_flow_style=False, sort_keys=False))
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-2 metrics aus Predictions rekonstruieren")
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ok = skipped = 0
    for rd in args.run_dirs:
        if not rd.is_dir():
            continue
        try:
            if rebuild(rd, args.dry_run, args.force) is not None:
                ok += 1
            else:
                skipped += 1
        except Exception as exc:      # noqa: BLE001
            print(f"  FEHLER {rd.name}: {exc}")
            skipped += 1
    print(f"\nFertig: {ok} rekonstruiert, {skipped} uebersprungen"
          + (" (--dry-run: nichts geschrieben)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
