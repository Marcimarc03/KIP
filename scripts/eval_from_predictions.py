#!/usr/bin/env python
"""metrics.json aus vorhandenen COCO-Predictions nachtraeglich erzeugen.

Anwendungsfall: Ein Stage-1-Lauf hat trainiert und Predictions geschrieben,
die metrics.json fehlt aber (z.B. Abbruch nach der Evaluation). Statt neu zu
trainieren, wird hier dieselbe evaluate_coco-Logik auf die vorhandene
predictions/coco_predictions.json angewendet und eine schema-konforme
metrics.json geschrieben.

Metadaten (model, seed, augmentation, tag, ...) werden aus der config.yaml
des Laufs uebernommen; der Split-Fingerprint wird aus dem GT-JSON neu
berechnet, damit der Lauf sauber ins aggregate_stage1.py faellt.

Nutzung:
    python scripts/eval_from_predictions.py results/component_benchmark/<run_dir>
    python scripts/eval_from_predictions.py <run_dir> --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kip import CLASS_NAMES
from kip.stage1.evaluator import evaluate_coco

_GT_JSON = _ROOT / "data" / "coco_converted" / "test.json"
_TOOL_RE = re.compile(r"(tool\d+)")


def _fingerprint(gt_json: Path) -> dict:
    """Split-Fingerprint analog zu run_stage1.py neu berechnen."""
    raw = gt_json.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()[:16]
    gt = json.loads(raw)
    tools = sorted({m.group(1) for im in gt["images"]
                    if (m := _TOOL_RE.match(im["file_name"]))})
    cat_ids = sorted({a["category_id"] for a in gt["annotations"]})
    return {
        "n_test": len(gt["images"]),
        "test_tools": tools,
        "test_json_sha256": sha,
        "n_classes_total": len(CLASS_NAMES),
        "n_classes_in_test_gt": len(cat_ids),
        "test_category_ids": cat_ids,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="metrics.json aus Predictions erzeugen")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--gt", type=Path, default=_GT_JSON)
    ap.add_argument("--dry-run", action="store_true",
                    help="nur rechnen und anzeigen, nichts schreiben")
    ap.add_argument("--force", action="store_true",
                    help="vorhandene metrics.json ueberschreiben")
    args = ap.parse_args()

    run_dir = args.run_dir
    pred = run_dir / "predictions" / "coco_predictions.json"
    cfg_path = run_dir / "config.yaml"
    out = run_dir / "metrics.json"

    for p in (run_dir, pred, args.gt):
        if not p.exists():
            sys.exit(f"ERROR: fehlt: {p}")
    if out.exists() and not (args.force or args.dry_run):
        sys.exit(f"ERROR: {out} existiert bereits (--force zum Ueberschreiben)")

    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    if not cfg:
        print(f"WARN: keine config.yaml in {run_dir} - Metadaten unvollstaendig")

    print(f"[eval] run     : {run_dir.name}")
    print(f"[eval] preds   : {pred}")
    print(f"[eval] GT      : {args.gt}")

    metrics = evaluate_coco(args.gt, pred, CLASS_NAMES)

    model = cfg.get("model", "unknown")
    aug = bool(cfg.get("augmentation", True))
    seed = int(cfg.get("seed", 42))
    tag = cfg.get("tag", "") or ""
    init_w = cfg.get("weights") or cfg.get("init_weights") or None
    if model in ("yolo", "yolo26") and not init_w:
        init_w = "yolo26n-seg.pt" if model == "yolo26" else "yolo11n-seg.pt"

    fp = _fingerprint(args.gt)
    payload = {
        "schema_version": "1.0",
        "run_id": run_dir.name,
        "stage": 1,
        "model": model,
        "tag": tag,
        "init_weights": init_w,
        "augmentation": aug,
        "seed": seed,
        "smoke": bool(cfg.get("smoke", False)),
        "device": cfg.get("device", "cuda:0"),
        "environment": {"kip_workers": 0, "rebuilt_from_predictions": True},
        "split": {
            "scheme": f"tool_disjoint_{'+'.join(fp['test_tools'])}",
            "n_folds": 1,
            "test_set": "real_v3/test",
            "fingerprint": fp,
        },
        "dataset": {"n_test": fp["n_test"]},
        "metrics": metrics,
        "note": ("metrics.json nachtraeglich aus vorhandenen Predictions "
                 "erzeugt (kein Neutraining); identische evaluate_coco-Logik."),
    }

    print(f"[eval] segm_mAP50={metrics['segm_map50']:.4f}  "
          f"segm_mAP50_95={metrics['segm_map50_95']:.4f}  "
          f"bbox_mAP50={metrics['bbox_map50']:.4f}")
    print(f"[eval] model={model} aug={'on' if aug else 'off'} seed={seed} tag='{tag}'")

    if args.dry_run:
        print("[eval] --dry-run: nichts geschrieben")
        return

    out.write_text(json.dumps(payload, indent=2))
    print(f"[eval] geschrieben: {out}")


if __name__ == "__main__":
    main()
