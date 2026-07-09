#!/usr/bin/env python
"""Eval-only: bewertet einen vorhandenen Stage-1-Checkpoint auf dem Testsplit — OHNE Training.

Anwendungsfall: Sim-to-Real-Domänenlücke (Strategie A, synth-only) konsistent auf dem
korrigierten Testsplit (tool98) bestimmen, ohne das Modell neu zu trainieren. Nutzt exakt
die vorhandene `predict_to_coco`- und `evaluate_coco`-Logik wie `run_stage1.py`.

Beispiel (CUDA-Server oder lokal in der venv):
    python scripts/eval_stage1_checkpoint.py --model yolo \
        --weights results/results/yolo_runs/A_synth_only/weights/best.pt \
        --imgsz 640 --device cuda:0 --tag strategyA_synth_only

Hinweis: A_synth_only wurde bei 640 px trainiert -> Inferenz bei 640; die Zahl ist
gegenüber den 1088-px-Läufen indikativ (Auflösung im Paper als Fußnote nennen).
Ausgaben landen unter results/component_benchmark/ (NIE in results/results/).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kip import CLASS_NAMES
from kip.config import Stage1Config, seed_everything
from kip.stage1.evaluator import evaluate_coco

_DATA_ROOT = _ROOT / "data" / "object_segmentation_real_v3_1088"
_DATA_YAML = _DATA_ROOT / "data.yaml"
_COCO_ROOT = _ROOT / "data" / "coco_converted"
_TEST_JSON = _COCO_ROOT / "test.json"
_IMAGES_TEST = _DATA_ROOT / "images" / "test"
_RESULTS_BASE = _ROOT / "results" / "component_benchmark"


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval-only eines Stage-1-Checkpoints auf dem Testsplit")
    p.add_argument("--model", choices=["yolo", "mask2former"], default="yolo")
    p.add_argument("--weights", required=True, help="zu bewertender Checkpoint (wird NICHT verändert)")
    p.add_argument("--imgsz", type=int, default=640, help="Inferenz-Bildgröße (A_synth_only wurde @640 trainiert)")
    p.add_argument("--device", default="cpu", help="cpu / mps / cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="eval_only", help="Label für den Ausgabeordner")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    seed_everything(args.seed)

    ckpt = Path(args.weights)
    if not ckpt.exists():
        sys.exit(f"ERROR: Checkpoint nicht gefunden: {ckpt}")
    for pth in (_DATA_YAML, _TEST_JSON, _IMAGES_TEST):
        if not pth.exists():
            sys.exit(f"ERROR: fehlt: {pth}\n  Zuerst apply_stage1_split.py + prepare_stage1_coco.py laufen lassen.")

    out_dir = _RESULTS_BASE / f"{args.model}_{args.tag}"
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    pred_json = out_dir / "predictions" / "coco_predictions.json"

    cfg = Stage1Config(
        model=args.model, augmentation=False, epochs=0, imgsz=args.imgsz,
        batch=1, lr=1e-4, freeze_backbone_epochs=0,
        device=args.device, seed=args.seed, smoke=False,
    )

    n_test = len(json.loads(_TEST_JSON.read_text())["images"])
    t0 = time.time()

    if args.model == "yolo":
        from kip.stage1.yolo_trainer import YoloSegTrainer
        trainer = YoloSegTrainer(cfg=cfg, data_yaml=_DATA_YAML, run_dir=out_dir, weights=str(ckpt))
        trainer.predict_to_coco(ckpt=str(ckpt), coco_gt_json=_TEST_JSON,
                                images_dir=_IMAGES_TEST, out_json=pred_json)
    else:
        from kip.stage1.mask2former_trainer import Mask2FormerTrainer
        trainer = Mask2FormerTrainer(cfg=cfg, coco_train_json=_TEST_JSON, coco_val_json=_TEST_JSON,
                                     images_dir=_IMAGES_TEST, run_dir=out_dir)
        trainer.predict_to_coco(ckpt_dir=str(ckpt), coco_gt_json=_TEST_JSON,
                                images_dir=_IMAGES_TEST, out_json=pred_json)

    infer_ms = (time.time() - t0) / max(n_test, 1) * 1000

    metrics = evaluate_coco(gt_json=_TEST_JSON, pred_json=pred_json, class_names=CLASS_NAMES)

    print(f"\n[eval-only] {args.model}  weights={ckpt.name}  imgsz={args.imgsz}  n_test={n_test}")
    print(f"[eval-only] segm_mAP50={metrics['segm_map50']:.4f}  "
          f"segm_mAP50_95={metrics['segm_map50_95']:.4f}  bbox_mAP50={metrics['bbox_map50']:.4f}")
    print("[eval-only] Per-Klasse (segm AP50 / AP50-95; -1 = keine GT im Test):")
    for name, d in metrics.get("per_class", {}).items():
        print(f"    {name:24s} {d['segm_ap50']:6.3f} / {d['segm_ap50_95']:6.3f}")

    payload = {
        "eval_only": True,
        "note": ("Domänenlücken-Eval eines Checkpoints auf dem korrigierten Testsplit (tool98); "
                 "NICHT neu trainiert. A_synth_only @640px -> indikativ."),
        "model": args.model, "weights": str(ckpt), "imgsz": args.imgsz,
        "test_set": "real_v3/test (tool98)", "n_test": n_test,
        "infer_ms_per_image": round(infer_ms, 2), "metrics": metrics,
    }
    (out_dir / "metrics_eval_only.json").write_text(json.dumps(payload, indent=2))
    print(f"\n[eval-only] geschrieben: {out_dir / 'metrics_eval_only.json'}")


if __name__ == "__main__":
    main()
