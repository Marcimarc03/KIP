#!/usr/bin/env python
"""Mask2Former auf dem SYNTHETISCHEN Datensatz vortrainieren (Strategie-C-Basis).

Nutzt die gepatchte Trainer-Kopie (mask2former_trainer.py), damit die
produktive Pipeline (mask2former_trainer.py / run_stage1.py) unveraendert
bleibt und alle bestehenden Ergebnisse reproduzierbar sind.

Konfiguration identisch zu den bestehenden M2F-Laeufen:
AdamW lr 1e-4, batch 8, 100 Epochen, Backbone 20 Epochen eingefroren.
Einziger Unterschied: Trainingsdaten = data/coco_synth statt real.

Ausgabe: results/m2f_synth1088/checkpoints  (fuer --weights beim Finetuning)

Nutzung (in tmux, venv-kip aktiv):
    python scripts/pretrain_m2f_synth.py --epochs 100 --batch 8 --device cuda:0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kip.config import Stage1Config, seed_everything
from kip.stage1.mask2former_trainer import Mask2FormerTrainer

_SYNTH_COCO = _ROOT / "data" / "coco_synth"
_SYNTH_IMAGES = _ROOT / "data" / "synth_Daten" / "images" / "train"
_OUT = _ROOT / "results" / "m2f_synth1088"


def main() -> None:
    ap = argparse.ArgumentParser(description="M2F synth-Vortraining")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--freeze-backbone-epochs", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--aug", choices=["on", "off"], default="on")
    args = ap.parse_args()

    train_json = _SYNTH_COCO / "train.json"
    val_json = _SYNTH_COCO / "val.json"
    for p in (train_json, val_json, _SYNTH_IMAGES):
        if not p.exists():
            sys.exit(f"ERROR: fehlt: {p}")

    seed_everything(args.seed)
    _OUT.mkdir(parents=True, exist_ok=True)

    cfg = Stage1Config(
        model="mask2former",
        augmentation=(args.aug == "on"),
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        smoke=False,
    )

    print(f"[pretrain] synth-COCO : {train_json}")
    print(f"[pretrain] Bilder     : {_SYNTH_IMAGES}")
    print(f"[pretrain] Ausgabe    : {_OUT}")
    print(f"[pretrain] epochs={args.epochs} batch={args.batch} lr={args.lr} "
          f"freeze={args.freeze_backbone_epochs} seed={args.seed}")

    trainer = Mask2FormerTrainer(
        cfg=cfg,
        coco_train_json=train_json,
        coco_val_json=val_json,
        images_dir=_SYNTH_IMAGES,
        run_dir=_OUT,
    )

    t0 = time.time()
    ckpt = trainer.train()
    print(f"[pretrain] fertig in {(time.time()-t0)/3600:.2f} h")
    print(f"[pretrain] CHECKPOINT: {ckpt}")
    print("[pretrain] Fuer das Finetuning:")
    print(f"    export KIP_M2F_INIT={ckpt}")


if __name__ == "__main__":
    main()
