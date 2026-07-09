#!/usr/bin/env bash
# Stage-1 Volltraining: Primaervergleich + 2 Ablationen (7 Laeufe, sequenziell).
# Als Skript ausfuehren (nicht Block einfuegen!) - idealerweise in tmux:
#   tmux new -s kip ; bash scripts/run_stage1_all.sh
#
# Umgeht das Copy-Paste-Verkleben und den /dev/shm-Engpass des Containers:
# KIP_WORKERS=0 laedt Daten im Hauptprozess (kein Shared Memory noetig).
set -u
cd "$(dirname "$0")/.."
mkdir -p logs

export KIP_WORKERS=0                       # YOLO ohne DataLoader-Worker -> kein /dev/shm
rm -f /dev/shm/torch_* 2>/dev/null || true # verwaiste shm-Reste abgestuerzter Laeufe

D="--device cuda:0 --seed 42"

echo "=== [1/7] YOLO aug on (Primaervergleich) ==="
python scripts/run_stage1.py --model yolo --aug on --epochs 100 --imgsz 1088 --batch 16 $D 2>&1 | tee logs/yolo_augon.log
echo "=== [2/7] Mask2Former aug on (Primaervergleich) ==="
python scripts/run_stage1.py --model mask2former --aug on --epochs 100 --imgsz 1088 --batch 8 --lr 1e-4 --freeze-backbone-epochs 20 $D 2>&1 | tee logs/m2f_augon.log
echo "=== [3/7] YOLO aug off (Ablation Augmentierung) ==="
python scripts/run_stage1.py --model yolo --aug off --epochs 100 --imgsz 1088 --batch 16 $D 2>&1 | tee logs/yolo_augoff.log
echo "=== [4/7] Mask2Former aug off (Ablation Augmentierung) ==="
python scripts/run_stage1.py --model mask2former --aug off --epochs 100 --imgsz 1088 --batch 8 --lr 1e-4 --freeze-backbone-epochs 20 $D 2>&1 | tee logs/m2f_augoff.log
echo "=== [5/7] YOLO aug on + synth-Vortraining (Ablation, A_synth_only) ==="
python scripts/run_stage1.py --model yolo --aug on --epochs 100 --imgsz 1088 --batch 16 $D --weights results/results/yolo_runs/A_synth_only/weights/best.pt 2>&1 | tee logs/yolo_augon_synthpre.log
echo "=== [6/7] YOLO aug off + synth-Vortraining (Ablation, A_synth_only) ==="
python scripts/run_stage1.py --model yolo --aug off --epochs 100 --imgsz 1088 --batch 16 $D --weights results/results/yolo_runs/A_synth_only/weights/best.pt 2>&1 | tee logs/yolo_augoff_synthpre.log
echo "=== [7/7] Mask2Former aug on lr5e-5 (optionaler LR-Check) ==="
python scripts/run_stage1.py --model mask2former --aug on --epochs 100 --imgsz 1088 --batch 8 --lr 5e-5 --freeze-backbone-epochs 30 $D 2>&1 | tee logs/m2f_augon_lr5e5.log

echo "=== Stage 1 komplett - Ergebnisse in results/component_benchmark/, Logs in logs/ ==="
