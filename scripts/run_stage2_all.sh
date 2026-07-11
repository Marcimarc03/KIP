#!/usr/bin/env bash
# Stage-2 Volltraining (Defekterkennung), sequenziell, idealerweise in tmux:
#   tmux new -s s2 ; bash scripts/run_stage2_all.sh
# LOTO = Primaerprotokoll; fixed = Sekundaerprotokoll (Anker; verletzt Tool-Disjunktheit by design).
# Augmentierung nur beim supervised U-Net; die unsupervised-Verfahren duerfen ihre
# Gut-Trainingsmenge NICHT augmentieren (wuerde die Normal-Verteilung verwaessern).
set -u
cd "$(dirname "$0")/.."
mkdir -p logs

run_block () {   # $1 = split (loto|fixed)
  S="$1"
  echo "=== [$S] PatchCore ==="
  python scripts/run_stage2.py --method patchcore --split "$S" --tile 256 --device cuda:0 --seed 42 2>&1 | tee "logs/s2_${S}_patchcore.log"
  echo "=== [$S] PaDiM ==="
  python scripts/run_stage2.py --method padim --split "$S" --tile 256 --device cuda:0 --seed 42 2>&1 | tee "logs/s2_${S}_padim.log"
  echo "=== [$S] ConvAE ==="
  python scripts/run_stage2.py --method ae --epochs 200 --split "$S" --tile 256 --device cuda:0 --seed 42 2>&1 | tee "logs/s2_${S}_ae.log"
  echo "=== [$S] U-Net (Augmentierung an) ==="
  python scripts/run_stage2.py --method unet --aug on --epochs 300 --loss bce_dice --split "$S" --tile 256 --device cuda:0 --seed 42 2>&1 | tee "logs/s2_${S}_unet_augon.log"
}

echo "===== PRIMAER: LOTO ====="
run_block loto
echo "=== [loto] U-Net (Augmentierung aus, Ablation) ==="
python scripts/run_stage2.py --method unet --aug off --epochs 300 --loss bce_dice --split loto --tile 256 --device cuda:0 --seed 42 2>&1 | tee "logs/s2_loto_unet_augoff.log"

echo "===== SEKUNDAER: fixed (Anker) ====="
run_block fixed

echo "=== Stage 2 komplett - Ergebnisse in results/defect_detection/, Logs in logs/ ==="
