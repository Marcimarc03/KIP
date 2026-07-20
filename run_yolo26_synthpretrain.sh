#!/usr/bin/env bash
# =====================================================================
# Kreuzung Architektur x Vortraining: YOLO26 mit synthetischem Vortraining.
#
# Ergaenzt die bestehende Ablation (bisher nur YOLO11 = Strategie C) um
# YOLO26, damit die Frage "profitieren andere Architekturen ebenso vom
# synthetischen Vortraining?" fuer zwei von drei Architekturen beantwortet
# werden kann. Mask2Former ist nicht enthalten: Die Trainingspipeline
# unterstuetzt weder einen frei waehlbaren Datensatz noch einen eigenen
# M2F-Checkpoint als Initialisierung (run_stage1.py:55, :184).
#
#   Teil 1: YOLO26 auf dem SYNTHETISCHEN Datensatz vortrainieren (1 Seed).
#           Das Vortraining liefert nur die Initialisierung -> ein Seed
#           genuegt; die Streuung wird im Finetuning gemessen.
#   Teil 2: 3x Finetuning auf real (Seeds 42/1/2), Konfiguration identisch
#           zu den bestehenden YOLO26-Laeufen (100 Epochen, imgsz 1088,
#           batch 16, aug on, KIP_WORKERS=0) -> direkt vergleichbar.
#
# Nutzung (in tmux, venv-kip aktiv):
#   tmux new -s y26synth
#   source /workspace/blum/venv-kip/bin/activate
#   bash run_yolo26_synthpretrain.sh
#   abkoppeln: Strg+b, dann d
#
# Laufzeit: ~2,5 h Vortraining + ~2,5 h Finetuning = ~5 h.
# =====================================================================
set -u
cd /workspace/blum/KIP
export KIP_WORKERS=0
mkdir -p logs results/synth1088_y26
TS=$(date +%Y%m%d_%H%M)
MASTER="logs/y26_synthpretrain_${TS}.log"
SEEDS=(42 1 2)
PRETRAIN_SEED=42
CKPT="results/synth1088_y26/seed${PRETRAIN_SEED}/weights/best.pt"
declare -a RESULTS=()

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER"; }
run_step(){ local l="$1"; shift; log "=== START: $l ==="
  if "$@" >>"$MASTER" 2>&1; then log "=== OK:    $l ==="; RESULTS+=("OK   $l")
  else log "=== FAIL:  $l ==="; RESULTS+=("FAIL $l"); fi; }

# --- 0. Pre-Flight -----------------------------------------------------
log "Seeds (Finetuning)=${SEEDS[*]} ; Vortraining-Seed=${PRETRAIN_SEED}"
[ -f data/synth_Daten/data.yaml ] || { log "ABBRUCH: synth-Datensatz fehlt"; exit 1; }
GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
if [ "${GPU_FREE:-0}" -lt 12000 ]; then
  log "ABBRUCH: nur ${GPU_FREE:-?} MiB GPU frei (<12000). Laeuft noch etwas?"; exit 1
fi
log "GPU frei: ${GPU_FREE} MiB (ok)"

# --- 1. Synthetisches Vortraining (YOLO26, 1 Seed) ---------------------
if [ -f "$CKPT" ]; then
  log "Vortrainings-Checkpoint existiert bereits -> Teil 1 uebersprungen"
  RESULTS+=("SKIP synth-Vortraining (Checkpoint vorhanden)")
else
  run_step "YOLO26 synth-Vortraining @1088 seed${PRETRAIN_SEED}" \
    yolo segment train \
      data=data/synth_Daten/data.yaml \
      model=yolo26n-seg.pt \
      epochs=100 imgsz=1088 batch=16 workers=0 \
      device=cuda:0 seed=$PRETRAIN_SEED \
      project=results/synth1088_y26 name=seed${PRETRAIN_SEED} exist_ok=True
fi

# Ultralytics-CLI schreibt ggf. nach runs/segment/<project> -> Symlink anlegen
if [ ! -f "$CKPT" ] && [ -f "runs/segment/results/synth1088_y26/seed${PRETRAIN_SEED}/weights/best.pt" ]; then
  rm -rf results/synth1088_y26
  ln -s /workspace/blum/KIP/runs/segment/results/synth1088_y26 results/synth1088_y26
  log "Symlink auf runs/segment/... angelegt"
fi

if [ ! -f "$CKPT" ]; then
  log "ABBRUCH: Vortrainings-Checkpoint nicht gefunden ($CKPT)"
  log "===================== ZUSAMMENFASSUNG ====================="
  for r in "${RESULTS[@]}"; do log "$r"; done
  exit 1
fi
log "Vortrainings-Checkpoint: $CKPT"

# --- 2. Finetuning auf real, 3 Seeds -----------------------------------
#     Konfiguration identisch zu den bestehenden YOLO26-aug-on-Laeufen,
#     einzige Aenderung: --weights (synth-Checkpoint statt COCO-Default).
for s in "${SEEDS[@]}"; do
  run_step "YOLO26 synth-pretrain -> real seed${s}" \
    python scripts/run_stage1.py --model yolo26 --aug on \
      --epochs 100 --imgsz 1088 --batch 16 --device cuda:0 --seed "$s" \
      --weights "$CKPT" --tag synthpretrain
done

# --- Zusammenfassung ---------------------------------------------------
log "===================== ZUSAMMENFASSUNG ====================="
for r in "${RESULTS[@]}"; do log "$r"; done
log "Fertig. Aggregat: python scripts/aggregate_stage1.py"
log "Vergleich: yolo26 on '-' (COCO-Init) vs. yolo26 on 'synthpretrain'"
