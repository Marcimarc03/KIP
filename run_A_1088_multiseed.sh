#!/usr/bin/env bash
# =====================================================================
# Strategie A (synth-only) bei 1088 px, 3 Seeds — fuer den auflösungs-
# sauberen A/B/C-Sim-to-Real-Vergleich.
#
#   Teil 1: YOLO11n-seg auf dem SYNTHETISCHEN Datensatz trainieren (1088).
#   Teil 2: jeden Checkpoint auf dem REALEN 148er-Testsplit evaluieren
#           (eval_stage1_checkpoint.py -> results/component_benchmark/).
#
# A wird damit auflösungsgleich zu B/C (1088). Startgewichte wie B/C:
# yolo11n-seg.pt (COCO-pretrained) -> Default von `yolo segment train`.
#
# Nutzung (in tmux, venv-kip aktiv):
#   tmux new -s aTrain
#   source /workspace/blum/venv-kip/bin/activate
#   bash run_A_1088_multiseed.sh
#   abkoppeln: Strg+b, dann d
# =====================================================================
set -u
cd /workspace/blum/KIP
export KIP_WORKERS=0
mkdir -p logs results/synth1088
TS=$(date +%Y%m%d_%H%M)
MASTER="logs/A_1088_${TS}.log"
SEEDS=(42 1 2)
declare -a RESULTS=()

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER"; }
run_step(){ local l="$1"; shift; log "=== START: $l ==="
  if "$@" >>"$MASTER" 2>&1; then log "=== OK:    $l ==="; RESULTS+=("OK   $l")
  else log "=== FAIL:  $l ==="; RESULTS+=("FAIL $l"); fi; }

# --- Pre-Flight ---
log "KIP_WORKERS=$KIP_WORKERS ; Seeds=${SEEDS[*]}"
[ -f data/synth_Daten/data.yaml ] || { log "ABBRUCH: synth-Datensatz fehlt"; exit 1; }
GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
if [ "${GPU_FREE:-0}" -lt 12000 ]; then
  log "ABBRUCH: nur ${GPU_FREE:-?} MiB GPU frei (<12000) -> OOM-Gefahr."; exit 1
fi
log "GPU frei: ${GPU_FREE} MiB (ok)"

# --- Teil 1: synth-only Training @1088, 3 Seeds ---
for s in "${SEEDS[@]}"; do
  run_step "A-train synth @1088 seed$s" \
    yolo segment train \
      data=data/synth_Daten/data.yaml \
      model=yolo11n-seg.pt \
      epochs=100 imgsz=1088 batch=16 workers=0 \
      device=cuda:0 seed=$s \
      project=results/synth1088 name=seed$s exist_ok=True
done

# --- Teil 2: Eval jedes Checkpoints auf dem REALEN 148er-Split @1088 ---
for s in "${SEEDS[@]}"; do
  CKPT="results/synth1088/seed$s/weights/best.pt"
  if [ -f "$CKPT" ]; then
    run_step "A-eval on real @1088 seed$s" \
      python scripts/eval_stage1_checkpoint.py --model yolo \
        --weights "$CKPT" --imgsz 1088 --device cuda:0 \
        --seed "$s" --tag synthonly1088
  else
    log "=== SKIP eval seed$s: Checkpoint fehlt ($CKPT) ==="
    RESULTS+=("FAIL A-eval seed$s (kein ckpt)")
  fi
done

# --- Zusammenfassung ---
log "===================== ZUSAMMENFASSUNG ====================="
for r in "${RESULTS[@]}"; do log "$r"; done
log "Fertig. Aggregat: python scripts/aggregate_stage1.py"
