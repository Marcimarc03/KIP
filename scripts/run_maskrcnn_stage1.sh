#!/usr/bin/env bash
# =====================================================================
# Mask R-CNN Stage-1: fairer Vergleich analog YOLO/M2F.
#   * aug-on 3 Seeds + aug-off 3 Seeds (Architektur + Augmentierungs-Ablation)
#   * Synthetik-Ablation: synth-Vortraining -> real-Finetuning, 3 Seeds
# Identische Konfig wie die uebrigen Modelle: KIP_WORKERS=0, Split-Guard Option B,
# GPU-Hard-Guard. continue-on-error, per-Schritt OK/FAIL, Zusammenfassung am Ende.
# Ergaenzt (ersetzt NICHT) run_multiseed_stage1.sh.
#
# Nutzung (in tmux, venv-kip aktiv, auf der DGX):
#   tmux new -s mrcnn ; bash scripts/run_maskrcnn_stage1.sh
#   abkoppeln: Strg+b, dann d.
# Hinweis: batch 4 fuer Mask R-CNN (ResNet50-FPN-v2 @800 ist speicherintensiv).
# =====================================================================
set -u
cd "$(dirname "$0")/.."
export KIP_WORKERS=0
mkdir -p logs
TS=$(date +%Y%m%d_%H%M)
MASTER="logs/maskrcnn_${TS}.log"
declare -a RESULTS=()
SEEDS=(42 1 2)
BATCH=4
SYNTH_CKPT="results/maskrcnn_synth/weights/maskrcnn.pt"

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER"; }
run_step(){ local l="$1"; shift; log "=== START: $l ==="
  if "$@" >>"$MASTER" 2>&1; then log "=== OK:    $l ==="; RESULTS+=("OK   $l")
  else log "=== FAIL:  $l (Details in $MASTER) ==="; RESULTS+=("FAIL $l"); fi; }

# --- 0. Pre-Flight + Split-Guard (identisch zu run_multiseed_stage1.sh) ---
log "KIP_WORKERS=$KIP_WORKERS ; Seeds=${SEEDS[*]} ; batch=$BATCH"
log "GPU:"; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | tee -a "$MASTER"
GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
if [ "${GPU_FREE:-0}" -lt 12000 ]; then
  log "ABBRUCH: nur ${GPU_FREE:-?} MiB GPU frei (<12000 noetig) -> OOM-Gefahr. Spaeter starten."
  exit 1
fi
log "GPU frei: ${GPU_FREE} MiB (ok)"
split_state(){ python -c "import json,re;d=json.load(open('data/coco_converted/test.json'));t=sorted({re.match(r'(tool\\d+)',i['file_name']).group(1) for i in d['images']});print(len(d['images']),','.join(t))" 2>/dev/null; }
S=$(split_state); N=${S%% *}; T=${S#* }
if [ "${N:-0}" != "148" ] || [ "${T:-x}" != "tool10,tool98" ]; then
  log "ABBRUCH: Split ist nicht Option B (n=${N:-?}, tools=${T:-?}). Erst apply_stage1_split.py + prepare_stage1_coco.py."
  exit 1
fi
log "Split: n_test=${N:-?}  tools=${T:-?}  (Option B ok)"

# --- 1. Architektur + Augmentierung: aug-on / aug-off, je 3 Seeds ---
for s in "${SEEDS[@]}"; do
  run_step "MaskRCNN aug-on seed$s (=realonly)" \
    python scripts/run_stage1.py --model maskrcnn --aug on  --epochs 100 --imgsz 800 --batch "$BATCH" --device cuda:0 --seed "$s" --tag realonly
  run_step "MaskRCNN aug-off seed$s" \
    python scripts/run_stage1.py --model maskrcnn --aug off --epochs 100 --imgsz 800 --batch "$BATCH" --device cuda:0 --seed "$s" --tag realonly
done

# --- 2. Synthetik-Ablation: 1x synth-Vortraining, dann real-Finetuning 3 Seeds ---
run_step "MaskRCNN synth-Vortraining (1x)" \
  python scripts/pretrain_maskrcnn_synth.py --epochs 100 --batch "$BATCH" --imgsz 800 --device cuda:0 --seed 42 --aug on
if [ -f "$SYNTH_CKPT" ]; then
  export KIP_MASKRCNN_INIT="$SYNTH_CKPT"
  log "KIP_MASKRCNN_INIT=$KIP_MASKRCNN_INIT"
  for s in "${SEEDS[@]}"; do
    run_step "MaskRCNN synth-pretrain->real seed$s" \
      python scripts/run_stage1.py --model maskrcnn --aug on --epochs 100 --imgsz 800 --batch "$BATCH" --device cuda:0 --seed "$s" --tag synthpretrain
  done
  unset KIP_MASKRCNN_INIT
else
  log "FAIL: Synth-Checkpoint fehlt ($SYNTH_CKPT) -> Synth-Ablation uebersprungen."
  RESULTS+=("FAIL MaskRCNN synth-Ablation (kein Checkpoint)")
fi

# --- Abschluss ---
log "===================== ZUSAMMENFASSUNG ====================="
for r in "${RESULTS[@]}"; do log "$r"; done
log "Fertig. Aggregat: python scripts/aggregate_stage1.py"
log "Volllog: $MASTER"
