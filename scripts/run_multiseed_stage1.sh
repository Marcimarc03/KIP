#!/usr/bin/env bash
# =====================================================================
# Multi-Seed Stage-1 fuer BELASTBARE Ergebnisse (Mittel +/- Std ueber Seeds).
# Alle Laeufe unter IDENTISCHER Konfig (KIP_WORKERS=0, gleiche imgsz/batch/epochs);
# der Seed ist die EINZIGE Variable. Ein-Stueck-Lauf, robust:
#   * Pre-Flight + Split-Guard (Option B: 148, tool10+tool98) -> sonst Abbruch.
#   * continue-on-error, per-Schritt OK/FAIL, Zusammenfassung am Ende.
#   * Prioritaet: (1) Architektur aug-on 3 Seeds  (2) Synthetik-Ablation S2 3 Seeds
#                 (3) S3 sim-to-real  (4) OPTIONAL aug-off 3 Seeds (zuletzt).
# YOLO11 aug-on == S1 real-only (Tag realonly) -> dient zugleich als Ablations-Baseline.
#
# Nutzung (in tmux):  tmux new -s ms ; bash scripts/run_multiseed_stage1.sh
#   abkoppeln: Strg+b, dann d.  Laufzeit ~24-30 h fuer die essentiellen Bloecke 1-3.
# =====================================================================
set -u
cd "$(dirname "$0")/.."
export KIP_WORKERS=0
mkdir -p logs
TS=$(date +%Y%m%d_%H%M)
MASTER="logs/multiseed_${TS}.log"
declare -a RESULTS=()
A_SYNTH="results/results/yolo_runs/A_synth_only/weights/best.pt"
SEEDS=(42 1 2)

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER"; }
run_step(){ local l="$1"; shift; log "=== START: $l ==="
  if "$@" >>"$MASTER" 2>&1; then log "=== OK:    $l ==="; RESULTS+=("OK   $l")
  else log "=== FAIL:  $l (Details in $MASTER) ==="; RESULTS+=("FAIL $l"); fi; }

# --- 0. Pre-Flight + Split-Guard ---
log "KIP_WORKERS=$KIP_WORKERS ; Seeds=${SEEDS[*]}"
[ -f "$A_SYNTH" ] || { log "ABBRUCH: A_synth_only fehlt ($A_SYNTH)"; exit 1; }
log "GPU:"; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | tee -a "$MASTER"
# GPU-Hard-Guard: genug freier Speicher? (geteilter Container -> OOM waere Totalausfall)
GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
if [ "${GPU_FREE:-0}" -lt 12000 ]; then
  log "ABBRUCH: nur ${GPU_FREE:-?} MiB GPU frei (<12000 noetig) -> OOM-Gefahr. Spaeter starten."
  exit 1
fi
log "GPU frei: ${GPU_FREE} MiB (ok)"
log "Disk:"; df -h . | tail -1 | tee -a "$MASTER"
split_state(){ python -c "import json,re;d=json.load(open('data/coco_converted/test.json'));t=sorted({re.match(r'(tool\\d+)',i['file_name']).group(1) for i in d['images']});print(len(d['images']),','.join(t))" 2>/dev/null; }
S=$(split_state); N=${S%% *}; T=${S#* }
if [ "${N:-0}" != "148" ] || [ "${T:-x}" != "tool10,tool98" ]; then
  log "Split (n=${N:-?}, tools=${T:-?}) != Option B -> apply_stage1_split + prepare_coco"
  python scripts/apply_stage1_split.py  >>"$MASTER" 2>&1
  python scripts/prepare_stage1_coco.py >>"$MASTER" 2>&1
  S=$(split_state); N=${S%% *}; T=${S#* }
fi
log "Split: n_test=${N:-?}  tools=${T:-?}  (erwartet 148 / tool10,tool98)"
if [ "${N:-0}" != "148" ] || [ "${T:-x}" != "tool10,tool98" ]; then
  log "ABBRUCH: Split ist nicht Option B."; exit 1
fi

# --- 1. Architektur aug-on, 3 Seeds (YOLO11 realonly=S1, YOLO26, M2F) ---
for s in "${SEEDS[@]}"; do
  run_step "YOLO11 aug-on seed$s (=S1 realonly)" \
    python scripts/run_stage1.py --model yolo   --aug on --epochs 100 --imgsz 1088 --batch 16 --device cuda:0 --seed "$s" --tag realonly
  run_step "YOLO26 aug-on seed$s" \
    python scripts/run_stage1.py --model yolo26 --aug on --epochs 100 --imgsz 1088 --batch 16 --device cuda:0 --seed "$s"
  run_step "M2F aug-on seed$s" \
    python scripts/run_stage1.py --model mask2former --aug on --epochs 100 --batch 8 --lr 1e-4 --freeze-backbone-epochs 20 --device cuda:0 --seed "$s"
done

# --- 2. Synthetik-Ablation S2 (synth-pretrain -> real-finetune), 3 Seeds ---
#     S1 (real-only) sind die YOLO11-realonly-Laeufe aus Block 1.
for s in "${SEEDS[@]}"; do
  run_step "S2 synth-pretrain seed$s" \
    python scripts/run_stage1.py --model yolo --aug on --epochs 100 --imgsz 1088 --batch 16 --device cuda:0 --seed "$s" \
      --weights "$A_SYNTH" --tag synthpretrain
done

# --- 3. S3 sim-to-real (eval-only, deterministisch -> 1 Lauf) ---
run_step "S3 sim-to-real (eval-only)" \
  python scripts/eval_stage1_checkpoint.py --model yolo --weights "$A_SYNTH" --imgsz 640 --device cuda:0 --tag simtoreal

# --- 4. OPTIONAL: aug-off 3 Seeds (YOLO11 + M2F) -- niedrigste Prio, laeuft zuletzt ---
for s in "${SEEDS[@]}"; do
  run_step "YOLO11 aug-off seed$s" \
    python scripts/run_stage1.py --model yolo --aug off --epochs 100 --imgsz 1088 --batch 16 --device cuda:0 --seed "$s" --tag realonly
  run_step "M2F aug-off seed$s" \
    python scripts/run_stage1.py --model mask2former --aug off --epochs 100 --batch 8 --lr 1e-4 --freeze-backbone-epochs 20 --device cuda:0 --seed "$s"
done

# --- Abschluss ---
log "===================== ZUSAMMENFASSUNG ====================="
for r in "${RESULTS[@]}"; do log "$r"; done
log "Fertig. Aggregat: python scripts/aggregate_stage1.py"
log "Volllog: $MASTER"
