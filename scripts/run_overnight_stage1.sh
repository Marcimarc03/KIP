#!/usr/bin/env bash
# =====================================================================
# Overnight Stage-1 batch: YOLO26, Synthetik-Ablation (YOLO11n), Mask R-CNN.
#
# Robust by design:
#   * KIP_WORKERS=0  -> YOLO-DataLoader ohne Worker (kleines /dev/shm im Container!)
#   * Pre-Flight: A_synth_only-Checkpoint, GPU-/Disk-Status, Split = Option B (148).
#   * Split-Guard prueft Bildzahl (148) UND Tools ({tool10,tool98}); sonst Abbruch.
#   * continue-on-error: ein fehlgeschlagener Schritt killt nicht die Nacht.
#   * Mask R-CNN (ungetestet) laeuft zuletzt und nur, wenn sein Smoke gruen ist.
#   * Jeder Schritt loggt OK/FAIL; am Ende eine Zusammenfassung.
#
# Nutzung (in tmux):
#   tmux new -s s1 ; source /workspace/blum/venv-kip/bin/activate
#   bash scripts/run_overnight_stage1.sh        # abkoppeln: Strg+b, dann d
# =====================================================================
set -u
cd "$(dirname "$0")/.."
export KIP_WORKERS=0                 # KRITISCH: sonst crashen YOLO-Laeufe am /dev/shm
mkdir -p logs
TS=$(date +%Y%m%d_%H%M)
MASTER="logs/overnight_${TS}.log"
declare -a RESULTS=()
A_SYNTH="results/results/yolo_runs/A_synth_only/weights/best.pt"

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER"; }

run_step(){   # $1 = Label, Rest = Kommando
  local label="$1"; shift
  log "=== START: ${label} ==="
  if "$@" >>"$MASTER" 2>&1; then
    log "=== OK:    ${label} ==="; RESULTS+=("OK   ${label}"); return 0
  else
    log "=== FAIL:  ${label} (Details in ${MASTER}) ==="; RESULTS+=("FAIL ${label}"); return 1
  fi
}

# ---------------------------------------------------------------------
# 0. Pre-Flight
# ---------------------------------------------------------------------
log "KIP_WORKERS=${KIP_WORKERS}"
if [ ! -f "$A_SYNTH" ]; then
  log "ABBRUCH: A_synth_only-Checkpoint fehlt ($A_SYNTH) -> S2/S3 nicht moeglich."; exit 1
fi
log "GPU-Status:"; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | tee -a "$MASTER"
log "Speicherplatz:"; df -h . | tail -1 | tee -a "$MASTER"

# Split-Guard: Option B (148 Bilder, Tools {tool10,tool98})
split_state(){ python -c "import json,re;d=json.load(open('data/coco_converted/test.json'));t=sorted({re.match(r'(tool\\d+)',i['file_name']).group(1) for i in d['images']});print(len(d['images']),','.join(t))" 2>/dev/null; }
STATE=$(split_state); NTEST=${STATE%% *}; TOOLS=${STATE#* }
if [ "${NTEST:-0}" != "148" ] || [ "${TOOLS:-x}" != "tool10,tool98" ]; then
  log "Split ist (n=${NTEST:-?}, tools=${TOOLS:-?}) != Option B -> setze apply_stage1_split + prepare_coco"
  python scripts/apply_stage1_split.py  >>"$MASTER" 2>&1
  python scripts/prepare_stage1_coco.py >>"$MASTER" 2>&1
  STATE=$(split_state); NTEST=${STATE%% *}; TOOLS=${STATE#* }
fi
log "Split: n_test=${NTEST:-?}  tools=${TOOLS:-?}  (erwartet 148 / tool10,tool98)"
if [ "${NTEST:-0}" != "148" ] || [ "${TOOLS:-x}" != "tool10,tool98" ]; then
  log "ABBRUCH: Split ist nicht Option B. Keine Laeufe gestartet."; exit 1
fi

# ---------------------------------------------------------------------
# 1. YOLO26-seg (validiert per Smoke) -- 1088, wie YOLO11
# ---------------------------------------------------------------------
run_step "YOLO26 aug on"  python scripts/run_stage1.py --model yolo26 --aug on  --epochs 100 --imgsz 1088 --batch 16 --device cuda:0 --seed 42
run_step "YOLO26 aug off" python scripts/run_stage1.py --model yolo26 --aug off --epochs 100 --imgsz 1088 --batch 16 --device cuda:0 --seed 42

# ---------------------------------------------------------------------
# 2. Synthetik-Ablation YOLO11n (AP2) -- alle gegen dasselbe Option-B-Testset.
#    Reihenfolge: S1 real-only -> S3 sim-to-real (billig, prueft A_synth_only frueh)
#    -> S2 synth-pretrain (teuer). A_synth_only ist leakage-frei; NICHT C_* verwenden.
# ---------------------------------------------------------------------
run_step "S1 YOLO11n real-only" \
  python scripts/run_stage1.py --model yolo --aug on --epochs 100 --imgsz 1088 --batch 16 --device cuda:0 --seed 42 --tag realonly
run_step "S3 sim-to-real (eval-only)" \
  python scripts/eval_stage1_checkpoint.py --model yolo --weights "$A_SYNTH" --imgsz 640 --device cuda:0 --tag simtoreal
run_step "S2 YOLO11n synth-pretrain" \
  python scripts/run_stage1.py --model yolo --aug on --epochs 100 --imgsz 1088 --batch 16 --device cuda:0 --seed 42 \
    --weights "$A_SYNTH" --tag synthpretrain

# ---------------------------------------------------------------------
# 3. Mask R-CNN (ungetestet) -- Smoke im echten 800px-Regime, Volllauf nur bei Erfolg; batch 2 (OOM-Schutz)
# ---------------------------------------------------------------------
if run_step "Mask R-CNN SMOKE (800px)" python scripts/run_stage1.py --model maskrcnn --aug on --smoke --imgsz 800 --batch 2 --device cuda:0 --tag smoke; then
  run_step "Mask R-CNN aug on"  python scripts/run_stage1.py --model maskrcnn --aug on  --epochs 100 --imgsz 800 --batch 2 --device cuda:0 --seed 42
  run_step "Mask R-CNN aug off" python scripts/run_stage1.py --model maskrcnn --aug off --epochs 100 --imgsz 800 --batch 2 --device cuda:0 --seed 42
else
  log "Mask R-CNN Smoke fehlgeschlagen -> Volllaeufe uebersprungen."
  RESULTS+=("SKIP Mask R-CNN full (Smoke fehlgeschlagen)")
fi

# ---------------------------------------------------------------------
# Abschluss
# ---------------------------------------------------------------------
log "===================== ZUSAMMENFASSUNG ====================="
for r in "${RESULTS[@]}"; do log "$r"; done
log "Fertig. Volllog: ${MASTER}"
log "Ergebnisse: results/component_benchmark/summary.csv"
