#!/usr/bin/env bash
# =====================================================================
# Stage-2 Multi-Seed — methodisch differenziert:
#
#   * U-Net und ConvAE trainieren gradientenbasiert  -> 3 Seeds (42/1/2),
#     um die Trainingsvarianz zu messen.
#   * PatchCore und PaDiM sind deterministisch (eingefrorenes Backbone bzw.
#     geschlossene Statistik) -> genau 1 Lauf; ihre Streuung wird über die
#     LOTO-Folds berichtet, nicht über Seeds.
#
# Fairness: identische Vorverarbeitung, identische Folds (LOTO ist
# seed-unabhaengig), beide Protokolle (loto primaer, fixed als Leakage-Anker)
# fuer alle Verfahren.
#
# Nutzung (in tmux, venv-kip aktiv, NACH Abschluss der A-Laeufe):
#   tmux new -s s2
#   source /workspace/blum/venv-kip/bin/activate
#   bash run_stage2_multiseed.sh
#   abkoppeln: Strg+b, dann d
# =====================================================================
set -u
cd /workspace/blum/KIP
mkdir -p logs
TS=$(date +%Y%m%d_%H%M)
MASTER="logs/stage2_multiseed_${TS}.log"
SEEDS=(42 1 2)
SPLITS=(loto fixed)
declare -a RESULTS=()

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER"; }
run_step(){ local l="$1"; shift; log "=== START: $l ==="
  if "$@" >>"$MASTER" 2>&1; then log "=== OK:    $l ==="; RESULTS+=("OK   $l")
  else log "=== FAIL:  $l ==="; RESULTS+=("FAIL $l"); fi; }

# --- 0. Pre-Flight -----------------------------------------------------
log "Seeds (nur gradientenbasierte Verfahren)=${SEEDS[*]} ; Splits=${SPLITS[*]}"
GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
if [ "${GPU_FREE:-0}" -lt 12000 ]; then
  log "ABBRUCH: nur ${GPU_FREE:-?} MiB GPU frei (<12000). Laeuft noch Stage-1?"
  exit 1
fi
log "GPU frei: ${GPU_FREE} MiB (ok)"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tee -a "$MASTER"

# Konfiguration IDENTISCH zu scripts/run_stage2_all.sh (Vergleichbarkeit mit
# den bisherigen Laeufen vom 12.07.): tile 256, ae 200 Epochen,
# unet 300 Epochen mit bce_dice-Loss.
TILE=256
EP_AE=200
EP_UNET=300

# --- 1. Deterministische Verfahren: genau ein Lauf je Split -------------
#     Seeds waeren hier wirkungslos (kein stochastisches Training).
for sp in "${SPLITS[@]}"; do
  run_step "PatchCore ${sp} (deterministisch, 1 Lauf)" \
    python scripts/run_stage2.py --method patchcore --split "$sp" \
      --tile "$TILE" --device cuda:0 --seed 42
  run_step "PaDiM ${sp} (deterministisch, 1 Lauf)" \
    python scripts/run_stage2.py --method padim --split "$sp" \
      --tile "$TILE" --device cuda:0 --seed 42
done

# --- 2. Gradientenbasiert: ConvAE, 3 Seeds ------------------------------
#     ConvAE trainiert auf Gutteilen -> Augmentierung bleibt AUS
#     (wuerde die modellierte Normalverteilung aufweiten).
for sp in "${SPLITS[@]}"; do
  for s in "${SEEDS[@]}"; do
    run_step "ConvAE ${sp} seed${s}" \
      python scripts/run_stage2.py --method ae --split "$sp" \
        --epochs "$EP_AE" --tile "$TILE" --device cuda:0 --seed "$s"
  done
done

# --- 3. Gradientenbasiert: U-Net, 3 Seeds, Augmentierungs-Ablation ------
#     Nur hier ist Augmentierung sinnvoll (supervised) -> an/aus beibehalten.
for sp in "${SPLITS[@]}"; do
  for a in on off; do
    for s in "${SEEDS[@]}"; do
      run_step "U-Net ${sp} aug-${a} seed${s}" \
        python scripts/run_stage2.py --method unet --split "$sp" --aug "$a" \
          --epochs "$EP_UNET" --loss bce_dice --tile "$TILE" \
          --device cuda:0 --seed "$s"
    done
  done
done

# --- Zusammenfassung ---------------------------------------------------
log "===================== ZUSAMMENFASSUNG ====================="
for r in "${RESULTS[@]}"; do log "$r"; done
OKN=$(printf '%s\n' "${RESULTS[@]}" | grep -c '^OK'   || true)
FAILN=$(printf '%s\n' "${RESULTS[@]}" | grep -c '^FAIL' || true)
log "OK: ${OKN}   FAIL: ${FAILN}"
log "Ergebnisse liegen unter results/defect_detection/ (run_name enthaelt Seed)."
