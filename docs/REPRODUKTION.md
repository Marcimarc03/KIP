# Reproduktion der Ergebnisse

Dieses Dokument beschreibt die vollständige Befehlsfolge zur Reproduktion
aller Ergebnisse auf einem CUDA-fähigen Server.
Lokale Smoke-Tests laufen auf MPS / CPU (macOS).

## Voraussetzungen

```bash
pip install -e .
# Alle Abhängigkeiten sind in requirements-dev.txt gelistet.
```

Tool-basierten Split setzen/prüfen, dann COCO-Konvertierungen unter `data/coco_converted/` erzeugen:
```bash
python scripts/apply_stage1_split.py    # test=tool98, val=tool03, Rest train (idempotent; ersetzt make_test_split.py)
                                        # -> train=803 / val=101 / test=58, tool-disjunkt, test deckt alle 6 realen Klassen
python scripts/prepare_stage1_coco.py   # COCO aus train/val/test (falls noch nicht erledigt)
```
Hintergrund und Herleitung des Splits: `docs/split_begruendung.md`. Der frühere `make_test_split.py`
(test=tool10) deckte nur 4 der 6 realen Klassen ab und wird nicht mehr verwendet.

---

## Stage-1: Komponenten-Segmentierung (YOLO11n-seg vs. Mask2Former)

Struktur: ein Primärvergleich (Augmentierung AN) plus zwei Ablationen. Vergleichskritisch
identisch: Datensplit, Testset (tool98), `evaluator.py` (pycocotools), Box-/Masken-Metriken,
Augmentierungs-Policy je Block, Seed 42. Jedes Modell mit seinem Standard-Rezept
(YOLO: Ultralytics-Defaults; Mask2Former: AdamW, lr 1e-4, Backbone-Freeze 20). Ausgaben nur
unter `results/component_benchmark/`.

Auflösung: YOLO 1088, Mask2Former default 800. Für die Fairness-Aussage möglichst gleich —
falls M2F bei `--imgsz 1088` (Batch 8) nicht am Speicher scheitert, dort ebenfalls 1088;
sonst die Differenz als jeweiligen Standard-Betriebspunkt dokumentieren.

### Primärvergleich (Augmentierung AN)

```bash
python scripts/run_stage1.py --model yolo --aug on --epochs 100 --imgsz 1088 \
    --batch 16 --device cuda:0 --seed 42

python scripts/run_stage1.py --model mask2former --aug on --epochs 100 --batch 8 \
    --lr 1e-4 --freeze-backbone-epochs 20 --device cuda:0 --seed 42
```

### Ablation 1 — Augmentierung AUS (beide Modelle, sonst identisch)

```bash
python scripts/run_stage1.py --model yolo --aug off --epochs 100 --imgsz 1088 \
    --batch 16 --device cuda:0 --seed 42

python scripts/run_stage1.py --model mask2former --aug off --epochs 100 --batch 8 \
    --lr 1e-4 --freeze-backbone-epochs 20 --device cuda:0 --seed 42
```

### Ablation 2 — Synthetisches Vortraining (nur YOLO), leakage-frei

Initialisierung aus `A_synth_only` (rein synthetisch, hat nie ein reales Tool gesehen).
NICHT `C_synth_pretrain_real_finetune` verwenden: dieser Checkpoint wurde unter dem alten Split
real feingetunt und sah dabei tool98 — das ist jetzt das Testset (Leakage).

```bash
python scripts/run_stage1.py --model yolo --aug on --epochs 100 --imgsz 1088 \
    --batch 16 --device cuda:0 --seed 42 \
    --weights results/results/yolo_runs/A_synth_only/weights/best.pt

python scripts/run_stage1.py --model yolo --aug off --epochs 100 --imgsz 1088 \
    --batch 16 --device cuda:0 --seed 42 \
    --weights results/results/yolo_runs/A_synth_only/weights/best.pt
```

### Optionaler Mask2Former-Lernraten-Check (nur zur Konfig-Wahl; berichte NUR die bessere)

```bash
python scripts/run_stage1.py --model mask2former --aug on --epochs 100 --batch 8 \
    --lr 5e-5 --freeze-backbone-epochs 30 --device cuda:0 --seed 42
```

### Optionaler Eval-only: Sim-to-Real-Gap (Strategie A, kein Neutraining)

Bewertet den vorhandenen synth-only-Checkpoint auf dem korrigierten Testsplit (tool98).
A_synth_only wurde @640 px trainiert -> indikativ (Auflösung im Paper als Fußnote).

```bash
python scripts/eval_stage1_checkpoint.py --model yolo \
    --weights results/results/yolo_runs/A_synth_only/weights/best.pt \
    --imgsz 640 --device cuda:0 --tag strategyA_synth_only
```

### Multi-Seed-Studie (belastbare Ergebnisse, DGX)

Einzelläufe streuen ~0,04 segm-mAP50 (Trainings-Nichtdeterminismus: cuDNN, stochastische
Augmentierung, Datenreihenfolge). Für belastbare Aussagen: 3 Seeds je Modell/Konfiguration,
dann Mittel±Std + Bootstrap-CI. Ein-Stück-Batch (Split-Guard Option B, GPU-Guard,
`KIP_WORKERS=0`, continue-on-error):

```bash
tmux new -s ms
source /workspace/blum/venv-kip/bin/activate
bash scripts/run_multiseed_stage1.sh        # Seeds 42/1/2; ~24-39 h; abkoppeln: Strg+b, d
```

Auswertung (kein Retraining, nutzt gespeicherte Predictions):
```bash
python scripts/aggregate_stage1.py          # Mittel±Std je (Modell, aug, tag); verweigert Konfig-Mix
python scripts/bootstrap_stage1.py \
  results/component_benchmark/<yolo26_run> \
  results/component_benchmark/<yolo11_run> \
  results/component_benchmark/<m2f_run>     # 95%-CI + gepaarte Modellvergleiche
```
Regel: Modellabstand nur behaupten, wenn Seed-Bänder disjunkt UND die gepaarte Bootstrap-CI 0 ausschließt.

### Lokale Smoke-Tests (MPS / CPU)

```bash
# Schnelltest YOLO auf MPS (<=2 Epochen, <=40 Bilder)
.venv/bin/python scripts/run_stage1.py --model yolo --aug on --smoke --device mps

# Schnelltest Mask2Former auf CPU (<=2 Epochen, <=20 Bilder)
.venv/bin/python scripts/run_stage1.py --model mask2former --aug off --smoke --device cpu
```

---

<!-- WP4 ergänzt hier: Stage-2 Befehle, Figuren-Erzeugung -->

## Stufe 2 — Defekterkennung (BGAD): 4 komplementäre Methoden

Manifest (nichtdestruktiv):
```bash
.venv/bin/python scripts/build_manifest.py --bgad data/BGAD \
  --out results/defect_detection/manifest --missing-mask-policy normal
```

Smoke (dieser Rechner, MPS/CPU) — alle vier Methoden, beide Protokolle:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
for split in fixed loto; do for m in patchcore padim ae unet; do
  .venv/bin/python scripts/run_stage2.py --method $m --split $split --smoke --device mps
done; done
```

Volltraining (CUDA-Server) — komplett per Skript (ausgeführt 2026-07-12, Seed 42):
```bash
tmux new -s s2
bash scripts/run_stage2_all.sh
```
Das Skript rechnet sequenziell (Ausgaben nur unter `results/defect_detection/`):
- **LOTO (primär, tool-disjunkt):** PatchCore, PaDiM, ConvAE (`--epochs 200`),
  U-Net (`--aug on --epochs 300 --loss bce_dice`); zusätzlich U-Net (`--aug off`)
  als Augmentierungs-Ablation.
- **fixed (Sekundär-Anker, verletzt Tool-Disjunktheit by design):** dieselben vier Methoden.

Augmentierung ausschließlich beim supervised U-Net (die unsupervised-Verfahren dürfen
ihre Gut-Trainingsmenge nicht augmentieren). Einzelaufruf-Muster (falls nur eine Methode):
```bash
python scripts/run_stage2.py --method {patchcore|padim|ae|unet} --split {loto|fixed} \
    --tile 256 --device cuda:0 --seed 42 [--aug on --epochs 300 --loss bce_dice]
```
`--missing-mask-policy {normal|unlabeled|error}` steuert die Annahme "Bild ohne Maske = Gutteil".
Nur `smoke=False`-Läufe sind berichtbar. LOTO ist primär; `--split fixed` nutzt den vorhandenen
train/val-Split (sekundär, optimistisch). Aggregat aller Läufe: `results/defect_detection/summary.csv`.

## Gesamter Smoke-Durchlauf
```bash
bash scripts/smoke_all.sh
```
