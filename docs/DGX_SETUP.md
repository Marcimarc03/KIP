# Volltraining auf der wbk-DGX — Ablauf

Zugangsdaten: siehe E-Mail (SSH `root@141.3.142.150 -p 2632`, Netzlaufwerk `\\141.3.142.150\dgx_blum`,
eigene Dateien im Container unter `/workspace/blum`). Container: Python 3.12 + 3.14, CUDA 12.6.
**Python 3.12 verwenden** (requirements sind für 3.10–3.12 getestet).

Sicherheitshinweis: Shared-Container mit root-Login — **keine GitHub-Tokens/Credentials dort
speichern**. Das Repo ist öffentlich klonbar; Ergebnisse gehen über das Netzlaufwerk zurück.

## 1. Daten hochladen (vom Windows-Rechner)

Im Explorer `\\141.3.142.150\dgx_blum` öffnen (Login laut E-Mail) und hineinkopieren:

- `C:\Dev\KIP\data\BGAD\` (19 Defektbilder + Masken, ~70 MB)
- `C:\Dev\KIP\data\object_segmentation_real_v3_1088\` (enthält bereits den test-Split!, ~95 MB)

Tipp: vorher zippen, im Container entpacken — deutlich schneller als viele Einzeldateien.

## 2. Verbinden und Repo aufsetzen

```bash
ssh root@141.3.142.150 -p 2632
cd /workspace/blum
git clone https://github.com/Marcimarc03/KIP.git
cd KIP
mkdir -p data
# hochgeladene Datenordner/Zips von /workspace/blum nach data/ verschieben bzw. entpacken:
# mv ../BGAD data/ && mv ../object_segmentation_real_v3_1088 data/
```

## 3. Umgebung

```bash
python3.12 -m venv /workspace/blum/venv-kip
source /workspace/blum/venv-kip/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())" 2>/dev/null \
  || pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-dev.txt
pip install -e .
```

## 4. Verifikation (Pflicht vor dem Volltraining)

```bash
pytest tests/ -q                      # muss 84/84 grün sein
python scripts/build_manifest.py --bgad data/BGAD --out results/defect_detection/manifest --missing-mask-policy normal
                                      # Erwartung: 19 Bilder, 7 Tools, 8 good / 11 defect
python scripts/prepare_stage1_coco.py # train/val/test, pycocotools-Validierung OK
python scripts/run_stage2.py --method patchcore --split fixed --smoke --device cuda:0
                                      # Schnelltest GPU: image_auroc ~0.8 erwartet
```

## 5. Volltraining (in tmux, überlebt SSH-Abbruch)

```bash
tmux new -s kip
mkdir -p logs
```

Stage 1 — acht Läufe (Reihenfolge egal, sequenziell ausführen):

```bash
for AUG in on off; do
  python scripts/run_stage1.py --model yolo --aug $AUG --epochs 100 --imgsz 1088 \
    --batch 16 --device cuda:0 --seed 42 2>&1 | tee logs/yolo_aug$AUG.log
  python scripts/run_stage1.py --model yolo --aug $AUG --epochs 100 --imgsz 1088 \
    --batch 16 --device cuda:0 --seed 42 \
    --weights results/results/yolo_runs/C_synth_pretrain_real_finetune/weights/best.pt \
    2>&1 | tee logs/yolo_aug${AUG}_synthpre.log
  python scripts/run_stage1.py --model mask2former --aug $AUG --epochs 100 --batch 8 \
    --lr 1e-4 --freeze-backbone-epochs 20 --device cuda:0 --seed 42 \
    2>&1 | tee logs/m2f_aug${AUG}_lr1e4.log
  python scripts/run_stage1.py --model mask2former --aug $AUG --epochs 100 --batch 8 \
    --lr 5e-5 --freeze-backbone-epochs 30 --device cuda:0 --seed 42 \
    2>&1 | tee logs/m2f_aug${AUG}_lr5e5.log
done
```

Stage 2 — LOTO primär (fixed optional als Sekundärprotokoll):

```bash
python scripts/run_stage2.py --method patchcore --split loto --tile 256 --device cuda:0 --seed 42 2>&1 | tee logs/s2_patchcore.log
python scripts/run_stage2.py --method padim     --split loto --tile 256 --device cuda:0 --seed 42 2>&1 | tee logs/s2_padim.log
python scripts/run_stage2.py --method ae        --split loto --tile 256 --epochs 200 --device cuda:0 --seed 42 2>&1 | tee logs/s2_ae.log
python scripts/run_stage2.py --method unet --aug on  --split loto --tile 256 --epochs 300 --loss bce_dice --device cuda:0 --seed 42 2>&1 | tee logs/s2_unet_augon.log
python scripts/run_stage2.py --method unet --aug off --split loto --tile 256 --epochs 300 --loss bce_dice --device cuda:0 --seed 42 2>&1 | tee logs/s2_unet_augoff.log
```

tmux-Basics: `Strg+B, dann D` = abkoppeln (läuft weiter), `tmux attach -t kip` = wieder ankoppeln.

## 6. Ergebnisse sichern

```bash
python scripts/make_figures.py --stage all
cd /workspace/blum
zip -r kip_results_$(date +%Y%m%d).zip KIP/results/component_benchmark KIP/results/defect_detection KIP/results/figures KIP/logs
# Zip liegt dann im Netzlaufwerk-Ordner -> am Windows-Rechner herunterladen,
# in C:\Dev\KIP einspielen, prüfen (kein "smoke": true!) und committen.
```

## Grobe Zeitschätzung

Stage 1: YOLO-Läufe je ca. 0,5–1 h; Mask2Former je ca. 2–4 h → zusammen grob ein GPU-Tag.
Stage 2: PatchCore/PaDiM Minuten; AE/U-Net je nach Fold-Anzahl 1–3 h. Nicht parallel starten
(GPU-Speicher), einfach sequenziell in tmux durchlaufen lassen.
