# DGX-Runbook (Copy-Paste, in Reihenfolge)

Kompakte Befehlssequenz für die komplette DGX-Session. Ausführliche Erklärungen: `DGX_SETUP.md`.
Voraussetzung: Daten sind bereits ins Netzlaufwerk hochgeladen (Schritt 1 in DGX_SETUP,
**vorher `labels\*.cache` löschen**).

## 0. Verbinden, Repo, Umgebung

```bash
ssh root@141.3.142.150 -p 2632
cd /workspace/blum
git clone https://github.com/Marcimarc03/KIP.git   # oder: cd KIP && git pull
cd KIP
mkdir -p data
# hochgeladene Ordner/Zips nach data/ verschieben bzw. entpacken:
# mv ../BGAD data/ && mv ../object_segmentation_real_v3_1088 data/

python3.12 -m venv /workspace/blum/venv-kip
source /workspace/blum/venv-kip/bin/activate
python -c "import torch;print(torch.__version__,torch.cuda.is_available())" 2>/dev/null \
  || pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-dev.txt
pip install -e .
```

## 1. Verifikation (Pflicht — muss sauber durchlaufen)

```bash
pytest tests/ -q                       # 84/84 grün
python scripts/apply_stage1_split.py   # Erwartung: train=713 / val=101 / test=148, tool-disjunkt,
                                       # test deckt alle 6 realen Klassen (keine train-only-Klasse)
python scripts/prepare_stage1_coco.py  # train/val/test -> COCO, pycocotools-Validierung OK
python scripts/build_manifest.py --bgad data/BGAD --out results/defect_detection/manifest --missing-mask-policy normal
                                       # Erwartung: 19 Bilder, 7 Tools, 8 good / 11 defect
python scripts/run_stage2.py --method patchcore --split fixed --smoke --device cuda:0
                                       # GPU-Smoke: image_auroc ~0.8
```

## 2. Optionaler Vorab-Check — Mask2Former bei 1088 px?

```bash
python scripts/run_stage1.py --model mask2former --aug on --smoke --imgsz 1088 --batch 8 --device cuda:0
```
Kein CUDA-OOM → in Stage 1 bei den M2F-Läufen `--imgsz 1088` ergänzen (Fairness).
OOM → M2F bei Default 800 lassen und die Auflösungsdifferenz im Paper dokumentieren.

## 3. Session starten

```bash
tmux new -s kip
mkdir -p logs
```
(Abkoppeln: `Strg+B`, dann `D`. Wieder ankoppeln: `tmux attach -t kip`.)

## 4. Stage 1 — 7 Läufe (Primärvergleich + Ablationen)

```bash
# Primärvergleich (Augmentierung AN)
python scripts/run_stage1.py --model yolo --aug on --epochs 100 --imgsz 1088 --batch 16 --device cuda:0 --seed 42 2>&1 | tee logs/yolo_augon.log
python scripts/run_stage1.py --model mask2former --aug on --epochs 100 --batch 8 --lr 1e-4 --freeze-backbone-epochs 20 --device cuda:0 --seed 42 2>&1 | tee logs/m2f_augon.log

# Ablation 1 — Augmentierung AUS
python scripts/run_stage1.py --model yolo --aug off --epochs 100 --imgsz 1088 --batch 16 --device cuda:0 --seed 42 2>&1 | tee logs/yolo_augoff.log
python scripts/run_stage1.py --model mask2former --aug off --epochs 100 --batch 8 --lr 1e-4 --freeze-backbone-epochs 20 --device cuda:0 --seed 42 2>&1 | tee logs/m2f_augoff.log

# Ablation 2 — synthetisches Vortraining (nur YOLO), leakage-frei (A_synth_only!)
python scripts/run_stage1.py --model yolo --aug on --epochs 100 --imgsz 1088 --batch 16 --device cuda:0 --seed 42 --weights results/results/yolo_runs/A_synth_only/weights/best.pt 2>&1 | tee logs/yolo_augon_synthpre.log
python scripts/run_stage1.py --model yolo --aug off --epochs 100 --imgsz 1088 --batch 16 --device cuda:0 --seed 42 --weights results/results/yolo_runs/A_synth_only/weights/best.pt 2>&1 | tee logs/yolo_augoff_synthpre.log

# Optional — M2F-Lernraten-Check (nur bessere Konfig berichten)
python scripts/run_stage1.py --model mask2former --aug on --epochs 100 --batch 8 --lr 5e-5 --freeze-backbone-epochs 30 --device cuda:0 --seed 42 2>&1 | tee logs/m2f_augon_lr5e5.log
```

## 5. Optional — Sim-to-Real-Gap (Strategie A, kein Neutraining)

```bash
python scripts/eval_stage1_checkpoint.py --model yolo \
  --weights results/results/yolo_runs/A_synth_only/weights/best.pt \
  --imgsz 640 --device cuda:0 --tag strategyA_synth_only 2>&1 | tee logs/eval_strategyA.log
```

## 6. Stage 2 — Defekterkennung (LOTO)

```bash
python scripts/run_stage2.py --method patchcore --split loto --tile 256 --device cuda:0 --seed 42 2>&1 | tee logs/s2_patchcore.log
python scripts/run_stage2.py --method padim     --split loto --tile 256 --device cuda:0 --seed 42 2>&1 | tee logs/s2_padim.log
python scripts/run_stage2.py --method ae        --split loto --tile 256 --epochs 200 --device cuda:0 --seed 42 2>&1 | tee logs/s2_ae.log
python scripts/run_stage2.py --method unet --aug on  --split loto --tile 256 --epochs 300 --loss bce_dice --device cuda:0 --seed 42 2>&1 | tee logs/s2_unet_augon.log
python scripts/run_stage2.py --method unet --aug off --split loto --tile 256 --epochs 300 --loss bce_dice --device cuda:0 --seed 42 2>&1 | tee logs/s2_unet_augoff.log
```

## 7. Ergebnisse sichern

```bash
python scripts/make_figures.py --stage all
cd /workspace/blum
zip -r kip_results_$(date +%Y%m%d).zip KIP/results/component_benchmark KIP/results/defect_detection KIP/results/figures KIP/logs
# Zip über das Netzlaufwerk herunterladen, lokal in C:\Dev\KIP einspielen,
# prüfen (KEIN "smoke": true in den finalen JSONs!) und committen.
```

## Grobe Zeit

Stage 1: YOLO je ~0,5–1 h, Mask2Former je ~2–4 h → zusammen ~1 GPU-Tag.
Stage 2: PatchCore/PaDiM Minuten, AE/U-Net je 1–3 h. Sequenziell laufen lassen (GPU-Speicher).
