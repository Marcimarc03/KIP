# KIP — Visuelle Inspektion & Defekterkennung von Winkelschleifer-Komponenten (Kreislauffabrik)

KIT-Seminarprojekt. Zweistufige Pipeline: **Stage 1** Instanzsegmentierung von 9 Komponenten
(YOLO11n-seg vs. Mask2Former Swin-T), **Stage 2** Defekterkennung auf Spindel-Nahaufnahmen
(PatchCore, PaDiM, ConvAE unsupervised; U-Net supervised) bei nur 19 BGAD-Bildern.

## Wichtigste Referenzen (zuerst lesen)

- `docs/BUILD_PLAN.md` — **Single Source of Truth**: alle Interfaces, Datenfakten, Risiken. Strikt befolgen.
- `docs/REPRODUKTION.md` — Repro-Befehle (Smoke lokal, Volltraining CUDA-Server).
- `results/ergebnisse_analyse*.txt` — Ergebnisanalysen (AP2, AP3, Modellvergleich).

## Harte Regeln

- **NIE** in `results/results/*` oder `results/stage2_bgad/*` schreiben (Alt-Ergebnisse, read-only).
  Neue Outputs nur unter `results/component_benchmark/` bzw. `results/defect_detection/`.
- Keine Ergebnisse erfinden. Smoke-Läufe rechnen echte Metriken, tragen aber `"smoke": true`
  und sind NIE als finale Zahlen zitierbar.
- Splits strikt tool-basiert (LOTO primär), auf Bildebene VOR dem Tiling; `assert_no_tool_leakage`
  nicht umgehen. `fixed`-Split verletzt Tool-Disjunktheit by design (nur Sekundärprotokoll).
- Stage-1-Fairness: beide Modelle nur über `kip/stage1/evaluator.py` (pycocotools) bewerten,
  nie ultralytics-interne `model.val()`-Zahlen berichten. Box- und Segm-Metriken klar trennen.
- Klassenreihenfolge fixiert (`kip/__init__.py::CLASS_NAMES`), Spindel = Klasse 3.
- Defekttypen sind Metadaten — keine Multiclass-Claims (nur 2–5 Beispiele/Typ).
- Vor größeren Änderungen kurz zusammenfassen; vor Überschreiben/Löschen bestehender Dateien fragen.
- Kleine, nachvollziehbare Änderungen; alles muss im Seminar erklärbar sein.

## Aktueller Stand (2026-07-15)

- Haupt-Repo `Marcimarc03/KIP`, gearbeitet auf `main` (Remote `origin`).
  `justusschenk/KIP` = historische Referenz (read-only).
- **Beide Stages sind gerechnet.**
  - Stage 1 (Option-B-Split, test=tool98+tool10=148 Bilder): YOLO11n-seg vs. YOLO26-seg vs.
    Mask2Former Swin-T über den gemeinsamen pycocotools-Evaluator. Befund: **YOLO26 > M2F
    signifikant** (Bootstrap); YOLO11 ≈ M2F; YOLO26 vs. YOLO11 nicht robust trennbar.
    WICHTIG: Einzelläufe streuen ~0,04 segm-mAP50 (Trainings-Nichtdeterminismus) → Ergebnisse
    als **Mittel/Spanne + Bootstrap-CI** berichten, NIE Einzel-/Bestwerte (Cherry-Picking).
  - Stage 2 (BGAD, 19 Bilder, LOTO): PatchCore/PaDiM/ConvAE/U-Net gerechnet; Bildebene nicht
    belastbar (nur tool02 gemischt) → Pixel-/Regionenebene (AUPRO ~0,83) berichten;
    `fixed`-Split nur als Leakage-Anker.
- **Läuft gerade:** Multi-Seed-Studie Stage 1 (`scripts/run_multiseed_stage1.sh`, Seeds 42/1/2)
  auf der DGX — vom Betreuer freigegeben.
- **Neue Skripte:** `run_multiseed_stage1.sh` (Ein-Stück-Batch mit Split-/GPU-Guard),
  `aggregate_stage1.py` (Mittel±Std, verweigert Konfig-Mix), `bootstrap_stage1.py`
  (CI über Testbilder, gepaarter Vergleich). `run_stage1.py` stempelt jetzt
  Split-Fingerprint + Umgebung in metrics.json/summary.csv; Modelle: `{yolo|yolo26|mask2former|maskrcnn}`.
  `scripts/make_test_split.py` entfernt (deprecated; Split via `apply_stage1_split.py`).

## Umgebung (dieser Windows-Rechner)

- venv: `C:\venvs\kip` (Python 3.12) — NICHT im Repo, NICHT in OneDrive.
- **torch gepinnt auf 2.5.1+cpu** — torch 2.12.1 wirft `WinError 1114` (c10.dll) beim Import.
- Daten (alle gitignored): `data/BGAD/` (19 Defektbilder), `data/object_segmentation_real_v3_1088/`
  (713/101/148 nach `scripts/apply_stage1_split.py`, Option B: test=tool98+tool10), `data/coco_converted/`.
- Backup der alten Repo-Version: `C:\Users\rothm\OneDrive\Desktop\Uni\KIP Seminar\Code\KIP_alt_backup`.

## Offene Aufgaben

1. Multi-Seed-Studie Stage 1 auswerten (nach Durchlauf): `aggregate_stage1.py` +
   `bootstrap_stage1.py` → Ergebnistabelle Mittel±Std + CIs.
2. Paper finalisieren: AP2-Synthetik-/Real-Ablation im Bericht (S1 real-only / S2 synth-pretrain /
   S3 sim-to-real), echte Precision/Recall/IoU + Konfusionsmatrizen (statt der AR-Proxies im
   Evaluator), Auflösungs-Disclosure (YOLO 1088 vs. M2F/Mask R-CNN 800), Kap. 5/6 + Abstract
   einbauen, Kap. 4 entdoppeln.
3. `results/ergebnisse_analyse*.txt` + `make_figures.py` mit den finalen (Multi-Seed-)Zahlen
   aktualisieren.
4. Optional: Jetson Orin Nano Deployment (INT8/TensorRT) — laut Betreuer nice-to-have,
   niedrigste Priorität (argumentative Deployment-Tauglichkeit reicht).

## Befehle

```powershell
C:\venvs\kip\Scripts\Activate.ps1
pytest tests\ -q                                   # 84 Tests, muss grün sein
python scripts\build_manifest.py --bgad data\BGAD --out results\defect_detection\manifest --missing-mask-policy normal
python scripts\run_stage1.py --model {yolo|yolo26|mask2former|maskrcnn} --aug {on|off} --smoke --device cpu
python scripts\run_stage2.py --method {patchcore|padim|ae|unet} --split {loto|fixed} --smoke --device cpu
```

Multi-Seed Stage 1 (DGX, in tmux) + Auswertung:
```bash
bash scripts/run_multiseed_stage1.sh        # Seeds 42/1/2, Split-/GPU-Guard, ~24-39 h
python scripts/aggregate_stage1.py          # Mittel±Std je Modell (verweigert Konfig-Mix)
python scripts/bootstrap_stage1.py <run_dir> [<run_dir> ...]   # 95%-CI ueber Testbilder
```
