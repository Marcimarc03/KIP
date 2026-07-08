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

## Aktueller Stand (2026-07-07)

- Branch **`reconstruct-kip-data`** (Basis: main@a26d8ea). Enthält: Rekonstruktion von `kip/data/`
  (war wegen `.gitignore`-Pattern `data/` ohne Slash nie committet!), `.gitignore`-Fix,
  `scripts/make_test_split.py`, LF-Fix in `save_manifest`, Windows-Hinweis in requirements-dev.
- Rekonstruktion validiert: 84/84 pytest, `build_manifest.py` byte-identisch zur committeten
  Referenz; Smoke-Läufe reproduzieren die MPS-Referenzwerte (PatchCore/PaDiM/AE deterministisch
  bis ~7. Nachkommastelle, U-Net weicht wegen Gradiententraining plattformbedingt ab).
- **Haupt-Repo des Projekts ist `Marcimarc03/KIP`** (im Team abgeklärt, 2026-07-08).
  `justusschenk/KIP` ist nur noch historische Referenz (Remote `upstream`, read-only).
  Gearbeitet wird auf `main` des eigenen Repos (Remote `origin`).
- **Code-Ownership: Marc.** Das rekonstruierte `kip/data/` ist die maßgebliche Version für
  alles Weitere. Kein Abgleich und keine Wiederherstellung von Justus' Original.

## Umgebung (dieser Windows-Rechner)

- venv: `C:\venvs\kip` (Python 3.12) — NICHT im Repo, NICHT in OneDrive.
- **torch gepinnt auf 2.5.1+cpu** — torch 2.12.1 wirft `WinError 1114` (c10.dll) beim Import.
- Daten (alle gitignored): `data/BGAD/` (19 Defektbilder), `data/object_segmentation_real_v3_1088/`
  (771/101/90 nach `scripts/make_test_split.py`), `data/coco_converted/`.
- Backup der alten Repo-Version: `C:\Users\rothm\OneDrive\Desktop\Uni\KIP Seminar\Code\KIP_alt_backup`.

## Offene Aufgaben

1. CUDA-Volltraining nach `docs/REPRODUKTION.md` (8× Stage 1 à 100 Epochen; Stage 2 alle 4 Methoden
   LOTO, U-Net zusätzlich aug on/off) — erst dann existieren berichtbare Zahlen.
2. ~~Draft-PR~~ entfällt: `Marcimarc03/KIP` ist das Haupt-Repo. `reconstruct-kip-data` in
   `main` mergen und künftig auf `main` arbeiten.
3. `make_figures.py` + Analyse-Texte mit Volltraining-Zahlen aktualisieren.
4. Optional: Jetson Orin Nano Deployment (INT8/TensorRT) — nice-to-have, niedrigste Priorität.

## Befehle

```powershell
C:\venvs\kip\Scripts\Activate.ps1
pytest tests\ -q                                   # 84 Tests, muss grün sein
python scripts\build_manifest.py --bgad data\BGAD --out results\defect_detection\manifest --missing-mask-policy normal
python scripts\run_stage1.py --model {yolo|mask2former} --aug {on|off} --smoke --device cpu
python scripts\run_stage2.py --method {patchcore|padim|ae|unet} --split {loto|fixed} --smoke --device cpu
```
