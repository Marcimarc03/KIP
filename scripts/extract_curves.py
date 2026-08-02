#!/usr/bin/env python
"""Trainingskurven aus allen Stage-1-Laeufen extrahieren (fuer Overfitting-Analyse).

Sammelt fuer jeden Lauf die Epochen-Kurven (train-loss, val-loss, val-mAP) und
schreibt:
  - je Lauf eine normierte CSV nach curves_out/<run>.csv  (zum Plotten)
  - eine Uebersicht curves_out/_summary.csv mit Konvergenz-Kennzahlen

Deckt YOLO (yolo_train/results.csv) und Mask2Former/Mask R-CNN
(curves/*.json ODER results.csv im run-dir) ab. Die Spaltennamen
unterscheiden sich je Framework und werden hier vereinheitlicht.

Nutzung:
    python scripts/extract_curves.py
    python scripts/extract_curves.py --base results/component_benchmark --out curves_out
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _flt(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _load_yolo(csv_path: Path) -> list[dict]:
    """Ultralytics results.csv -> Liste normierter Epochen-Dicts."""
    rows = list(csv.DictReader(open(csv_path)))
    out = []
    for r in rows:
        r = {k.strip(): v for k, v in r.items()}
        out.append({
            "epoch": _flt(r.get("epoch")),
            "train_loss": _flt(r.get("train/seg_loss")),      # Haupt-Trainingsverlust (Maske)
            "val_loss": _flt(r.get("val/seg_loss")),
            "train_box_loss": _flt(r.get("train/box_loss")),
            "val_box_loss": _flt(r.get("val/box_loss")),
            "val_map50": _flt(r.get("metrics/mAP50(M)")),
            "val_map50_95": _flt(r.get("metrics/mAP50-95(M)")),
        })
    return out


def _load_m2f(run_dir: Path) -> list[dict] | None:
    """Mask2Former/Mask R-CNN: curves/*.json ODER results.csv im run-dir."""
    # Variante 1: curves/ mit JSON-Dateien
    cdir = run_dir / "curves"
    if cdir.is_dir():
        files = sorted(cdir.glob("*.json"))
        if files:
            # jede Datei ist typ. {"epoch":[...], "train_loss":[...], "val_map50":[...]}
            merged: dict[str, list] = {}
            for fp in files:
                try:
                    d = json.loads(fp.read_text())
                    if isinstance(d, dict):
                        for k, v in d.items():
                            if isinstance(v, list):
                                merged.setdefault(k, v)
                except Exception:
                    pass
            if merged and "epoch" in merged:
                n = len(merged["epoch"])
                keys = ["epoch", "train_loss", "val_loss", "val_map50", "val_map50_95"]
                return [{k: (_flt(merged[k][i]) if k in merged and i < len(merged[k]) else None)
                         for k in keys} for i in range(n)]
    # Variante 2: flache results.csv
    csvp = run_dir / "results.csv"
    if csvp.exists():
        rows = list(csv.DictReader(open(csvp)))
        out = []
        for r in rows:
            r = {k.strip(): v for k, v in r.items()}
            # tolerante Spaltensuche
            def pick(*names):
                for nm in names:
                    if nm in r:
                        return _flt(r[nm])
                return None
            out.append({
                "epoch": pick("epoch", "step"),
                "train_loss": pick("train_loss", "train/loss", "loss"),
                "val_loss": pick("val_loss", "val/loss", "eval_loss"),
                "val_map50": pick("val_map50", "metrics/mAP50(M)", "segm_map50", "map50"),
                "val_map50_95": pick("val_map50_95", "metrics/mAP50-95(M)", "segm_map50_95"),
            })
        return out if out else None
    return None


def _stats(rows: list[dict]) -> dict:
    maps = [r["val_map50"] for r in rows if r.get("val_map50") is not None]
    vloss = [r["val_loss"] for r in rows if r.get("val_loss") is not None]
    tloss = [r["train_loss"] for r in rows if r.get("train_loss") is not None]
    st = {"n_epochs": len(rows)}
    if maps:
        bi = max(range(len(maps)), key=lambda i: maps[i])
        st["best_map50"] = round(maps[bi], 4)
        st["best_epoch"] = bi + 1
        st["last_map50"] = round(maps[-1], 4)
        st["best_minus_last"] = round(maps[bi] - maps[-1], 4)
    # Overfitting-Indikator: steigt val_loss im letzten Drittel, waehrend train_loss faellt?
    if len(vloss) >= 6 and len(tloss) >= 6:
        k = max(2, len(vloss) // 3)
        v_trend = vloss[-1] - vloss[-k]        # >0 => val-loss gestiegen
        t_trend = tloss[-1] - tloss[-k]        # <0 => train-loss gefallen
        st["val_loss_trend_last3rd"] = round(v_trend, 4)
        st["train_loss_trend_last3rd"] = round(t_trend, 4)
        st["overfit_flag"] = bool(v_trend > 0.02 and t_trend < -0.02)
    return st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(_ROOT / "results" / "component_benchmark"))
    ap.add_argument("--out", default=str(_ROOT / "curves_out"))
    args = ap.parse_args()

    base = Path(args.base)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(d for d in base.iterdir() if d.is_dir())
    summary = []

    for d in run_dirs:
        yolo_csv = d / "yolo_train" / "results.csv"
        if yolo_csv.exists():
            rows, kind = _load_yolo(yolo_csv), "yolo"
        else:
            rows = _load_m2f(d)
            kind = "m2f/other"
        if not rows:
            continue

        # normierte Kurven-CSV schreiben
        cols = ["epoch", "train_loss", "val_loss", "train_box_loss",
                "val_box_loss", "val_map50", "val_map50_95"]
        with open(out / f"{d.name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})

        st = _stats(rows)
        st["run"] = d.name
        st["kind"] = kind
        summary.append(st)

    # Uebersicht schreiben + drucken
    keys = ["run", "kind", "n_epochs", "best_epoch", "best_map50", "last_map50",
            "best_minus_last", "val_loss_trend_last3rd", "train_loss_trend_last3rd",
            "overfit_flag"]
    with open(out / "_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for s in summary:
            w.writerow({k: s.get(k, "") for k in keys})

    print(f"{'run':<52} {'ep':>3} {'bestE':>5} {'best':>6} {'last':>6} {'b-l':>7} {'vΔ':>7} {'tΔ':>7} of")
    print("-" * 110)
    for s in sorted(summary, key=lambda x: x["run"]):
        print(f"{s['run']:<52} {s.get('n_epochs',''):>3} "
              f"{s.get('best_epoch',''):>5} {s.get('best_map50',''):>6} "
              f"{s.get('last_map50',''):>6} {s.get('best_minus_last',''):>7} "
              f"{s.get('val_loss_trend_last3rd',''):>7} "
              f"{s.get('train_loss_trend_last3rd',''):>7} "
              f"{'!' if s.get('overfit_flag') else '.'}")
    print(f"\nKurven je Lauf: {out}/<run>.csv   Uebersicht: {out}/_summary.csv")
    print("Spalten b-l = best-minus-last (mAP50), vΔ/tΔ = val-/train-loss-Trend letztes Drittel")
    print("of = Overfitting-Flag (val-loss steigt UND train-loss faellt im letzten Drittel)")


if __name__ == "__main__":
    main()
