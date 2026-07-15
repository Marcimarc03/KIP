#!/usr/bin/env python
"""Aggregate Stage-1 multi-seed runs into mean +/- std per (model, aug, tag/init).

Rigor guard: within a group, all runs must share the same split fingerprint
(test_sha, n_test) and worker config. If not, the group is flagged INKONSISTENT
and must NOT be reported as a single mean (mixing configs = fake variance).

Usage:  python scripts/aggregate_stage1.py
"""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

SUMMARY = Path(__file__).resolve().parents[1] / "results/component_benchmark/summary.csv"


def g(row: dict, name: str) -> str:
    """Read a column tolerant to the 'metric.' prefix append_summary may add."""
    for k in (name, "metric." + name):
        v = row.get(k)
        if v not in (None, ""):
            return v
    return ""


def main() -> None:
    if not SUMMARY.exists():
        print(f"summary.csv nicht gefunden: {SUMMARY}")
        return
    rows = [r for r in csv.DictReader(open(SUMMARY))
            if str(g(r, "smoke")).lower() == "false"]

    groups: dict = defaultdict(list)
    for r in rows:
        aug = "on" if str(g(r, "augmentation")).lower() == "true" else "off"
        key = (g(r, "model"), aug, g(r, "tag") or "-", g(r, "init_weights") or "-")
        groups[key].append(r)

    def stat(rs, metric):
        vals = []
        for r in rs:
            try:
                vals.append(float(g(r, metric)))
            except (TypeError, ValueError):
                pass
        if not vals:
            return "  --  ", []
        if len(vals) == 1:
            return f"{vals[0]:.4f} (n=1)", vals
        return f"{statistics.mean(vals):.4f}±{statistics.stdev(vals):.4f}", vals

    print(f"{'model':12}{'aug':4}{'tag':13}{'n':>2}  "
          f"{'segm50 (mean+-std)':>20}{'segm50-95':>18}{'bbox50':>16}   Fingerprint")
    print("-" * 118)
    for key in sorted(groups):
        model, aug, tag, init = key
        rs = groups[key]
        shas = {g(r, "test_sha") for r in rs}
        ntests = {g(r, "n_test") for r in rs}
        workers = {g(r, "workers") for r in rs}
        seeds = [g(r, "seed") for r in rs]
        consistent = len(shas) <= 1 and len(ntests) <= 1 and len(workers) <= 1

        s50, v50 = stat(rs, "segm_map50")
        s5095, _ = stat(rs, "segm_map50_95")
        sbox, _ = stat(rs, "bbox_map50")
        fp = f"sha={sorted(shas)} n={sorted(ntests)} w={sorted(workers)}"
        flag = "" if consistent else "  <<< INKONSISTENT -> NICHT aggregieren"
        print(f"{model:12}{aug:4}{tag[:12]:13}{len(rs):>2}  "
              f"{s50:>20}{s5095:>18}{sbox:>16}   {fp}{flag}")
        # seed<->value paarweise sammeln (nicht zippen: sonst verrutscht die
        # Zuordnung, falls ein Lauf segm_map50 nicht hat)
        pair_list = []
        for r in rs:
            v = g(r, "segm_map50")
            try:
                pair_list.append((g(r, "seed"), f"{float(v):.4f}"))
            except (TypeError, ValueError):
                pair_list.append((g(r, "seed"), "--"))
        pairs = ", ".join(f"{sd}:{vv}" for sd, vv in sorted(pair_list))
        print(f"    seeds/segm50: {pairs}")

    print("\nRegeln: (1) nur konsistente Gruppen aggregieren. (2) Differenz zwischen")
    print("Modellen nur behaupten, wenn die Seed-Baender disjunkt sind (idealerweise")
    print("zusaetzlich per Bootstrap-CI ueber die 148 Testbilder abgesichert).")


if __name__ == "__main__":
    main()
