#!/usr/bin/env python
"""Apply a leave-tool-out train/val/test split for real_v3 (reproducible, idempotent).

Reorganises images/ and labels/ so that every tool ends up in its target split,
regardless of where the files currently sit. Running twice is a no-op.

Default target (Option A, 2026-07-09):
    test = {tool98}   -> only tool with all 6 real classes present
    val  = {tool03}   -> unchanged from delivered split
    train= everything else (incl. tool10, which moves back from test)

Rationale: the delivered split (train / val=tool03 / test=tool10) never tested
bearing_plate and shaft (they exist only on tool01/98/99). tool98 is the single
tool covering all 6 real classes, so it becomes the test set; tool99 stays in
train so bearing_plate/shaft remain trainable. See docs + seminar notes.

Usage:
    python scripts/apply_stage1_split.py [--dataset ...] [--dry-run]
    python scripts/apply_stage1_split.py --test-tools tool98 --val-tools tool03
"""
from __future__ import annotations

import argparse
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SPLITS = ("train", "val", "test")


def _tool_of(name: str) -> str | None:
    m = re.match(r"^(tool\d+)_", name)
    return m.group(1) if m else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",
                   default=str(_ROOT / "data" / "object_segmentation_real_v3_1088"))
    p.add_argument("--test-tools", nargs="+", default=["tool98"])
    p.add_argument("--val-tools", nargs="+", default=["tool03"])
    p.add_argument("--dry-run", action="store_true",
                   help="only report planned moves, change nothing")
    args = p.parse_args()

    ds = Path(args.dataset)
    test_tools, val_tools = set(args.test_tools), set(args.val_tools)

    def target_split(tool: str) -> str:
        if tool in test_tools:
            return "test"
        if tool in val_tools:
            return "val"
        return "train"

    for sub in ("images", "labels"):
        for s in _SPLITS:
            (ds / sub / s).mkdir(parents=True, exist_ok=True)

    moves = 0
    per_split_before: dict[str, set] = defaultdict(set)
    planned: list[tuple[str, str, str]] = []

    for sub in ("images", "labels"):
        for cur in _SPLITS:
            src_dir = ds / sub / cur
            if not src_dir.is_dir():
                continue
            for f in sorted(src_dir.iterdir()):
                if f.suffix == ".cache" or not f.is_file():
                    continue
                tool = _tool_of(f.name)
                if tool is None:
                    continue
                if sub == "images":
                    per_split_before[cur].add(tool)
                tgt = target_split(tool)
                if tgt != cur:
                    planned.append((str(f.relative_to(ds)), cur, tgt))
                    if not args.dry_run:
                        shutil.move(str(f), str(ds / sub / tgt / f.name))
                    moves += 1

    # Drop stale YOLO caches so labels get re-read (best-effort)
    if not args.dry_run:
        for c in (ds / "labels").glob("*.cache"):
            try:
                c.unlink()
            except OSError as e:
                print(f"  warn: could not delete {c.name} ({e}); "
                      f"delete it manually before training")

    print(f"{'[DRY-RUN] ' if args.dry_run else ''}target: "
          f"test={sorted(test_tools)} val={sorted(val_tools)} train=rest")
    print(f"planned moves: {len(planned)} files ({moves} incl. labels)")
    tool_moves = Counter((a, b) for _, a, b in planned if True)
    for (a, b), n in sorted(tool_moves.items()):
        print(f"  {a} -> {b}: {n} files")

    # Report + leakage check on final (or would-be) image folders
    print("\nfinal split (images):")
    seen: dict[str, str] = {}
    leak = False
    for s in _SPLITS:
        d = ds / "images" / s
        tools = sorted({_tool_of(f.name) for f in d.glob("*") if _tool_of(f.name)})
        n = sum(1 for f in d.glob("*") if f.is_file())
        print(f"  {s:5}: {n:4d} images | tools={tools}")
        for t in tools:
            if t in seen:
                print(f"  !! LEAKAGE: {t} in both {seen[t]} and {s}")
                leak = True
            seen[t] = s
    if not leak:
        print("  tool-disjoint OK (no tool in two splits)")


if __name__ == "__main__":
    main()
