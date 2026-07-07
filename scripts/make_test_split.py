#!/usr/bin/env python
"""Create the tool-based test split for real_v3 (standalone).

Replicates create_tool_based_split() from kip_train.py (tool10 -> test)
without triggering kip_train's module-level training pipeline.

Usage:
    python scripts/make_test_split.py [--dataset data/object_segmentation_real_v3_1088]
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _tool_of(name: str) -> str | None:
    m = re.match(r"^(tool\d+)_", name)
    return m.group(1) if m else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default=str(_ROOT / "data" / "object_segmentation_real_v3_1088"))
    parser.add_argument("--test-tools", nargs="+", default=["tool10"])
    args = parser.parse_args()

    ds = Path(args.dataset)
    img_val, lbl_val = ds / "images" / "val", ds / "labels" / "val"
    img_test, lbl_test = ds / "images" / "test", ds / "labels" / "test"

    if not img_val.is_dir():
        raise SystemExit(f"ERROR: {img_val} not found - copy the dataset first.")
    if img_test.is_dir() and any(img_test.iterdir()):
        print(f"Test split already exists ({sum(1 for _ in img_test.glob('*'))} images) - nothing to do.")
        return

    img_test.mkdir(parents=True, exist_ok=True)
    lbl_test.mkdir(parents=True, exist_ok=True)

    test_set = set(args.test_tools)
    moved = 0
    for img in sorted(img_val.glob("*")):
        if _tool_of(img.name) in test_set:
            shutil.move(str(img), str(img_test / img.name))
            lbl = lbl_val / (img.stem + ".txt")
            if lbl.exists():
                shutil.move(str(lbl), str(lbl_test / lbl.name))
            moved += 1

    for cache in (ds / "labels" / "val.cache", ds / "labels" / "test.cache"):
        if cache.exists():
            cache.unlink()

    print(f"moved {moved} -> test/   val={sum(1 for _ in img_val.glob('*'))}   "
          f"test={sum(1 for _ in img_test.glob('*'))}")


if __name__ == "__main__":
    main()
