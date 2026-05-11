#!/usr/bin/env python3
"""Tag notebook cells with `skip-export` so nbconvert's TagRemovePreprocessor
can strip them when producing a deployable Python script.

The original notebook used `# %% [skip-export]` line markers, but those are
just comments — nbconvert only respects ``cell.metadata["tags"]``. This
script writes the proper metadata in-place.

Run:
    python tools/tag_cells.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "kip_inspection.ipynb"

# Cell indices (0-based) that should be tagged ``skip-export``. Anything not
# listed is kept in the converted script. Keep: imports (3), path constants
# (4), and the deployment ``__main__`` cell (31).
SKIP_EXPORT_INDICES: list[int] = [
    2,   # %pip install (no-op on Jetson)
    5,   # markdown: AP1 heading
    6,   # AP1: load_label_counts EDA
    7,   # AP1: bar plot
    8,   # AP1: dataset_summary
    9,   # AP1: show_samples (matplotlib)
    10,  # AP1: create_test_split
    11,  # AP1: write_eval_yaml
    12,  # markdown: AP2 heading
    13,  # AP2: YOLO config flags
    14,  # AP2: train_yolo / evaluate_yolo helpers
    15,  # AP2: Exp A training
    16,  # AP2: Exp B training
    17,  # AP2: Exp C training
    18,  # AP2: comparison plot
    19,  # AP2: show_predictions plot
    20,  # markdown: AP3 heading
    21,  # AP3: extract_component_crops
    22,  # AP3: train_patchcore + fallback
    23,  # AP3: detect_defect (used only by training/notebook PatchCore)
    24,  # AP3: PatchCore demo plot
    25,  # markdown: AP4 heading
    26,  # AP4: export YOLO ONNX
    27,  # AP4: export PatchCore ONNX
    28,  # AP4: CPU latency benchmark
    29,  # markdown: cost comparison table
    30,  # markdown: conversion notes
]

# Patches applied to specific cells before tagging. Each entry is
# (cell_index, search_text, replacement_text). KeyError if search_text not found.
PATCHES: list[tuple[int, str, str]] = [
    # FIX 5.5: PatchCore ONNX export must use dynamic batch axis so the engine
    # supports the batched (N,3,224,224) inference path in jetson_inference.py.
    (
        27,
        '        torch.onnx.export(extractor, dummy, str(onnx_path),\n'
        '                          input_names=["input"], output_names=["features"],\n'
        '                          opset_version=12, dynamic_axes=None)\n',
        '        torch.onnx.export(extractor, dummy, str(onnx_path),\n'
        '                          input_names=["input"], output_names=["features"],\n'
        '                          opset_version=12,\n'
        '                          dynamic_axes={"input": {0: "N"}, "features": {0: "N"}})\n',
    ),
    # FIX 4: rewrite the conversion-notes markdown so primary deployment is
    # the standalone script and nbconvert is shown as a fallback.
    (
        30,
        '## 5. Conversion Notes for Jetson\n\n'
        'Convert the notebook to a script (skipping cells tagged `skip-export`):',
        '## 5. Conversion Notes for Jetson\n\n'
        '**Primary deployment:** use `scripts/jetson_inference.py` directly. It is\n'
        'a standalone, dependency-light Python script (TensorRT + pycuda + opencv\n'
        'only) that does NOT need nbconvert and is the recommended entry point on\n'
        'the Jetson.\n\n'
        '**Fallback:** if you ever need to derive a script from this notebook,\n'
        'cells are properly tagged `skip-export` (see `tools/tag_cells.py`) so:',
    ),
]


def main() -> None:
    """Apply patches, write `skip-export` tag metadata, and save the notebook."""
    nb = nbformat.read(NOTEBOOK, as_version=4)
    n = len(nb.cells)
    print(f"Loaded notebook with {n} cells: {NOTEBOOK}")

    # 1) Apply targeted source patches.
    for idx, find, repl in PATCHES:
        if idx >= n:
            print(f"  [warn] patch idx {idx} out of range; skipping")
            continue
        src = nb.cells[idx].source
        if find not in src:
            print(f"  [warn] patch text not found in cell {idx}; skipping")
            continue
        nb.cells[idx].source = src.replace(find, repl)
        print(f"  [ok] patched cell {idx}")

    # 2) Tag cells with skip-export, preserving any pre-existing tags.
    tagged = 0
    for idx in SKIP_EXPORT_INDICES:
        if idx >= n:
            print(f"  [warn] skip-export idx {idx} out of range; skipping")
            continue
        meta = nb.cells[idx].metadata
        tags = list(meta.get("tags", []))
        if "skip-export" not in tags:
            tags.append("skip-export")
            meta["tags"] = tags
            tagged += 1
    print(f"Tagged {tagged} cells with 'skip-export'")

    # 3) Save back. Use validate_nb=False so we don't choke on missing exec counts.
    nbformat.write(nb, NOTEBOOK)
    print(f"Wrote: {NOTEBOOK}")

    # 4) Quick verification.
    nb2 = nbformat.read(NOTEBOOK, as_version=4)
    have = sum(1 for c in nb2.cells if "skip-export" in c.metadata.get("tags", []))
    print(f"Verification: {have} cells have 'skip-export' tag.")


if __name__ == "__main__":
    main()
