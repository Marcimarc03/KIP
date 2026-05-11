#!/usr/bin/env bash
# export_to_jetson.sh
# Convert ONNX models to TensorRT engines on the Jetson Orin Nano.
# Run this ON the Jetson after copying the ONNX files over.
#
# IMPORTANT: run "sudo nvpmodel -m 0 && sudo jetson_clocks" before deploying
# for stable max-clock performance. Otherwise the Orin Nano will throttle and
# your measured FPS will be 30-50% lower than the engine's true ceiling.
#
# Usage:
#   bash export_to_jetson.sh /path/to/models_dir [/path/to/calib_images_dir]
#
# Requires: TensorRT (>=8.6) installed in JetPack, trtexec on PATH,
# Python with pycuda + tensorrt + opencv (for build_int8_cache.py).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${1:-./models}"
CALIB_DIR="${2:-./data/object_segmentation_real_v3_1088/images/train}"

YOLO_ONNX="${MODELS_DIR}/yolo11n_seg.onnx"
PATCHCORE_ONNX="${MODELS_DIR}/patchcore_resnet18.onnx"

BUILD_DIR="${MODELS_DIR}/build"
INT8_CACHE="${BUILD_DIR}/yolo11n_seg.int8.cache"
mkdir -p "${BUILD_DIR}"

echo "[1/3] Generating INT8 calibration cache for YOLO..."
python3 "${SCRIPT_DIR}/build_int8_cache.py" \
  --onnx "${YOLO_ONNX}" \
  --images "${CALIB_DIR}" \
  --cache_file "${INT8_CACHE}" \
  --batch 8 \
  --imgsz 640 \
  --max_images 500

echo "[2/3] Building YOLOv11n-seg TensorRT engine (INT8)..."
trtexec \
  --onnx="${YOLO_ONNX}" \
  --saveEngine="${MODELS_DIR}/yolo11n_seg.int8.engine" \
  --int8 --fp16 \
  --calib="${INT8_CACHE}" \
  --memPoolSize=workspace:1024 \
  --shapes=images:1x3x640x640 \
  --verbose

echo "[3/3] Building PatchCore TensorRT engine (FP16, dynamic batch)..."
# PatchCore is sensitive to quantization in the feature extractor; use FP16.
# Dynamic batch shapes let jetson_inference.py batch all crops from a frame
# into a single PatchCore call instead of N sequential ones.
trtexec \
  --onnx="${PATCHCORE_ONNX}" \
  --saveEngine="${MODELS_DIR}/patchcore_resnet18.fp16.engine" \
  --fp16 \
  --memPoolSize=workspace:1024 \
  --minShapes=input:1x3x224x224 \
  --optShapes=input:8x3x224x224 \
  --maxShapes=input:16x3x224x224 \
  --verbose

echo "Done. Engines written to ${MODELS_DIR}/"
echo "Run inference with: python3 jetson_inference.py --yolo ${MODELS_DIR}/yolo11n_seg.int8.engine --patchcore ${MODELS_DIR}/patchcore_resnet18.fp16.engine"
