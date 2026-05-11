#!/usr/bin/env python3
"""Build a TensorRT INT8 calibration cache for the YOLOv11n-seg engine.

trtexec's ``--calib`` flag expects a binary cache produced by an
``IInt8Calibrator`` implementation, NOT a directory of images. This helper
runs an ``IInt8EntropyCalibrator2`` over a folder of representative real
images, writes the resulting binary cache, and then trtexec can be invoked
with ``--int8 --calib=<cache_file>``.

Run ON the Jetson (or any machine with TensorRT + pycuda):

    python build_int8_cache.py \
        --images data/object_segmentation_real_v3_1088/images/train \
        --cache_file build/yolo11n_seg.int8.cache \
        --batch 8 --imgsz 640 --max_images 500
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import cv2

import tensorrt as trt
import pycuda.autoinit  # noqa: F401  (required for pycuda context init)
import pycuda.driver as cuda


def _list_images(image_dir: Path, max_images: int) -> list[Path]:
    """Return up to ``max_images`` image paths from ``image_dir`` (sorted)."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)
    return files[:max_images]


def _letterbox(img: np.ndarray, new_shape: int) -> np.ndarray:
    """Resize-and-pad ``img`` to ``new_shape`` x ``new_shape`` (BGR uint8)."""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = (new_shape - new_unpad[0]) // 2, (new_shape - new_unpad[1]) // 2
    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    canvas[dh:dh + new_unpad[1], dw:dw + new_unpad[0]] = resized
    return canvas


class ImageBatchStream:
    """Iterates calibration images and produces NCHW float32 batches in [0,1]."""

    def __init__(self, files: list[Path], batch: int, imgsz: int):
        self.files = files
        self.batch = batch
        self.imgsz = imgsz
        self.idx = 0

    def __len__(self) -> int:
        return (len(self.files) + self.batch - 1) // self.batch

    def reset(self) -> None:
        self.idx = 0

    def next_batch(self) -> np.ndarray | None:
        """Return next batch as (B, 3, H, W) float32 in [0,1] RGB, or None when done."""
        if self.idx >= len(self.files):
            return None
        chunk = self.files[self.idx:self.idx + self.batch]
        self.idx += self.batch
        out = np.empty((len(chunk), 3, self.imgsz, self.imgsz), dtype=np.float32)
        for i, p in enumerate(chunk):
            img = cv2.imread(str(p))
            if img is None:
                # Fill with mid-gray as a harmless fallback.
                img = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
            else:
                img = _letterbox(img, self.imgsz)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x = rgb.astype(np.float32) / 255.0
            out[i] = np.transpose(x, (2, 0, 1))
        # Pad short final batch up to ``self.batch`` so allocator stays consistent.
        if out.shape[0] < self.batch:
            pad = np.zeros((self.batch - out.shape[0], 3, self.imgsz, self.imgsz), dtype=np.float32)
            out = np.concatenate([out, pad], axis=0)
        return np.ascontiguousarray(out)


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """IInt8EntropyCalibrator2 that streams real images from disk."""

    def __init__(self, stream: ImageBatchStream, cache_file: Path, input_name: str = "images"):
        super().__init__()
        self.stream = stream
        self.cache_file = Path(cache_file)
        self.input_name = input_name
        self.device_input = cuda.mem_alloc(
            stream.batch * 3 * stream.imgsz * stream.imgsz * np.dtype(np.float32).itemsize
        )
        self._batches_seen = 0

    def get_batch_size(self) -> int:
        return self.stream.batch

    def get_batch(self, names):  # noqa: D401, ANN001
        """Copy next batch to device and return list of bindings (or None when done)."""
        batch = self.stream.next_batch()
        if batch is None:
            return None
        cuda.memcpy_htod(self.device_input, batch)
        self._batches_seen += 1
        if self._batches_seen % 10 == 0:
            print(f"  ... calibrated {self._batches_seen}/{len(self.stream)} batches")
        return [int(self.device_input)]

    def read_calibration_cache(self) -> bytes | None:
        if self.cache_file.exists():
            print(f"Reading existing cache: {self.cache_file}")
            return self.cache_file.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_bytes(cache)
        print(f"Wrote calibration cache ({len(cache)} bytes) -> {self.cache_file}")


def build_cache(onnx_path: Path, images: Path, cache_file: Path,
                batch: int, imgsz: int, max_images: int,
                workspace_mb: int = 1024) -> Path:
    """Build (or refresh) an INT8 calibration cache by running a calibration pass.

    We build a throwaway INT8 engine from ``onnx_path`` purely to trigger the
    calibrator and persist its cache to ``cache_file``. The engine itself is
    discarded; the real engine is built later by trtexec.
    """
    files = _list_images(images, max_images)
    if not files:
        raise FileNotFoundError(f"No calibration images found in {images}")
    print(f"Calibration pool: {len(files)} images @ {imgsz}x{imgsz}, batch={batch}")

    stream = ImageBatchStream(files, batch=batch, imgsz=imgsz)
    calibrator = EntropyCalibrator(stream, cache_file)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"Failed to parse ONNX:\n{errs}")

    config = builder.create_builder_config()
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))
    except AttributeError:
        config.max_workspace_size = workspace_mb * (1 << 20)  # older TRT
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator

    # Pin a fixed calibration shape via an optimization profile.
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    in_name = input_tensor.name
    profile.set_shape(in_name, (batch, 3, imgsz, imgsz),
                                (batch, 3, imgsz, imgsz),
                                (batch, 3, imgsz, imgsz))
    config.add_optimization_profile(profile)
    config.set_calibration_profile(profile)

    print("Building throwaway INT8 engine to drive calibration...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("INT8 engine build failed; cache may be incomplete.")
    print("Calibration done. Cache saved.")
    return cache_file


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Build a TensorRT INT8 calibration cache.")
    ap.add_argument("--onnx", type=Path, required=True,
                    help="Path to the YOLO ONNX file used for calibration.")
    ap.add_argument("--images", type=Path, required=True,
                    help="Directory of representative real images.")
    ap.add_argument("--cache_file", type=Path, required=True,
                    help="Output path for the binary INT8 cache.")
    ap.add_argument("--batch", type=int, default=8, help="Calibration batch size.")
    ap.add_argument("--imgsz", type=int, default=640, help="Calibration input size.")
    ap.add_argument("--max_images", type=int, default=500,
                    help="Maximum number of calibration images to use.")
    ap.add_argument("--workspace_mb", type=int, default=1024,
                    help="Workspace memory pool size in MB.")
    args = ap.parse_args()

    if not args.onnx.exists():
        raise FileNotFoundError(args.onnx)
    if not args.images.exists():
        raise FileNotFoundError(args.images)
    args.cache_file.parent.mkdir(parents=True, exist_ok=True)
    build_cache(args.onnx, args.images, args.cache_file,
                batch=args.batch, imgsz=args.imgsz,
                max_images=args.max_images, workspace_mb=args.workspace_mb)


if __name__ == "__main__":
    main()
