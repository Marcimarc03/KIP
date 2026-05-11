#!/usr/bin/env python3
"""Real-time visual inspection on NVIDIA Jetson Orin Nano (8 GB).

Pipeline:
    Camera frame -> YOLOv11n-seg (TensorRT INT8) -> per-component crop ->
    PatchCore (TensorRT FP16) -> overlay (green=OK / red=DEFECT) -> display.

This script is the production deployment artifact; the Jupyter notebook
(`notebooks/kip_inspection.ipynb`) covers training and evaluation.

On the dev machine the same code path falls back to ONNX Runtime (CPU/GPU)
if the TensorRT runtime is unavailable.

The --camera flag accepts either an integer device index (e.g. ``0``) or a
GStreamer pipeline string. Example for a Jetson IMX219 CSI camera:

    nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720 ! \
        nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! \
        video/x-raw,format=BGR ! appsink

The --calibrate-threshold flag runs the PatchCore engine over a directory of
known-OK crops and prints a recommended ``defect_threshold`` (mean + 3*std).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Optional imports - guarded so the script still loads on the dev machine.
try:
    import tensorrt as trt  # type: ignore
    import pycuda.autoinit  # type: ignore  # noqa: F401
    import pycuda.driver as cuda  # type: ignore
    HAS_TRT = True
except Exception:
    HAS_TRT = False

try:
    import onnxruntime as ort
    HAS_ORT = True
except Exception:
    HAS_ORT = False


CLASS_NAMES = [
    "anti-vibration_handle", "bearing_plate", "bevel_gear_drive",
    "bevel_gear_spindle", "gearbox_housing", "intermediate_gearbox",
    "motor_housing", "shaft", "wheel_guard",
]


# ---------------------------------------------------------------------------
# Backend abstractions
# ---------------------------------------------------------------------------

class TRTEngine:
    """Thin TensorRT wrapper supporting multi-output engines (e.g. YOLO-seg)."""

    def __init__(self, engine_path: Path):
        """Load a serialized .engine file and allocate I/O buffers for every binding."""
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings: list[int] = []
        # Inputs: list of (host_buf, dev_buf, shape, dtype, name)
        self.inputs: list[tuple] = []
        # Outputs: list of (host_buf, dev_buf, shape, dtype, name) preserved in binding order
        self.outputs: list[tuple] = []
        for i in range(self.engine.num_bindings):
            shape = tuple(self.engine.get_binding_shape(i))
            size = int(np.prod(shape))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            host = cuda.pagelocked_empty(size, dtype)
            dev = cuda.mem_alloc(host.nbytes)
            self.bindings.append(int(dev))
            name = self.engine.get_binding_name(i)
            entry = (host, dev, shape, dtype, name)
            if self.engine.binding_is_input(i):
                self.inputs.append(entry)
            else:
                self.outputs.append(entry)
        # Convenience aliases for single-input engines.
        if self.inputs:
            self.host_in = self.inputs[0][0]
            self.dev_in = self.inputs[0][1]
            self.in_shape = self.inputs[0][2]

    def infer(self, x: np.ndarray) -> tuple[np.ndarray, ...]:
        """Run forward pass; returns one ndarray per output binding (in binding order)."""
        np.copyto(self.host_in, x.ravel())
        cuda.memcpy_htod_async(self.dev_in, self.host_in, self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        for (host, dev, _, _, _) in self.outputs:
            cuda.memcpy_dtoh_async(host, dev, self.stream)
        self.stream.synchronize()
        return tuple(host.reshape(shape) for (host, _, shape, _, _) in self.outputs)


class ORTEngine:
    """ONNX Runtime fallback for the dev machine; returns list of outputs."""

    def __init__(self, onnx_path: Path):
        """Build an ORT InferenceSession with the best available provider."""
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(str(onnx_path), providers=providers)
        self.in_name = self.sess.get_inputs()[0].name

    def infer(self, x: np.ndarray) -> tuple[np.ndarray, ...]:
        """Run forward pass on a preprocessed NCHW float32 tensor; returns tuple of outputs."""
        return tuple(self.sess.run(None, {self.in_name: x}))


def load_backend(path: Path):
    """Pick TensorRT for .engine files, ONNX Runtime for .onnx files.

    If a requested ``.engine`` file is missing, falls back to a sibling ``.onnx``
    loaded via ORTEngine (with a warning) so the dev machine doesn't crash.
    """
    suffix = path.suffix.lower()
    if suffix == ".engine":
        if not path.exists():
            sibling = path.with_suffix(".onnx")
            if sibling.exists():
                print(f"[WARN] {path} not found, falling back to ONNX: {sibling}")
                return load_backend(sibling)
            raise FileNotFoundError(f"Neither {path} nor {sibling} exist.")
        if not HAS_TRT:
            sibling = path.with_suffix(".onnx")
            if sibling.exists() and HAS_ORT:
                print(f"[WARN] TensorRT not installed, loading ONNX fallback: {sibling}")
                return ORTEngine(sibling)
            raise RuntimeError(f"{path} requires TensorRT but it is not installed.")
        return TRTEngine(path)
    if suffix == ".onnx":
        if not HAS_ORT:
            raise RuntimeError(f"{path} requires onnxruntime but it is not installed.")
        return ORTEngine(path)
    raise ValueError(f"Unsupported model format: {path}")


# ---------------------------------------------------------------------------
# Pre-/post-processing
# ---------------------------------------------------------------------------

def letterbox(img: np.ndarray, new_shape: int = 640) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize-and-pad an image to a square; returns (image, scale, (pad_x, pad_y))."""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = (new_shape - new_unpad[0]) // 2, (new_shape - new_unpad[1]) // 2
    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    canvas[dh:dh + new_unpad[1], dw:dw + new_unpad[0]] = resized
    return canvas, r, (dw, dh)


def preprocess_yolo(img: np.ndarray, imgsz: int = 640) -> tuple[np.ndarray, float, tuple[int, int]]:
    """BGR uint8 -> NCHW float32 [0,1] with letterboxing."""
    canvas, r, pad = letterbox(img, imgsz)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]  # NCHW
    return np.ascontiguousarray(x), r, pad


def preprocess_patchcore(crop: np.ndarray, size: int = 224) -> np.ndarray:
    """Resize a crop to PatchCore's input size and ImageNet-normalize it."""
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    x = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))[None]
    return np.ascontiguousarray(x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically-stable sigmoid for numpy arrays."""
    return 1.0 / (1.0 + np.exp(-x))


def parse_yolo_seg(outputs: tuple[np.ndarray, ...], conf_thr: float, scale: float,
                   pad: tuple[int, int], orig_shape: tuple[int, int]):
    """Decode YOLOv11-seg outputs (boxes head + mask prototypes) into detections.

    Args:
        outputs: tuple of (boxes_tensor, protos_tensor) where
            boxes_tensor: (1, 4 + nc + 32, num_anchors)
            protos_tensor: (1, 32, 160, 160)
        conf_thr: confidence threshold
        scale: letterbox scale factor
        pad: letterbox (pad_x, pad_y)
        orig_shape: (h, w) of the original frame

    Returns:
        List of (xyxy, conf, cls_id, mask) tuples. ``mask`` is a uint8 binary
        mask the size of the bbox crop (or None if protos missing).
    """
    # Detect which output is the boxes head and which is the protos head by ndim.
    boxes_tensor = None
    protos_tensor = None
    for arr in outputs:
        if arr.ndim == 4:
            protos_tensor = arr
        else:
            boxes_tensor = arr
    if boxes_tensor is None:
        return []
    if boxes_tensor.ndim == 3:
        boxes_tensor = boxes_tensor[0]
    out = boxes_tensor.transpose(1, 0)  # (anchors, 4+nc+32)

    nc = len(CLASS_NAMES)
    boxes_xywh = out[:, :4]
    cls_scores = out[:, 4:4 + nc]
    mask_coeffs = out[:, 4 + nc:4 + nc + 32] if out.shape[1] >= 4 + nc + 32 else None
    cls_ids = cls_scores.argmax(axis=1)
    confs = cls_scores.max(axis=1)
    keep = confs > conf_thr
    boxes_xywh, cls_ids, confs = boxes_xywh[keep], cls_ids[keep], confs[keep]
    if mask_coeffs is not None:
        mask_coeffs = mask_coeffs[keep]

    if len(confs) == 0:
        return []

    # Keep an xywh copy for OpenCV NMS (which expects xywh, not xyxy).
    boxes_xywh_for_nms = np.stack([
        boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2,
        boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2,
        boxes_xywh[:, 2],
        boxes_xywh[:, 3],
    ], axis=1)

    # xywh (centered) -> xyxy in letterbox coords, then undo letterbox.
    xyxy = np.zeros_like(boxes_xywh)
    xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    xyxy[:, [0, 2]] -= pad[0]
    xyxy[:, [1, 3]] -= pad[1]
    xyxy /= scale
    h, w = orig_shape
    xyxy[:, 0::2] = xyxy[:, 0::2].clip(0, w - 1)
    xyxy[:, 1::2] = xyxy[:, 1::2].clip(0, h - 1)

    # Class-agnostic NMS for simplicity; OpenCV expects xywh (top-left + size).
    idxs = cv2.dnn.NMSBoxes(
        bboxes=boxes_xywh_for_nms.tolist(),
        scores=confs.tolist(),
        score_threshold=conf_thr,
        nms_threshold=0.45,
    )

    detections = []
    if len(idxs) == 0:
        return detections
    idxs = np.array(idxs).flatten()

    # Decode masks if we have protos and mask coefficients.
    full_masks = None
    if protos_tensor is not None and mask_coeffs is not None and len(idxs) > 0:
        protos = protos_tensor[0]                         # (32, 160, 160)
        c, mh, mw = protos.shape
        coeffs_kept = mask_coeffs[idxs]                   # (n, 32)
        proto_flat = protos.reshape(c, -1)                # (32, 160*160)
        mask_flat = _sigmoid(coeffs_kept @ proto_flat)    # (n, 160*160)
        full_masks = mask_flat.reshape(-1, mh, mw)        # (n, 160, 160)

    for k, i in enumerate(idxs):
        x1, y1, x2, y2 = xyxy[i].astype(int)
        crop_mask = None
        if full_masks is not None:
            mh, mw = full_masks.shape[1], full_masks.shape[2]
            # Map original-image bbox into the proto grid.
            mx1 = int(round(((x1 * scale) + pad[0]) / 640.0 * mw))
            my1 = int(round(((y1 * scale) + pad[1]) / 640.0 * mh))
            mx2 = int(round(((x2 * scale) + pad[0]) / 640.0 * mw))
            my2 = int(round(((y2 * scale) + pad[1]) / 640.0 * mh))
            mx1, my1 = max(0, mx1), max(0, my1)
            mx2, my2 = min(mw, mx2), min(mh, my2)
            if mx2 > mx1 and my2 > my1:
                sub = full_masks[k, my1:my2, mx1:mx2]
                if (x2 - x1) > 0 and (y2 - y1) > 0:
                    sub = cv2.resize(sub, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
                    crop_mask = (sub > 0.5).astype(np.uint8)
        detections.append((xyxy[i].astype(int), float(confs[i]), int(cls_ids[i]), crop_mask))
    return detections


def patchcore_score(out: np.ndarray) -> float:
    """Reduce a PatchCore output (per-pixel anomaly map or scalar) to a single score."""
    return float(out.max())


def calibrate_threshold(patchcore_engine, ok_crops_dir: Path) -> float:
    """Run PatchCore over a directory of known-OK crops and print a suggested threshold.

    Returns ``mean(scores) + 3*std(scores)`` which is a reasonable starting
    point for ``--defect-threshold``. Use this once after engine export.
    """
    crops = sorted(Path(ok_crops_dir).glob("*"))
    crops = [p for p in crops if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
    if not crops:
        raise FileNotFoundError(f"No image crops in {ok_crops_dir}")
    scores: list[float] = []
    for p in crops:
        img = cv2.imread(str(p))
        if img is None:
            continue
        x = preprocess_patchcore(img)
        out = patchcore_engine.infer(x)[0]
        scores.append(patchcore_score(out))
    if not scores:
        raise RuntimeError(f"Could not score any crops in {ok_crops_dir}")
    arr = np.array(scores, dtype=np.float32)
    mean, std = float(arr.mean()), float(arr.std())
    rec = mean + 3.0 * std
    print(f"[calibrate] n={len(arr)} mean={mean:.4f} std={std:.4f} max={arr.max():.4f}")
    print(f"[calibrate] recommended_threshold = mean + 3*std = {rec:.4f}")
    return rec


# ---------------------------------------------------------------------------
# Real-time loop
# ---------------------------------------------------------------------------

def _open_camera(camera_src):
    """Open an OpenCV VideoCapture from either an int index or a GStreamer string."""
    cap = cv2.VideoCapture(camera_src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_src!r}")
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def run_realtime_pipeline(camera_src, yolo_path: Path,
                          patchcore_path: Optional[Path],
                          conf: float = 0.25,
                          defect_threshold: float = 0.5,
                          imgsz: int = 640) -> None:
    """Open a camera and run the full detection + anomaly pipeline until 'q'."""
    yolo = load_backend(yolo_path)
    patchcore = load_backend(patchcore_path) if patchcore_path else None

    cap = _open_camera(camera_src)
    # These only apply for V4L2 / int-index sources; they are no-ops for GStreamer.
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    except Exception:
        pass

    fps_t0, fps_n, fps = time.time(), 0, 0.0
    fail_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                fail_count += 1
                if fail_count > 30:
                    print("Camera failed 30 frames in a row, exiting")
                    break
                continue
            fail_count = 0

            x, scale, pad = preprocess_yolo(frame, imgsz)
            yolo_out = yolo.infer(x)
            detections = parse_yolo_seg(yolo_out, conf, scale, pad, frame.shape[:2])

            # Collect valid crops for batched PatchCore inference.
            valid: list[tuple] = []  # list of (xyxy, score, cls_id, crop_input_tensor)
            for det in detections:
                xyxy, score, cls_id, _mask = det
                x1, y1, x2, y2 = xyxy
                if (x2 - x1) < 4 or (y2 - y1) < 4:
                    continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                pc_in = preprocess_patchcore(crop) if patchcore is not None else None
                valid.append((xyxy, score, cls_id, pc_in))

            anom_scores: list[float] = [0.0] * len(valid)
            if patchcore is not None and valid:
                # Stack into a single (N, 3, 224, 224) batch and run ONE inference.
                batch = np.concatenate([v[3] for v in valid], axis=0)
                batch = np.ascontiguousarray(batch)
                try:
                    pc_outs = patchcore.infer(batch)
                    pc_out = pc_outs[0]
                    # Split per-crop scores back. Assume first dim is batch.
                    for i in range(len(valid)):
                        anom_scores[i] = patchcore_score(pc_out[i])
                except Exception as exc:
                    # Fallback: per-crop inference (e.g. static-batch engine).
                    print(f"[WARN] batched PatchCore failed ({exc}); falling back to per-crop")
                    for i, v in enumerate(valid):
                        pc_out = patchcore.infer(v[3])[0]
                        anom_scores[i] = patchcore_score(pc_out)

            for (xyxy, score, cls_id, _pc_in), anom in zip(valid, anom_scores):
                x1, y1, x2, y2 = xyxy
                is_defect = (patchcore is not None) and (anom > defect_threshold)
                color = (0, 0, 255) if is_defect else (0, 200, 0)
                label = f"{CLASS_NAMES[cls_id]} {score:.2f}"
                if patchcore is not None:
                    label += f" | a={anom:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            fps_n += 1
            if fps_n >= 10:
                fps = fps_n / (time.time() - fps_t0)
                fps_t0, fps_n = time.time(), 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("KIP Inspection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    """Standard CLI for the deployment script."""
    p = argparse.ArgumentParser(description="KIP real-time visual inspection (Jetson).")
    p.add_argument("--yolo", type=Path, required=True,
                   help="Path to YOLOv11n-seg .engine (Jetson) or .onnx (dev).")
    p.add_argument("--patchcore", type=Path, default=None,
                   help="Optional PatchCore .engine/.onnx for defect detection.")
    p.add_argument("--camera", type=str, default="0",
                   help=("Camera source: integer V4L2 index (e.g. 0) OR a full "
                         "GStreamer pipeline string. Example for IMX219: "
                         "'nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,"
                         "height=720 ! nvvidconv ! video/x-raw,format=BGRx ! "
                         "videoconvert ! video/x-raw,format=BGR ! appsink'"))
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    p.add_argument("--defect-threshold", type=float, default=0.5,
                   help="Anomaly score above which a part is flagged as DEFECT.")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO input size.")
    p.add_argument("--calibrate-threshold", type=Path, default=None,
                   help=("Path to a directory of known-OK crops. If set, the script "
                         "runs PatchCore over them, prints recommended threshold "
                         "(mean + 3*std), and exits without opening the camera."))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Resolve --camera into either an int (V4L2 index) or a GStreamer string.
    try:
        cam_src = int(args.camera)
    except ValueError:
        cam_src = args.camera

    if args.calibrate_threshold is not None:
        if args.patchcore is None:
            raise SystemExit("--calibrate-threshold requires --patchcore <engine>")
        pc = load_backend(args.patchcore)
        calibrate_threshold(pc, args.calibrate_threshold)
    else:
        run_realtime_pipeline(
            camera_src=cam_src,
            yolo_path=args.yolo,
            patchcore_path=args.patchcore,
            conf=args.conf,
            defect_threshold=args.defect_threshold,
            imgsz=args.imgsz,
        )
