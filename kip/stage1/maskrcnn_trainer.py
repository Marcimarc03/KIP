"""Mask R-CNN (torchvision) trainer + COCO predictor for the Stage-1 benchmark.

Produces the same ``predictions/coco_predictions.json`` contract as the YOLO and
Mask2Former trainers, so the shared ``kip.stage1.evaluator.evaluate_coco``
compares all models on identical evaluation code.

Fairness-critical details
--------------------------
* Predicted ``category_id`` == COCO category id. torchvision reserves label 0 for
  background, so training labels are set to the COCO cat id directly and
  ``num_classes = max(cat_id) + 1``. Therefore the predicted label IS the COCO
  ``category_id`` -- NO ``+1`` shift (unlike YOLO, which is 0-based internally).
* Masks encoded as RLE (pycocotools); boxes as ``[x, y, w, h]``.
* ``box_score_thresh=0.0`` so COCO AP integrates over all scores.
* Standard operating point: ``min_size = cfg.imgsz`` (default 800, like M2F).

Usage (via run_stage1.py):
    trainer = MaskRCNNTrainer(cfg, train_json, val_json, images_dir, run_dir)
    ckpt = trainer.train()
    trainer.predict_to_coco(ckpt, coco_gt_json, images_dir, out_json)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from kip import CLASS_NAMES
from kip.config import Stage1Config

# Smoke: train on ~40 images so the path exercises end-to-end quickly.
_SMOKE_N = 40


class _CocoInstances(torch.utils.data.Dataset):
    """COCO ground truth -> (image tensor CHW in [0, 1], target dict).

    Target labels are the COCO category ids directly (see module docstring).
    Optional horizontal-flip augmentation adjusts boxes and masks consistently.
    """

    def __init__(self, coco: COCO, images_dir: Union[str, Path], augment: bool = False):
        self.coco = coco
        self.images_dir = Path(images_dir)
        self.augment = augment
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        img_id = self.ids[i]
        info = self.coco.imgs[img_id]
        bgr = cv2.imread(str(self.images_dir / info["file_name"]))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        boxes, labels, masks = [], [], []
        for a in anns:
            bx, by, bw, bh = a["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            boxes.append([bx, by, bx + bw, by + bh])
            labels.append(int(a["category_id"]))          # == COCO cat id
            masks.append(self.coco.annToMask(a))

        img = torch.as_tensor(rgb / 255.0, dtype=torch.float32).permute(2, 0, 1)
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if masks:
            masks_t = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        else:
            masks_t = torch.zeros((0, h, w), dtype=torch.uint8)

        if self.augment and boxes_t.numel() and torch.rand(1).item() < 0.5:
            img = torch.flip(img, dims=[2])
            masks_t = torch.flip(masks_t, dims=[2]) if masks_t.numel() else masks_t
            x1 = boxes_t[:, 0].clone()
            x2 = boxes_t[:, 2].clone()
            boxes_t[:, 0] = w - x2
            boxes_t[:, 2] = w - x1

        target = {
            "boxes": boxes_t,
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": masks_t,
            "image_id": torch.tensor([img_id]),
        }
        return img, target


class MaskRCNNTrainer:
    """Train torchvision Mask R-CNN and produce COCO-format predictions."""

    def __init__(
        self,
        cfg: Stage1Config,
        coco_train_json: Union[str, Path],
        coco_val_json: Union[str, Path],
        images_dir: Union[str, Path],
        run_dir: Union[str, Path],
    ):
        self.cfg = cfg
        self.device = cfg.device
        self.images_dir = Path(images_dir)
        self.run_dir = Path(run_dir)
        self._train_coco = COCO(str(coco_train_json))
        # Fixed class scheme (kip.CLASS_NAMES = 9 slots; label 0 doubles as
        # torchvision background, harmless as cat_id 0 never appears in real
        # data). Pinning to len(CLASS_NAMES) instead of max(cat_id)+1 guarantees
        # that a synth-pretrained checkpoint always transfers into the real
        # model with identical head sizes, regardless of which cat_ids a given
        # split happens to contain. For our data both equal 9.
        self.num_classes = len(CLASS_NAMES)
        self._model = None

    def _build(self):
        model = maskrcnn_resnet50_fpn_v2(
            weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
            box_score_thresh=0.0,               # COCO AP sweeps all scores
            min_size=self.cfg.imgsz,
            max_size=int(self.cfg.imgsz * 5 / 3),
        )
        in_feat = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, self.num_classes)
        in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feat_mask, 256, self.num_classes)
        return model.to(self.device)

    def train(self) -> Path:
        ds: torch.utils.data.Dataset = _CocoInstances(
            self._train_coco, self.images_dir, augment=self.cfg.augmentation
        )
        if self.cfg.smoke:
            ds = torch.utils.data.Subset(ds, list(range(min(_SMOKE_N, len(ds)))))
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=self.cfg.batch,
            shuffle=True,
            collate_fn=lambda b: tuple(zip(*b)),
        )
        model = self._build()
        # Optional synth-pretrained initialisation (leakage-free: the synthetic
        # data contains no real tool). Mirrors KIP_M2F_INIT for Mask2Former;
        # heads match because num_classes is fixed to len(CLASS_NAMES).
        init = os.environ.get("KIP_MASKRCNN_INIT")
        if init:
            model.load_state_dict(torch.load(init, map_location=self.device))
            print(f"[maskrcnn] Synth-Init geladen: {init}", flush=True)
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        # lr nach torchvision-Referenz auf batch skaliert (0.02 @ batch16 -> ~0.0025 @ batch2)
        base_lr = 0.0025
        opt = torch.optim.SGD(params, lr=base_lr, momentum=0.9, weight_decay=5e-4)

        ckpt = self.run_dir / "weights" / "maskrcnn.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        n_batches = max(1, len(loader))

        for epoch in range(self.cfg.epochs):
            running = 0.0
            for it, (imgs, targets) in enumerate(loader):
                # lineares LR-Warmup ueber die erste Epoche (Divergenzschutz)
                if epoch == 0:
                    for g in opt.param_groups:
                        g["lr"] = base_lr * min(1.0, (it + 1) / n_batches)
                imgs = [im.to(self.device) for im in imgs]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                loss = sum(model(imgs, targets).values())
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"[maskrcnn] nicht-finiter Loss (epoch {epoch}, iter {it}) -> Abbruch"
                    )
                opt.zero_grad()
                loss.backward()
                opt.step()
                running += float(loss.item())
            print(f"[maskrcnn] epoch {epoch + 1}/{self.cfg.epochs}  "
                  f"mean_loss={running / n_batches:.4f}", flush=True)
            if (epoch + 1) % 10 == 0:           # periodischer Checkpoint gegen Crash-Verlust
                torch.save(model.state_dict(), ckpt)

        torch.save(model.state_dict(), ckpt)
        self._model = model
        return ckpt

    @torch.no_grad()
    def predict_to_coco(
        self,
        ckpt: Union[str, Path],
        coco_gt_json: Union[str, Path],
        images_dir: Union[str, Path],
        out_json: Union[str, Path],
    ) -> Path:
        coco = COCO(str(coco_gt_json))
        if self._model is not None:
            model = self._model
        else:
            model = self._build()
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
        model.eval()
        images_dir = Path(images_dir)

        preds = []
        for img_id in coco.imgs:
            info = coco.imgs[img_id]
            bgr = cv2.imread(str(images_dir / info["file_name"]))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t = torch.as_tensor(rgb / 255.0, dtype=torch.float32).permute(2, 0, 1).to(self.device)
            out = model([t])[0]
            for box, label, score, mask in zip(
                out["boxes"], out["labels"], out["scores"], out["masks"]
            ):
                x1, y1, x2, y2 = box.tolist()
                binary = np.asfortranarray((mask[0].cpu().numpy() > 0.5).astype(np.uint8))
                rle = mask_utils.encode(binary)
                rle["counts"] = rle["counts"].decode("ascii")
                preds.append(
                    {
                        "image_id": int(img_id),
                        "category_id": int(label.item()),   # == COCO cat id (no shift)
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "segmentation": rle,
                        "score": float(score.item()),
                    }
                )

        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(preds))
        return out_json
