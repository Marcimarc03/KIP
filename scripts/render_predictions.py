#!/usr/bin/env python
"""Rendert vorhergesagte Instanzmasken aus einer COCO-Predictions-JSON auf ein Testbild.

Für qualitative YOLO-vs-Mask2Former-Vergleichsbilder aus den bereits gespeicherten
Vorhersagen (results/component_benchmark/<run>/predictions/coco_predictions.json).
Kein erneutes Inferieren nötig.

Wichtig: Für einen fairen Vergleich beide Modelle mit DEMSELBEN --image-name rendern.

Beispiel:
    python scripts/render_predictions.py \
      --pred results/component_benchmark/<run>/predictions/coco_predictions.json \
      --gt data/coco_converted/test.json \
      --images data/object_segmentation_real_v3_1088/images/test \
      --image-name tool10_motor_housing_0005.jpg \
      --score-thr 0.3 --out figures/qual_yolo.png
"""
import argparse, json, os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# category_id (1-basiert) -> Farbe
COLORS = {1: (230, 80, 75), 2: (60, 140, 230), 3: (30, 170, 120), 4: (200, 150, 20),
          5: (150, 80, 200), 6: (230, 120, 40), 7: (40, 180, 180), 8: (200, 60, 140),
          9: (120, 120, 120)}


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True, help="coco_predictions.json eines Laufs")
    p.add_argument("--gt", required=True, help="data/coco_converted/test.json")
    p.add_argument("--images", required=True, help="Bilderordner (images/test)")
    p.add_argument("--out", required=True)
    p.add_argument("--image-name", default=None,
                   help="Dateiname; ohne Angabe wird das Bild mit den meisten Detektionen gewählt")
    p.add_argument("--score-thr", type=float, default=0.3)
    return p.parse_args()


def main():
    a = _parse()
    gt = json.load(open(a.gt))
    catname = {c["id"]: c["name"] for c in gt["categories"]}
    id2img = {im["id"]: im for im in gt["images"]}
    name2id = {im["file_name"]: im["id"] for im in gt["images"]}

    preds = json.load(open(a.pred))
    by_img = {}
    for pr in preds:
        if pr.get("score", 1.0) >= a.score_thr:
            by_img.setdefault(pr["image_id"], []).append(pr)

    if a.image_name:
        if a.image_name not in name2id:
            raise SystemExit(f"ERROR: {a.image_name} nicht im Testset.")
        iid = name2id[a.image_name]
    else:
        if not by_img:
            raise SystemExit("ERROR: keine Detektionen ueber dem Schwellwert.")
        iid = max(by_img, key=lambda k: len(by_img[k]))

    fn = id2img[iid]["file_name"]
    img = Image.open(os.path.join(a.images, fn)).convert("RGB")
    W, H = img.size
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(16, W // 60))
    except Exception:
        font = ImageFont.load_default()

    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    dets = by_img.get(iid, [])
    for d in dets:
        col = COLORS.get(d["category_id"], (180, 180, 180))
        seg = d["segmentation"][0]
        pts = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
        od.polygon(pts, fill=col + (90,))
    img2 = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
    dr = ImageDraw.Draw(img2)
    for d in dets:
        col = COLORS.get(d["category_id"], (180, 180, 180))
        seg = d["segmentation"][0]
        pts = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
        dr.line(pts + [pts[0]], fill=col, width=3)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, y0 = min(xs), min(ys)
        lbl = f"{catname[d['category_id']]} {d.get('score', 0):.2f}"
        tb = dr.textbbox((0, 0), lbl, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ytop = max(0, y0 - th - 6)
        dr.rectangle([x0, ytop, x0 + tw + 8, ytop + th + 6], fill=col)
        dr.text((x0 + 4, ytop + 1), lbl, fill=(255, 255, 255), font=font)

    Path(os.path.dirname(a.out) or ".").mkdir(parents=True, exist_ok=True)
    img2.save(a.out)
    print(f"gespeichert: {a.out}  (Bild={fn}, {len(dets)} Detektionen >= {a.score_thr})")


if __name__ == "__main__":
    main()
