"""Build an Ultralytics detection dataset + cars.yaml from frames saved by tools.collect_frames.

  python -m tools.build_yolo_dataset --raw data/raw --out data/yolo --prelabel
  (review / fix the boxes in a labelling tool -- see below)
  yolo detect train model=yolo11n.pt data=cars.yaml imgsz=416 epochs=100 batch=16

Output layout (YOLO format, ONE class 'car' -- colour is decided later by Tier-1 classifier + VLM):
  data/yolo/images/{train,val}/*.png     data/yolo/labels/{train,val}/*.txt     cars.yaml (absolute path inside)

--prelabel runs the stock COCO detector at low confidence and writes its boxes as *proposals*. It will NOT find the
cars it is blind to (that is the reason you fine-tune), so open the folder in a labelling tool (labelImg, CVAT,
Roboflow, Label Studio; YOLO format, class 'car') and ADD the missing boxes / delete false ones. An image with an
empty .txt tells YOLO "no car here": do not train on unreviewed frames that contain cars.

The split is by contiguous blocks of frames (not random): consecutive frames are near-duplicates, and a random split
would leak them into val and inflate the metrics.

After training:  vln/config.py -> PerceptionCfg.detector_weights = "runs/detect/train/weights/best.pt"
                                  PerceptionCfg.det_classes = (0,)       # your single 'car' class
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
from typing import List, Sequence, Tuple


def yolo_line(box: Sequence[float], w: int, h: int, cls: int = 0) -> str:
    """Pixel xyxy -> 'cls xc yc bw bh' normalised to [0, 1] (clipped to the image)."""
    x1, y1, x2, y2 = (min(max(box[0], 0.0), w), min(max(box[1], 0.0), h),
                      min(max(box[2], 0.0), w), min(max(box[3], 0.0), h))
    return f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def split_blocks(files: List[str], val_frac: float) -> Tuple[List[str], List[str]]:
    n_val = max(1, int(round(len(files) * val_frac))) if len(files) > 1 else 0
    return files[:len(files) - n_val], files[len(files) - n_val:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw", help="folder of frames from tools.collect_frames")
    ap.add_argument("--out", default="data/yolo")
    ap.add_argument("--yaml", default="cars.yaml")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--prelabel", action="store_true", help="propose boxes with the stock detector")
    ap.add_argument("--conf", type=float, default=0.10, help="proposal confidence (low: you review them anyway)")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.raw, "*.png")) + glob.glob(os.path.join(a.raw, "*.jpg")))
    if not files:
        print(f"No frames in {a.raw}. Run first:  python -m tools.collect_frames --out {a.raw}", file=sys.stderr)
        return 1

    det = None
    if a.prelabel:
        import cv2  # noqa: F401  (heavy imports only when needed)
        from vln.config import PerceptionCfg
        from vln.stage5_perception import Detector
        det = Detector(PerceptionCfg(det_conf=a.conf))

    train, val = split_blocks(files, a.val_frac)
    n_boxes = 0
    for split, group in (("train", train), ("val", val)):
        os.makedirs(os.path.join(a.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(a.out, "labels", split), exist_ok=True)
        for f in group:
            name = os.path.basename(f)
            shutil.copy2(f, os.path.join(a.out, "images", split, name))
            lines: List[str] = []
            if det is not None:
                import cv2
                img = cv2.imread(f)
                h, w = img.shape[:2]
                lines = [yolo_line(d.box, w, h) for d in det(img)]
                n_boxes += len(lines)
            with open(os.path.join(a.out, "labels", split, os.path.splitext(name)[0] + ".txt"), "w") as fh:
                fh.write("\n".join(lines) + ("\n" if lines else ""))

    with open(a.yaml, "w") as fh:   # absolute path: Ultralytics resolves relative 'path' against its own datasets dir
        fh.write(f"path: {os.path.abspath(a.out).replace(os.sep, '/')}\n"
                 "train: images/train\nval: images/val\nnames:\n  0: car\n")
    print(f"{len(train)} train / {len(val)} val images -> {a.out}; {n_boxes} proposed boxes; wrote {a.yaml}")
    print("NEXT: review/correct the labels, then:  yolo detect train model=yolo11n.pt "
          f"data={a.yaml} imgsz=416 epochs=100 batch=16")
    return 0


if __name__ == "__main__":
    sys.exit(main())
