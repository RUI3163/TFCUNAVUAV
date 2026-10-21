"""Turn raw frames into a 3-class crop dataset  data/cls/{red_car,other_car,not_car}/*.png.

Labels are PROPOSALS (detector box + HSV redness heuristic). REVIEW AND CORRECT THEM by hand --
move misfiled crops between folders before training. not_car = detector false positives are NOT
captured here; add them by moving wrong 'other_car' crops, plus the random background patches
this script writes.

  python -m tools.build_classifier_dataset --raw data/raw --out data/cls
"""
from __future__ import annotations

import argparse
import glob
import os
import random

import cv2
import numpy as np

from vln.config import PerceptionCfg
from vln.stage5_perception import Detector, crop_box


def red_fraction(bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = (((hsv[..., 0] < 10) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 110) & (hsv[..., 2] > 60))
    return float(m.mean())


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/cls")
    ap.add_argument("--red-thresh", type=float, default=0.25)
    ap.add_argument("--bg-per-frame", type=int, default=2)
    a = ap.parse_args()
    for c in ("red_car", "other_car", "not_car"):
        os.makedirs(os.path.join(a.out, c), exist_ok=True)
    pc = PerceptionCfg(det_conf=0.10)
    det = Detector(pc)
    rng = random.Random(0)
    n = 0
    for path in sorted(glob.glob(os.path.join(a.raw, "*.png"))):
        img = cv2.imread(path)
        boxes = [d.box for d in det(img)]
        for i, b in enumerate(boxes):
            crop = crop_box(img, b, pc.crop_pad)
            if crop.size == 0:
                continue
            lab = "red_car" if red_fraction(crop) > a.red_thresh else "other_car"
            cv2.imwrite(os.path.join(a.out, lab, f"{os.path.basename(path)[:-4]}_{i}.png"),
                        cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA))
            n += 1
        h, w = img.shape[:2]
        for j in range(a.bg_per_frame):
            for _ in range(10):  # random 60x60 patch not overlapping any detection
                x, y = rng.randint(0, w - 60), rng.randint(0, h - 60)
                if all(iou((x, y, x + 60, y + 60), b) == 0 for b in boxes):
                    cv2.imwrite(os.path.join(a.out, "not_car", f"{os.path.basename(path)[:-4]}_bg{j}.png"),
                                cv2.resize(img[y:y + 60, x:x + 60], (96, 96)))
                    break
    print(f"wrote {n} vehicle crops (+ background) to {a.out}. NOW REVIEW THE LABELS.")


if __name__ == "__main__":
    main()
