"""Draw the YOLO labels onto the images so you can eyeball them, and list frames that need attention.

  python -m tools.preview_labels --data data/yolo            # writes data/yolo/preview/<split>/*.png + prints a summary
  python -m tools.preview_labels --data data/yolo --show     # step through in a window: any key = next, b = back, q = quit
  python -m tools.preview_labels --data data/yolo --only-empty --show   # only frames with NO box (the ones the detector missed)

Green box = a labelled car. Look for: cars with no box, boxes on non-cars (roofs, shadows, bushes), boxes that are
too loose/tight, and boxes cut off at the image border.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List, Tuple

import cv2


def read_boxes(label_path: str, w: int, h: int) -> List[Tuple[int, int, int, int, int]]:
    out = []
    if not os.path.exists(label_path):
        return out
    for ln in open(label_path).read().splitlines():
        p = ln.split()
        if len(p) != 5:
            continue
        c, xc, yc, bw, bh = int(p[0]), *map(float, p[1:])
        out.append((c, int((xc - bw / 2) * w), int((yc - bh / 2) * h), int((xc + bw / 2) * w), int((yc + bh / 2) * h)))
    return out


def draw(img, boxes):
    vis = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)   # 400x225 is tiny on screen
    for i, (_, x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(vis, (x1 * 3, y1 * 3), (x2 * 3, y2 * 3), (0, 255, 0), 2)
        cv2.putText(vis, str(i), (x1 * 3, max(14, y1 * 3 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return vis


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/yolo")
    ap.add_argument("--show", action="store_true", help="interactive window instead of writing preview images")
    ap.add_argument("--only-empty", action="store_true", help="only frames that have no boxes")
    a = ap.parse_args()

    items = []
    for split in ("train", "val"):
        for f in sorted(glob.glob(os.path.join(a.data, "images", split, "*.*"))):
            lab = os.path.join(a.data, "labels", split, os.path.splitext(os.path.basename(f))[0] + ".txt")
            items.append((split, f, lab))
    if not items:
        print(f"No images under {a.data}/images. Run tools.build_yolo_dataset first.", file=sys.stderr)
        return 1

    empty, total_boxes, rows = [], 0, []
    for split, f, lab in items:
        img = cv2.imread(f)
        h, w = img.shape[:2]
        boxes = read_boxes(lab, w, h)
        total_boxes += len(boxes)
        if not boxes:
            empty.append(os.path.basename(f))
        if a.only_empty and boxes:
            continue
        rows.append((split, f, draw(img, boxes)))

    print(f"{len(items)} images, {total_boxes} boxes, {len(empty)} images with NO box "
          f"({100 * len(empty) / len(items):.0f} %). If those frames do contain cars, label them.")
    if a.show:
        i = 0
        while 0 <= i < len(rows):
            cv2.imshow("labels (any key=next, b=back, q=quit)", rows[i][2])
            k = cv2.waitKey(0) & 0xFF
            i = i - 1 if k == ord("b") else i + 1
            if k == ord("q"):
                break
        cv2.destroyAllWindows()
    else:
        for split, f, vis in rows:
            d = os.path.join(a.data, "preview", split)
            os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, os.path.basename(f)), vis)
        print(f"Preview images written to {os.path.join(a.data, 'preview')}  (open the folder and flip through)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
