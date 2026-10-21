"""Run the whole perception stack on still images and draw what every stage thinks.

  python -m tools.debug_image --images debug/f_000480.jpg                 # one frame, all stages
  python -m tools.debug_image --images debug --out debug_out --no-vlm     # a folder, skip the slow VLM
  python -m tools.debug_image --images x.jpg --box 76 111 124 218         # force a region (e.g. Car_65) through classifier+VLM
  python -m tools.debug_image --images x.jpg --weights runs/detect/train/weights/best.pt --imgsz 416 640

Stages shown (same code and thresholds as the mission, taken from vln/config.py):
  YOLO   (green)  detector boxes at the first --imgsz, filtered to PerceptionCfg.det_classes   -> classifier p_red -> E1/E2/none
  RED    (red)    colour-blob proposer (goes straight to the VLM as E1 in the mission)
  BOX    (yellow) boxes you pass with --box
  VLM             P(yes) of Qwen2-VL on the padded crop (the exact crop the mission sends), threshold VerifierCfg.p_yes_threshold
The text report also lists the RAW detector output over ALL COCO classes (so you can see if a car is being called
'truck'/'bus'/'boat' and thrown away by the class filter) at every --imgsz you give.

Frames saved by --debug-dir are 2x upscaled with an overlay; they are resized back to the camera size (400x225) first,
because that is what the mission feeds the models.  Output: <out>/<name>_debug.png (frame + thumbnails of the crops).
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from vln.config import CLASSES, CameraCfg, PerceptionCfg, VerifierCfg
from vln.stage5_perception import crop_box, decide_event, detect_red_blobs

log = logging.getLogger("debug_image")
COL = {"yolo": (0, 200, 0), "red": (0, 0, 255), "box": (0, 220, 255)}      # BGR


@dataclass
class Item:
    src: str                                   # "yolo" | "red" | "box"
    box: Tuple[float, float, float, float]
    conf: float                                # detector confidence, or red fraction for blobs
    probs: Optional[np.ndarray] = None         # classifier probabilities in CLASSES order
    event: Optional[str] = None                # E1 | E2 | None (classifier decision; blobs are always E1)
    p_yes: Optional[float] = None              # VLM
    vlm_ms: Optional[float] = None
    crop: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def tag(self) -> str:
        return {"yolo": "YOLO", "red": "RED", "box": "BOX"}[self.src]


def load_frame(path: str, cam: CameraCfg) -> Optional[np.ndarray]:
    img = cv2.imread(path)
    if img is None:
        return None
    if (img.shape[1], img.shape[0]) != (cam.width, cam.height):
        img = cv2.resize(img, (cam.width, cam.height), interpolation=cv2.INTER_AREA)
    return img


def analyse_frame(img: np.ndarray, pc: PerceptionCfg, vc: VerifierCfg,
                  det: Optional[Callable], clf: Optional[Callable], vlm_p_yes: Optional[Callable],
                  manual_boxes: Sequence[Sequence[float]] = (), use_red: bool = True) -> List[Item]:
    """Pure pipeline logic with injectable models (so it can be unit-tested without torch)."""
    items: List[Item] = []
    if det is not None:
        for d in det(img):
            items.append(Item("yolo", d.box, d.conf))
    if use_red:
        for b in detect_red_blobs(img, pc):
            items.append(Item("red", b.box, b.conf, event="E1"))
    for b in manual_boxes:
        items.append(Item("box", tuple(float(v) for v in b), 1.0))
    for it in items:
        it.crop = crop_box(img, it.box, pc.crop_pad)
    if clf is not None:                                          # the mission classifies YOLO boxes only
        ys = [it for it in items if it.src in ("yolo", "box") and it.crop.size > 0]
        if ys:
            probs = clf([it.crop for it in ys])
            for it, p in zip(ys, probs):
                it.probs, it.event = p, decide_event(p, pc)
    if vlm_p_yes is not None:
        for it in items:
            ctx = crop_box(img, it.box, vc.context_pad)         # exactly what the mission hands to the VLM
            if ctx.size == 0:
                continue
            t0 = time.perf_counter()
            it.p_yes = float(vlm_p_yes(ctx))
            it.vlm_ms = (time.perf_counter() - t0) * 1000.0
            it.crop = ctx
    return items


def pipeline_verdict(items: List[Item], vc: VerifierCfg) -> str:
    """What the mission would do with this frame (Tier-1 pick, then the VLM verdict if it was run)."""
    best, score = None, -1.0
    for it in items:
        if it.src == "yolo" and it.probs is not None and it.event:
            s = float(it.probs[0])
        elif it.src == "red":
            s = it.conf
        else:
            continue
        if s > score:
            best, score = it, s
    if best is None:
        return "Tier-1: NO candidate -> nothing is sent to the VLM."
    msg = f"Tier-1: would submit {best.event} from {best.tag} (score {score:.2f})"
    if best.p_yes is not None:
        msg += f" -> VLM p_yes={best.p_yes:.2f} => {'TARGET' if best.p_yes >= vc.p_yes_threshold else 'rejected'}"
    return msg + "."


def render(img: np.ndarray, items: List[Item], vc: VerifierCfg, scale: int = 3, thumb: int = 170) -> np.ndarray:
    vis = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    for k, it in enumerate(items):
        x1, y1, x2, y2 = [int(v * scale) for v in it.box]
        c = COL[it.src]
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
        parts = [f"#{k} {it.tag} {it.conf:.2f}"]
        if it.probs is not None:
            parts.append(f"red {it.probs[0]:.2f}")
        if it.p_yes is not None:
            parts.append(f"VLM {it.p_yes:.2f}")
        cv2.putText(vis, " ".join(parts), (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    shown = [it for it in items if it.crop is not None and it.crop.size > 0][:7]
    w = max(vis.shape[1], len(shown) * (thumb + 6))
    canvas = np.full((vis.shape[0] + (thumb + 70 if shown else 0), w, 3), 30, np.uint8)
    canvas[:vis.shape[0], :vis.shape[1]] = vis
    for i, it in enumerate(shown):
        x0, y0 = i * (thumb + 6) + 3, vis.shape[0] + 4
        ch, cw = it.crop.shape[:2]
        s = thumb / max(ch, cw)
        t = cv2.resize(it.crop, (max(1, int(cw * s)), max(1, int(ch * s))), interpolation=cv2.INTER_CUBIC)
        canvas[y0:y0 + t.shape[0], x0:x0 + t.shape[1]] = t
        lines = [f"#{items.index(it)} {it.tag}"]
        if it.probs is not None:
            lines.append("r/o/n " + "/".join(f"{x:.2f}" for x in it.probs))
            lines.append(f"event {it.event or 'none'}")
        if it.p_yes is not None:
            lines.append(f"VLM {it.p_yes:.2f} {'YES' if it.p_yes >= vc.p_yes_threshold else 'no'}")
        for j, ln in enumerate(lines):
            cv2.putText(canvas, ln, (x0, y0 + thumb + 14 + 14 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    return canvas


def raw_yolo(det, img: np.ndarray, imgsz: int, conf: float, iou: float) -> List[Tuple[str, float, Tuple[int, ...]]]:
    """Unfiltered detector output over ALL classes at a given input size."""
    r = det.model.predict(img, imgsz=imgsz, conf=conf, iou=iou, classes=None, half=det.half,
                          device=det.dev.index if det.dev.type == "cuda" else "cpu", verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    xyxy, cf, cl = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)
    return [(str(r.names[int(c)]), float(s), tuple(int(v) for v in b)) for b, s, c in zip(xyxy, cf, cl)]


def collect_paths(inputs: Sequence[str]) -> List[str]:
    out: List[str] = []
    for p in inputs:
        if os.path.isdir(p):
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                out += sorted(glob.glob(os.path.join(p, ext)))
        else:
            out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="+", default=["debug"], help="image files and/or folders")
    ap.add_argument("--out", default="debug_out")
    ap.add_argument("--weights", default=None, help="detector weights (default PerceptionCfg.detector_weights)")
    ap.add_argument("--classifier", default=None, help="classifier weights (default PerceptionCfg.classifier_weights)")
    ap.add_argument("--imgsz", type=int, nargs="+", default=None, help="detector input sizes; first one drives the drawing")
    ap.add_argument("--conf", type=float, default=None, help="detector confidence (default PerceptionCfg.det_conf)")
    ap.add_argument("--box", type=float, nargs=4, action="append", metavar=("X1", "Y1", "X2", "Y2"), default=[],
                    help="extra region in 400x225 pixel coords to push through classifier + VLM (repeatable)")
    ap.add_argument("--color", default="red", help="target colour for the VLM prompt")
    ap.add_argument("--no-yolo", action="store_true")
    ap.add_argument("--no-classifier", action="store_true")
    ap.add_argument("--no-red", action="store_true", help="disable the colour-blob proposer")
    ap.add_argument("--no-vlm", action="store_true", help="skip Qwen2-VL (slow to load)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cam, pc, vc = CameraCfg(), PerceptionCfg(), VerifierCfg()
    if a.weights:
        pc.detector_weights = a.weights
    if a.classifier:
        pc.classifier_weights = a.classifier
    if a.conf is not None:
        pc.det_conf = a.conf
    sizes = a.imgsz or [pc.imgsz]
    pc.imgsz = sizes[0]

    paths = collect_paths(a.images)
    if not paths:
        print(f"No images found in {a.images}", file=sys.stderr)
        return 1

    det = clf = vlm = None
    if not a.no_yolo:
        try:
            from vln.stage5_perception import Detector
            det = Detector(pc)
        except Exception as exc:
            log.warning("Detector unavailable (%s) -> skipping YOLO.", exc)
    if not a.no_classifier:
        try:
            from vln.stage5_perception import CropClassifier
            clf = CropClassifier(pc)
        except Exception as exc:
            log.warning("Classifier unavailable (%s) -> skipping it.", exc)
    if not a.no_vlm:
        try:
            from vln.stage6_verifier import VLMVerifier
            vlm = VLMVerifier(vc, a.color)
        except Exception as exc:
            log.warning("VLM unavailable (%s) -> skipping it.", exc)

    os.makedirs(a.out, exist_ok=True)
    for path in paths:
        img = load_frame(path, cam)
        if img is None:
            log.warning("Cannot read %s", path)
            continue
        items = analyse_frame(img, pc, vc, det, clf, vlm.p_yes if vlm else None, a.box, use_red=not a.no_red)
        name = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(a.out, f"{name}_debug.png"), render(img, items, vc))
        print(f"\n=== {path} ({img.shape[1]}x{img.shape[0]}) ===")
        if not items:
            print("  no proposals from any stage")
        for k, it in enumerate(items):
            row = f"  #{k} {it.tag:<4} box={tuple(int(v) for v in it.box)} conf={it.conf:.2f}"
            if it.probs is not None:
                row += "  cls " + " ".join(f"{c}={p:.3f}" for c, p in zip(CLASSES, it.probs)) + f"  event={it.event or 'none'}"
            if it.p_yes is not None:
                row += f"  VLM p_yes={it.p_yes:.2f} ({it.vlm_ms:.0f} ms)"
            print(row)
        print("  " + pipeline_verdict(items, vc))
        if det is not None:
            for sz in sizes:
                raw = raw_yolo(det, img, sz, min(pc.det_conf, 0.10), pc.det_iou)
                shown = ", ".join(f"{n} {s:.2f}" for n, s, _ in raw[:8]) or "nothing"
                print(f"  raw YOLO (all classes, imgsz={sz}, conf>={min(pc.det_conf, 0.10):.2f}): {shown}")
    print(f"\nAnnotated images written to {os.path.abspath(a.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
