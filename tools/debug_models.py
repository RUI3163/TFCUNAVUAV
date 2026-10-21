"""Run YOLO detector -> classifier -> VLM on still images and draw what each model says.
Standalone: uses ONLY the original modules (Detector, CropClassifier, VLMVerifier, crop_box, decide_event),
so it works with the code you already have. Nothing else in the project needs to change.

  python -m tools.debug_models --images debug/f_000480.jpg
  python -m tools.debug_models --images debug --out debug_out --no-vlm        # whole folder, skip the slow VLM
  python -m tools.debug_models --images x.jpg --box 76 111 124 218           # force a region (e.g. Car_65) through classifier + VLM
  python -m tools.debug_models --images x.jpg --weights runs/detect/train-3/weights/best.pt --imgsz 416 640

Per box it shows: detector confidence -> classifier probabilities (red/other/not) + E1/E2/none -> VLM P(yes) on the
exact padded crop the mission would send. The text report also lists the RAW detector output over ALL classes, so you
can see whether a car is being called 'truck'/'bus'/'boat' and thrown away by PerceptionCfg.det_classes.
Frames saved by --debug-dir are 2x upscaled with an overlay; they are resized back to 400x225 first (what the models see).
  python -m tools.debug_models --images x.jpg --weights best.pt --det-classes 0 --imgsz 640   # fine-tuned single-class model
  python -m tools.debug_models --images x.jpg --no-yolo --no-classifier --box 76 111 124 218 --vlm-colors   # ask the VLM for the colour

NOTE: the pipeline view only draws boxes whose class id is in --det-classes (default PerceptionCfg.det_classes=(2,), COCO 'car').
A fine-tuned single-class model has its car at class 0, so pass --det-classes 0. Cyan boxes are the RAW, unfiltered detections.
Output: <out>/<name>_debug.png (frame + thumbnails of the crops each model received).
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
from vln.stage5_perception import crop_box, decide_event

log = logging.getLogger("debug_models")
COL = {"yolo": (0, 200, 0), "box": (0, 220, 255)}      # BGR: green = detector box, yellow = your --box


@dataclass
class Item:
    src: str                                   # "yolo" | "box"
    box: Tuple[float, float, float, float]
    conf: float
    probs: Optional[np.ndarray] = None         # classifier output in CLASSES order (red_car, other_car, not_car)
    event: Optional[str] = None                # E1 | E2 | None
    p_yes: Optional[float] = None              # VLM
    vlm_ms: Optional[float] = None
    colors: Optional[Dict[str, float]] = None  # VLM colour distribution (--vlm-colors)
    crop: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def tag(self) -> str:
        return "YOLO" if self.src == "yolo" else "BOX"


COLORS = ("Red", "Blue", "White", "Black", "Silver", "Gray", "Green", "Yellow", "Orange", "Brown")
COLOR_PROMPT = "What is the colour of the vehicle in the centre of this image? Answer with exactly one word."


def vlm_color_probs(vlm, bgr: np.ndarray, colors: Sequence[str] = COLORS) -> Dict[str, float]:
    """Ask the VLM for the colour and softmax its first-token logits over the colour words. Experimental: an open
    question avoids the yes-bias of 'Is this a RED car?'. Uses the loaded VLMVerifier's own model/processor."""
    from PIL import Image
    h, w = bgr.shape[:2]
    sc = vlm.cfg.min_side_px / max(1, min(h, w))
    if sc > 1.0:
        bgr = cv2.resize(bgr, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_CUBIC)
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": COLOR_PROMPT}]}]
    text = vlm.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = vlm.proc(text=[text], images=[img], return_tensors="pt").to(vlm.model.device)
    ids = [vlm.proc.tokenizer.encode(c)[0] for c in colors]
    if len(set(ids)) != len(ids):
        log.warning("Some colour words share a first token; their probabilities are merged arbitrarily.")
    with vlm.torch.inference_mode():
        logits = vlm.model(**inp).logits[0, -1].float()
    pr = vlm.torch.softmax(logits[ids], dim=0).cpu().numpy()
    return {c: float(x) for c, x in zip(colors, pr)}


def load_frame(path: str, cam: CameraCfg) -> Optional[np.ndarray]:
    img = cv2.imread(path)
    if img is None:
        return None
    if (img.shape[1], img.shape[0]) != (cam.width, cam.height):
        img = cv2.resize(img, (cam.width, cam.height), interpolation=cv2.INTER_AREA)
    return img


def analyse_frame(img: np.ndarray, pc: PerceptionCfg, vc: VerifierCfg, det: Optional[Callable],
                  clf: Optional[Callable], vlm_p_yes: Optional[Callable],
                  manual_boxes: Sequence[Sequence[float]] = (), vlm_colors: Optional[Callable] = None) -> List[Item]:
    items: List[Item] = []
    if det is not None:
        items += [Item("yolo", d.box, d.conf) for d in det(img)]
    items += [Item("box", tuple(float(v) for v in b), 1.0) for b in manual_boxes]
    for it in items:
        it.crop = crop_box(img, it.box, pc.crop_pad)                  # what the classifier sees
    if clf is not None:
        usable = [it for it in items if it.crop.size > 0]
        if usable:
            for it, p in zip(usable, clf([it.crop for it in usable])):
                it.probs, it.event = p, decide_event(p, pc)
    if vlm_p_yes is not None:
        for it in items:
            ctx = crop_box(img, it.box, vc.context_pad)               # exactly what the mission hands to the VLM
            if ctx.size == 0:
                continue
            t0 = time.perf_counter()
            it.p_yes = float(vlm_p_yes(ctx))
            it.vlm_ms = (time.perf_counter() - t0) * 1000.0
            it.crop = ctx
    if vlm_colors is not None:
        for it in items:
            ctx = crop_box(img, it.box, vc.context_pad)
            if ctx.size > 0:
                it.colors = vlm_colors(ctx)
                it.crop = ctx
    return items


def verdict(items: List[Item], vc: VerifierCfg) -> str:
    """What the mission would do: classifier picks the most red-like YOLO box; E1/E2 goes to the VLM."""
    cands = [it for it in items if it.src == "yolo" and it.probs is not None]
    if not cands:
        return "No YOLO box was classified -> nothing reaches the VLM."
    best = max(cands, key=lambda it: float(it.probs[0]))
    if not best.event:
        return f"Best YOLO box has p_red={best.probs[0]:.3f} -> no E1/E2 event -> nothing is sent to the VLM."
    msg = f"Would submit {best.event} (p_red={best.probs[0]:.2f})"
    if best.p_yes is not None:
        msg += f" -> VLM p_yes={best.p_yes:.2f} => {'TARGET' if best.p_yes >= vc.p_yes_threshold else 'rejected'}"
    return msg + "."


def render(img: np.ndarray, items: List[Item], vc: VerifierCfg, scale: int = 3, thumb: int = 170,
           raw: Sequence[Tuple[str, float, Tuple[int, ...]]] = ()) -> np.ndarray:
    vis = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    for name, cf, (rx1, ry1, rx2, ry2) in raw:                       # cyan = raw detector output, any class, unfiltered
        cv2.rectangle(vis, (rx1 * scale, ry1 * scale), (rx2 * scale, ry2 * scale), (255, 200, 0), 1)
        cv2.putText(vis, f"raw {name} {cf:.2f}", (rx1 * scale, ry2 * scale + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 200, 0), 1, cv2.LINE_AA)
    for k, it in enumerate(items):
        x1, y1, x2, y2 = [int(v * scale) for v in it.box]
        c = COL[it.src]
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
        parts = [f"#{k} {it.tag} {it.conf:.2f}"]
        if it.probs is not None:
            parts.append(f"red {it.probs[0]:.2f}")
        if it.p_yes is not None:
            parts.append(f"VLM {it.p_yes:.2f}")
        if it.colors:
            parts.append(max(it.colors, key=it.colors.get))
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
            lines += ["r/o/n " + "/".join(f"{x:.2f}" for x in it.probs), f"event {it.event or 'none'}"]
        if it.p_yes is not None:
            lines.append(f"VLM {it.p_yes:.2f} {'YES' if it.p_yes >= vc.p_yes_threshold else 'no'}")
        if it.colors:
            top = sorted(it.colors.items(), key=lambda kv: -kv[1])[:2]
            lines.append("colour " + " ".join(f"{c} {v:.2f}" for c, v in top))
        for j, ln in enumerate(lines):
            cv2.putText(canvas, ln, (x0, y0 + thumb + 14 + 14 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1,
                        cv2.LINE_AA)
    return canvas


def raw_yolo(det, img: np.ndarray, imgsz: int, conf: float, iou: float):
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
    ap.add_argument("--det-classes", type=int, nargs="+", default=None,
                    help="class ids kept by the detector (default PerceptionCfg.det_classes=(2,); use 0 for your fine-tuned model)")
    ap.add_argument("--vlm-colors", action="store_true", help="also ask the VLM for the colour (experimental)")
    ap.add_argument("--no-yolo", action="store_true")
    ap.add_argument("--no-classifier", action="store_true")
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
    if a.det_classes:
        pc.det_classes = tuple(a.det_classes)
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
        items = analyse_frame(img, pc, vc, det, clf, vlm.p_yes if vlm else None, a.box,
                              (lambda c: vlm_color_probs(vlm, c)) if (vlm and a.vlm_colors) else None)
        raws = {sz: raw_yolo(det, img, sz, min(pc.det_conf, 0.10), pc.det_iou) for sz in sizes} if det is not None else {}
        name = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(a.out, f"{name}_debug.png"), render(img, items, vc, raw=raws.get(sizes[0], [])))
        print(f"\n=== {path} ({img.shape[1]}x{img.shape[0]}) ===")
        if not items:
            print("  no boxes from the detector (and no --box given)")
        for k, it in enumerate(items):
            row = f"  #{k} {it.tag:<4} box={tuple(int(v) for v in it.box)} conf={it.conf:.2f}"
            if it.probs is not None:
                row += "  cls " + " ".join(f"{c}={p:.3f}" for c, p in zip(CLASSES, it.probs)) + f"  event={it.event or 'none'}"
            if it.p_yes is not None:
                row += f"  VLM p_yes={it.p_yes:.2f} ({it.vlm_ms:.0f} ms)"
            if it.colors:
                row += "  VLM colour " + " ".join(f"{c}={v:.2f}" for c, v in sorted(it.colors.items(), key=lambda kv: -kv[1])[:4])
            print(row)
        print("  " + verdict(items, vc))
        if det is not None:
            for sz in sizes:
                raw = raws[sz]
                shown = ", ".join(f"{n} {s:.2f}" for n, s, _ in raw[:8]) or "nothing"
                print(f"  raw YOLO (ALL classes, imgsz={sz}, conf>={min(pc.det_conf, 0.10):.2f}): {shown}")
    print(f"\nAnnotated images written to {os.path.abspath(a.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
