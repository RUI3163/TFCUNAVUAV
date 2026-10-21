"""Measure the Stage 6 VLM verifier on labelled crops BEFORE trusting it in flight.

  python -m tools.eval_verifier --data data/cls --color red
Uses the red_car folder as positives and the other folders as negatives; prints accuracy at the
configured threshold, the best threshold, AUROC, and latency. Small VLMs have been unreliable on
top-down sim imagery in your earlier llava experiments, so this number decides whether Stage 6 helps.
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import cv2
import numpy as np

from vln.config import VerifierCfg
from vln.stage6_verifier import VLMVerifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cls")
    ap.add_argument("--color", default="red")
    ap.add_argument("--max-per-class", type=int, default=100)
    ap.add_argument("--mode", choices=["color", "yesno"], default="color")
    a = ap.parse_args()
    cfg = VerifierCfg(mode=a.mode)
    v = VLMVerifier(cfg, a.color)
    scores, labels, lat = [], [], []
    for cls in ("red_car", "other_car", "not_car"):
        for path in sorted(glob.glob(os.path.join(a.data, cls, "*.png")))[: a.max_per_class]:
            img = cv2.imread(path)
            t0 = time.perf_counter()
            scores.append(v.p_yes(img))
            lat.append((time.perf_counter() - t0) * 1000)
            labels.append(1 if cls == "red_car" else 0)
    s, y = np.array(scores), np.array(labels)
    pos, neg = s[y == 1], s[y == 0]
    auroc = float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])) if len(pos) and len(neg) else float("nan")
    ths = np.linspace(0.05, 0.95, 19)
    accs = [((s >= t) == y).mean() for t in ths]
    print(f"n={len(y)} (pos {len(pos)}, neg {len(neg)})  AUROC={auroc:.3f}")
    print(f"acc@{cfg.p_yes_threshold}={((s >= cfg.p_yes_threshold) == y).mean():.3f}  best acc={max(accs):.3f} @ {ths[int(np.argmax(accs))]:.2f}")
    print(f"latency mean={np.mean(lat):.0f} ms  p95={np.percentile(lat, 95):.0f} ms")


if __name__ == "__main__":
    main()
