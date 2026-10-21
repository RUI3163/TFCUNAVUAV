"""Fine-tune MobileNetV3-Small (ImageNet-pretrained) on 96x96 crops: red_car / other_car / not_car.

  python -m tools.train_classifier --data data/cls --epochs 20 --out weights/mobilenetv3_small_cls.pt
Colour is the discriminative cue -> NO hue jitter; orientation is arbitrary from above -> flips/rotations ON.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms as T

from vln.config import CLASSES

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cls")
    ap.add_argument("--out", default="weights/mobilenetv3_small_cls.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_t = T.Compose([T.Resize((96, 96)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                      T.RandomRotation(180), T.ColorJitter(brightness=0.3, contrast=0.3),
                      T.ToTensor(), T.Normalize(MEAN, STD)])
    va_t = T.Compose([T.Resize((96, 96)), T.ToTensor(), T.Normalize(MEAN, STD)])
    full = datasets.ImageFolder(a.data)
    assert set(full.classes) == set(CLASSES), f"need folders {CLASSES}, found {full.classes}"
    n_val = max(1, int(0.15 * len(full)))
    g = torch.Generator().manual_seed(0)
    tr_idx, va_idx = random_split(range(len(full)), [len(full) - n_val, n_val], generator=g)
    tr = torch.utils.data.Subset(datasets.ImageFolder(a.data, tr_t), tr_idx.indices)
    va = torch.utils.data.Subset(datasets.ImageFolder(a.data, va_t), va_idx.indices)
    tl, vl = DataLoader(tr, a.bs, shuffle=True, num_workers=2), DataLoader(va, a.bs, num_workers=2)

    counts = np.bincount([full.targets[i] for i in tr_idx.indices], minlength=len(full.classes)).astype(float)
    w = torch.tensor(counts.sum() / (len(counts) * np.maximum(counts, 1)), dtype=torch.float32, device=dev)

    m = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(full.classes))
    m.to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    lossf = nn.CrossEntropyLoss(weight=w)
    best = 0.0
    for ep in range(a.epochs):
        m.train()
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            lossf(m(x), y).backward()
            opt.step()
        sched.step()
        m.eval()
        conf = np.zeros((len(full.classes),) * 2, int)
        with torch.no_grad():
            for x, y in vl:
                p = m(x.to(dev)).argmax(1).cpu().numpy()
                for t, q in zip(y.numpy(), p):
                    conf[t, q] += 1
        acc = conf.trace() / max(1, conf.sum())
        print(f"epoch {ep + 1:02d} val_acc={acc:.3f}")
        if acc >= best:
            best = acc
            torch.save({"state_dict": {k: v.cpu() for k, v in m.state_dict().items()}, "classes": full.classes}, a.out)
    print("classes:", full.classes, "\nconfusion (rows=true):\n", conf, f"\nbest val_acc={best:.3f} -> {a.out}")


if __name__ == "__main__":
    main()
