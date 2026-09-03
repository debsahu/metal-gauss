#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.5", "lpips>=0.1.4", "Pillow>=10", "numpy>=1.26"]
# ///
"""LPIPS (VGG, INRIA metrics.py convention) over a metal-gauss --eval-dump directory.

Full-frame on purpose: masked pixels render black against GT content, which is how every
historical LPIPS in earthbyte/slam was produced. Stage 4 gate: mean <= 0.25.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", type=Path)
    ap.add_argument("--net", default="vgg", choices=["vgg", "alex"])
    a = ap.parse_args()
    import lpips
    fn = lpips.LPIPS(net=a.net, verbose=False).eval()

    def load(p):
        arr = np.asarray(Image.open(p).convert("RGB"), np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1)[None]

    out = {}
    for r in sorted(a.dump.glob("*_render.png")):
        g = r.with_name(r.name.replace("_render", "_gt"))
        if not g.exists():
            sys.exit(f"missing ground truth for {r.name}")
        with torch.no_grad():
            out[r.stem[:-7]] = float(fn(load(r), load(g)))
    if not out:
        sys.exit(f"no *_render.png in {a.dump}")
    res = {"net": a.net, "per_view": out,
           "mean": float(np.mean(list(out.values()))), "n": len(out)}
    (a.dump / "lpips.json").write_text(json.dumps(res, indent=2))
    print(f"LPIPS({a.net}) mean {res['mean']:.4f} over {res['n']} views")


if __name__ == "__main__":
    main()
