#!/usr/bin/env python3
"""Task 21 step 2: render a trained ply's TRAIN or HELD-OUT views through the
trainer's own `evaluate()` path, into an `--eval-dump`-shaped directory.

    scripts/lpips_train_views.py --ply P.ply --colmap D/sparse/0 --images D/images
        [--masks D/masks] --max-resolution N [--num-downscales 0]
        --split {train,heldout} --n-views 24 --dump OUT

It renders and dumps; it does NOT score. Scoring is `bench/lpips_rescore.py`, so
that a train-view number and a held-out number come off the SAME chain -- the
one that reproduces the published lpips.json -- and the 0.03 comparison the
decision rule turns on is not between two different metrics.

THE SPLIT IS THE WHOLE MEASUREMENT. If `--split train` quietly scored held-out
views, train-view LPIPS would equal held-out LPIPS by construction, the
pre-registered reading ("within 0.03 => representation, not generalisation")
would fire, and nothing would look wrong. `select_views` is therefore a pure
function with its own tests, and its disjointness is asserted here at runtime as
well as in the suite.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def select_views(scene, split: str, n: int) -> list:
    """`n` evenly spaced views of `split`, or all of them if there are fewer.

    Evenly spaced rather than the first n: a walkthrough's first frames are one
    corner of one room, and `evaluate`'s own docstring records that scoring the
    first 10 of 21 views biased every number by up to ~0.5 dB.
    """
    if n < 1:
        raise ValueError(f"--n-views must be >= 1, got {n}")
    pool = {"train": scene.train, "heldout": scene.heldout}[split]
    if not pool:
        raise ValueError(f"split {split!r} is empty")
    if n >= len(pool):
        return list(pool)
    step = (len(pool) - 1) / (n - 1) if n > 1 else 0.0
    idx = sorted({int(round(i * step)) for i in range(n)})
    return [pool[i] for i in idx]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", default=None)
    ap.add_argument("--max-resolution", type=int, default=1920)
    ap.add_argument("--eval-split-every", type=int, default=8)
    ap.add_argument("--split", choices=["train", "heldout"], default="train")
    ap.add_argument("--n-views", type=int, default=24)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--report", default=None)
    a = ap.parse_args(argv)

    dump = Path(a.dump)
    if dump.exists() and any(dump.glob("*_render.png")):
        raise FileExistsError(f"refusing to write into a populated dump: {dump}")

    import torch
    from bench.lpips_attr import params_from_ply
    from metal_gauss.dataset import load_scene
    from metal_gauss.train import evaluate

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    scene = load_scene(a.colmap, a.images, max_resolution=a.max_resolution,
                       eval_split_every=a.eval_split_every, masks_dir=a.masks,
                       use_priors=False)
    picked = select_views(scene, a.split, a.n_views)

    heldout_names = {v.name for v in scene.heldout}
    picked_names = [v.name for v in picked]
    overlap = sorted(set(picked_names) & heldout_names)
    if a.split == "train" and overlap:
        raise RuntimeError(f"train selection overlaps the held-out split: {overlap[:5]}")
    if a.split == "heldout" and set(picked_names) - heldout_names:
        raise RuntimeError("heldout selection contains a non-held-out view")

    p, n = params_from_ply(a.ply, device)
    scene.heldout = picked                      # evaluate() iterates scene.heldout
    ev = evaluate(p, scene, device, sh_degree=a.sh_degree, active=n,
                  dump_dir=str(dump))
    rep = {"ply": a.ply, "n_splats": n, "split": a.split,
           "n_requested": a.n_views, "n_rendered": len(picked),
           "n_train_total": len(scene.train), "n_heldout_total": len(heldout_names),
           "overlap_with_heldout": overlap, "views": picked_names,
           "psnr_masked": ev["psnr_masked"], "psnr": ev["psnr"],
           "coverage": ev["coverage"], "dump": str(dump)}
    print(json.dumps({k: v for k, v in rep.items() if k != "views"}, indent=2))
    if a.report:
        rp = Path(a.report)
        if rp.exists():
            raise FileExistsError(f"refusing to overwrite {rp}")
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
