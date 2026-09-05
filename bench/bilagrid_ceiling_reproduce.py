#!/usr/bin/env python3
"""Task 22 control (a6): can THIS module reproduce Task 21's measured ceiling?

    bench/bilagrid_ceiling_reproduce.py --dump DIR --out JSON
        [--steps 500] [--lr 0.02] [--tv 10.0] [--max-views 25] [--device mps]

THE POINT. "Built wrong" and "built correctly but not worth adopting" return the
SAME trained-arm number, so they have to be separated by evidence that does not
depend on it. This is that evidence, and it was pre-registered as (a6) before any
number existed: fit THIS module's `slice_apply` and `tv_loss` post hoc to Task
21's own 25 held-out (render, gt) pairs, under Task 21's own protocol, and check
it recovers the ceiling Task 21 measured with DIFFERENT code on a DIFFERENT
branch. It certifies the model end to end against someone else's target,
independently of whether training can reach it.

TARGETS (bench/results/lpips_attribution/step3/pgeom__R1__converged.json and
__prereg.json, from feat/lpips-attribution):
    lr 0.02,  500 steps, tv 10  ->  dLPIPS +0.02667   ("converged")
    lr 2e-3,  500 steps, tv 10  ->  dLPIPS +0.02518   ("prereg")
Pre-registered acceptance band, that pair widened 10% each way: [0.0240, 0.0293].

PROTOCOL, matched to Task 21 exactly, because a protocol difference would show up
as a model difference: MSE + tv_weight * TV, Adam at the given lr with betas
(0.9, 0.999), the grid identity at init, ONE VIEW AT A TIME (holding 25 pairs of
2.76 Mpx resident on MPS took that process to 12 GB RSS on a full swap), and
every image clamped and ROUNDED THROUGH uint8 before scoring, so a gain that
exists only in un-rounded float is not counted.

NOT A TRAINING RUN. No splat is touched. No arm is graded here.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402
from PIL import Image                                                 # noqa: E402

from metal_gauss.bilagrid import identity_grids, slice_apply, tv_loss  # noqa: E402

BAND = (0.0240, 0.0293)


def load_rgb(p: Path) -> torch.Tensor:
    return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0)


def quantize(img: torch.Tensor) -> torch.Tensor:
    return (img.clamp(0.0, 1.0) * 255.0).round() / 255.0


def to_lpips(img: torch.Tensor) -> torch.Tensor:
    return (img * 2.0 - 1.0).permute(2, 0, 1)[None]


def fit_one(render, gt, *, dims, tv_weight, steps, lr, device):
    gx, gy, gl = dims
    r, g = render.to(device), gt.to(device)
    grid = identity_grids(1, dims, device=device).requires_grad_(True)
    opt = torch.optim.Adam([grid], lr=lr, betas=(0.9, 0.999))
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        out = slice_apply(grid, r)
        loss = ((out - g) ** 2).mean() + tv_weight * tv_loss(grid)
        loss.backward()
        opt.step()
    with torch.no_grad():
        fit = slice_apply(grid, r).cpu()
    del grid, opt, r, g
    if device == "mps":
        torch.mps.empty_cache()
    return fit.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--tv", type=float, default=10.0)
    ap.add_argument("--dims", type=int, nargs=3, default=[16, 16, 8])
    ap.add_argument("--max-views", type=int, default=25)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--net", default="vgg")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    dump = Path(a.dump)
    renders = sorted(dump.glob("*_render.png"))[:a.max_views]
    if not renders:
        sys.exit(f"no *_render.png in {dump}")
    names, R, G = [], [], []
    for rp in renders:
        gp = rp.with_name(rp.name.replace("_render", "_gt"))
        if not gp.exists():
            sys.exit(f"missing ground truth for {rp.name}")
        names.append(rp.stem[:-7]); R.append(load_rgb(rp)); G.append(load_rgb(gp))

    import lpips
    fn = lpips.LPIPS(net=a.net, verbose=False).eval()

    def metric(x, y):
        with torch.no_grad():
            return float(fn(to_lpips(x), to_lpips(y)))

    t0 = time.perf_counter()
    per_view, base_l, fit_l = {}, [], []
    for i, (nm, r, g) in enumerate(zip(names, R, G)):
        f = fit_one(r, g, dims=tuple(a.dims), tv_weight=a.tv, steps=a.steps,
                    lr=a.lr, device=a.device)
        b = metric(quantize(r), quantize(g))
        q = metric(quantize(f), quantize(g))
        base_l.append(b); fit_l.append(q)
        per_view[nm] = {"lpips_base": b, "lpips_fit": q, "delta_lpips": b - q,
                        "psnr_base": float(-10 * torch.log10(((quantize(r) - g) ** 2).mean().clamp_min(1e-10))),
                        "psnr_fit": float(-10 * torch.log10(((quantize(f) - g) ** 2).mean().clamp_min(1e-10)))}
        print(f"[{i+1}/{len(names)}] {nm} base {b:.4f} fit {q:.4f} "
              f"d {b - q:+.5f}  ({time.perf_counter() - t0:.0f}s)", flush=True)

    d = statistics.fmean(b - q for b, q in zip(base_l, fit_l))
    out = {"kind": "bilagrid_ceiling_reproduce", "schema": 1, "tag": a.tag,
           "dump": str(dump), "n_views": len(names), "net": a.net,
           "image_hw": list(R[0].shape[:2]),
           "config": {"steps": a.steps, "lr": a.lr, "tv_weight": a.tv,
                      "dims": a.dims, "device": a.device},
           "baseline_lpips": statistics.fmean(base_l),
           "fitted_lpips": statistics.fmean(fit_l),
           "delta_lpips": d,
           "delta_psnr": statistics.fmean(v["psnr_fit"] - v["psnr_base"]
                                          for v in per_view.values()),
           "acceptance_band": list(BAND),
           "in_band": bool(BAND[0] <= d <= BAND[1]),
           "wall_s": round(time.perf_counter() - t0, 1),
           "per_view": per_view}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\ndLPIPS {d:+.5f}  band {BAND}  IN BAND: {out['in_band']}  "
          f"baseline {out['baseline_lpips']:.4f} -> {out['fitted_lpips']:.4f}  "
          f"dPSNR {out['delta_psnr']:+.3f}  {out['wall_s']:.0f}s")


if __name__ == "__main__":
    main()
