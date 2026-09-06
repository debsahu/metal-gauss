#!/usr/bin/env python3
"""Task 21: calibrate LPIPS against a KNOWN degradation, so a number becomes a
statement about the reconstruction rather than a number.

    bench/lpips_degrade.py --dump DIR --scene S --out-root R
        [--sigmas 0.5,1,2,4] [--factors 2,3,4,6] [--max-views 25]

It scores DEGRADED GROUND TRUTH against ground truth -- the render is never
touched. Two degradations, both chosen because they are what a capacity-limited
splat model actually does to an image:

  blur:S      Gaussian blur of radius sigma pixels.
  resample:K  box-average down by K and nearest-upsample back. This is the
              closer analogue: a model with too few primitives per pixel cannot
              represent detail below its primitive spacing, which is a
              resolution loss, not a smoothing.

The output is a curve LPIPS(degradation). Reading an arm's LPIPS off it converts
"0.395" into "equivalent to losing a factor of K of linear detail", which is a
claim about the reconstruction that can be argued with. It does NOT prove the
cause is capacity: a blur and a bad reconstruction can land at the same LPIPS by
different routes. It bounds what the number MEANS, not what produced it.
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch                                                          # noqa: E402
import torch.nn.functional as F                                       # noqa: E402

from bench import lpips_attr as LA                                    # noqa: E402
from bench.lpips_rescore import box_downscale, load_rgb, pairs        # noqa: E402


def gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian, reflect-padded. sigma <= 0 is the identity."""
    if sigma <= 0:
        return img
    r = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-r, r + 1, dtype=img.dtype, device=img.device)
    k = torch.exp(-(x ** 2) / (2.0 * sigma * sigma))
    k = k / k.sum()
    t = img.permute(2, 0, 1)[None]                       # [1,3,H,W]
    c = t.shape[1]
    t = F.pad(t, (r, r, 0, 0), mode="reflect")
    t = F.conv2d(t, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    t = F.pad(t, (0, 0, r, r), mode="reflect")
    t = F.conv2d(t, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    return t[0].permute(1, 2, 0)


def resample(img: torch.Tensor, k: int) -> torch.Tensor:
    """Box-average down by k, nearest-upsample back to the ORIGINAL size.

    The size must be restored exactly: LPIPS compares two images and a
    resolution change would confound the degradation with the metric's own scale
    dependence (see `box_downscale`'s docstring).
    """
    if k <= 1:
        return img
    h, w, _ = img.shape
    small = box_downscale(img, k)
    up = small.repeat_interleave(k, 0).repeat_interleave(k, 1)
    out = img.clone()
    out[:up.shape[0], :up.shape[1]] = up                 # ragged edge keeps the original
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--sigmas", default="0.5,1,1.5,2,3,4,6")
    ap.add_argument("--factors", default="2,3,4,6,8")
    ap.add_argument("--max-views", type=int, default=25)
    ap.add_argument("--net", default="vgg")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)

    ps = pairs(Path(a.dump))
    if a.max_views:
        ps = ps[::max(1, len(ps) // a.max_views)][:a.max_views]
    gts = [load_rgb(g).to(a.device) for _, _, g, _ in ps]

    import lpips
    from bench.lpips_rescore import to_lpips
    fn = lpips.LPIPS(net=a.net, verbose=False).eval().to(a.device)

    def metric(x, y):
        with torch.no_grad():
            return float(fn(to_lpips(x).to(a.device), to_lpips(y).to(a.device)))

    ident = [metric(LA.quantize(g), LA.quantize(g)) for g in gts[:3]]
    if max(ident) > 1e-6:
        raise RuntimeError(f"LPIPS(gt, gt) is not zero: {ident} -- the chain is wrong")

    blur, res = {}, {}
    for s in [float(x) for x in a.sigmas.split(",") if x]:
        v = [metric(LA.quantize(gaussian_blur(g, s)), LA.quantize(g)) for g in gts]
        blur[f"{s:g}"] = statistics.fmean(v)
        print(f"blur sigma {s:>4g} px  LPIPS {blur[f'{s:g}']:.4f}", flush=True)
    for k in [int(x) for x in a.factors.split(",") if x]:
        v = [metric(LA.quantize(resample(g, k)), LA.quantize(g)) for g in gts]
        res[str(k)] = statistics.fmean(v)
        print(f"resample x{k:<4d}     LPIPS {res[str(k)]:.4f}", flush=True)

    out = LA.write_json(
        Path(a.out_root) / "degrade" / f"{a.scene}{('__' + a.tag) if a.tag else ''}.json",
        {"stage": "degrade_calibration", "scene": a.scene, "dump": str(a.dump),
         "net": a.net, "n_views": len(ps), "image_hw": list(gts[0].shape[:2]),
         "identity_lpips_max": max(ident),
         "blur_sigma_px": blur, "resample_factor": res})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
