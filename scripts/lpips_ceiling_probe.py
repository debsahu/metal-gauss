#!/usr/bin/env python3
"""Task 21 step 3: the photometric ceiling probe. NO TRAINING.

    scripts/lpips_ceiling_probe.py --dump DIR --scene S --arm A --out-root R
        [--fitters affine,bilagrid_tv10,bilagrid_tv0,ppisp] [--max-views N]
        [--device mps] [--bilagrid-steps 500] [--ppisp-steps 500]

For each held-out (render, gt) pair in an --eval-dump, fit POST HOC to the
FROZEN render, minimising L2 to the ground truth, and rescore LPIPS. The image
is clamped and rounded through uint8 before scoring, exactly as the baseline
PNGs were.

WHAT THIS BOUNDS, AND WHAT IT DOES NOT. Under this trainer's identity discipline
no appearance model ever touches a held-out render, so any gain one could
deliver is INDIRECT -- cleaner gaussians, not a corrected view. What the probe
measures is the PHOTOMETRIC COMPONENT of the held-out residual: the only
component any appearance model could reduce by any route. The reading is
therefore asymmetric, and it is asymmetric by design:

  * a SMALL ceiling is a definitive stop -- if a free, post-hoc, per-view fit
    cannot recover the LPIPS, nothing constrained and indirect can;
  * a LARGE ceiling is permission to build, never evidence that building works.

THE CONTROLS ARE NOT OPTIONAL. A fitter that silently does not fit returns
dLPIPS ~= 0, which is exactly the number a real null returns. `--synthetic-
control` injects each fitter's OWN family of distortion into the render, refits,
and reports the fraction of the induced dLPIPS recovered. A fitter below 0.90
there is broken and its null is not reportable.
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

import torch                                                          # noqa: E402

from bench import lpips_attr as LA                                    # noqa: E402
from bench.lpips_rescore import load_rgb, pairs, to_lpips             # noqa: E402

RECOVERY_FLOOR = 0.90          # pre-registered C1 bar


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(-10.0 * torch.log10(((a - b) ** 2).mean().clamp_min(1e-10)))


def make_metric(net: str, device: str):
    import lpips
    fn = lpips.LPIPS(net=net, verbose=False).eval().to(device)

    def m(a: torch.Tensor, b: torch.Tensor) -> float:
        with torch.no_grad():
            return float(fn(to_lpips(a).to(device), to_lpips(b).to(device)))
    return m


def _smooth_gain(h, w, device):
    """A low-frequency multiplicative field: what an appearance model is FOR,
    and what LPIPS is argued to be nearly blind to."""
    y = torch.linspace(-1, 1, h, device=device)[:, None]
    x = torch.linspace(-1, 1, w, device=device)[None, :]
    f = 1.0 + 0.22 * torch.cos(1.7 * x) * torch.cos(1.1 * y) - 0.13 * (x ** 2 + y ** 2)
    return (f[..., None] * torch.tensor([1.0, 0.95, 1.08], device=device))


def run_fitter(name, renders, gts, args):
    """-> (list of fitted images, list of per-view info)."""
    if name == "affine":
        out = [LA.fit_affine(r, g) for r, g in zip(renders, gts)]
        return [f for f, _ in out], [i for _, i in out]
    if name.startswith("bilagrid_tv"):
        tv = float(name[len("bilagrid_tv"):])
        out = [LA.fit_bilagrid(r, g, tv_weight=tv, steps=args.bilagrid_steps)
               for r, g in zip(renders, gts)]
        return [f for f, _ in out], [i for _, i in out]
    if name == "ppisp":
        fits, info = LA.fit_ppisp_shared(renders, gts, steps=args.ppisp_steps)
        return fits, [info] * len(fits)
    raise ValueError(f"unknown fitter {name!r}")


def synthetic_control(name, renders, metric, args, device):
    """C1. Corrupt the render with `name`'s OWN family, then refit toward the
    UNCORRUPTED render and report the fraction of the induced dLPIPS recovered.

    The target is the clean render, so LPIPS(target, target) == 0 and the induced
    distance IS the whole of what a working fitter must remove.
    """
    torch.manual_seed(0)
    clean = renders
    if name == "affine":
        m = torch.tensor([[1.09, 0.04, -0.02], [0.0, 0.94, 0.03],
                          [0.03, -0.01, 1.06]], device=device)
        b = torch.tensor([0.02, -0.012, 0.025], device=device)
        bad = [(r @ m.T + b).clamp(0, 1) for r in clean]
    elif name.startswith("bilagrid_tv"):
        bad = [(r * _smooth_gain(*r.shape[:2], device)).clamp(0, 1) for r in clean]
    elif name == "ppisp":
        vig = torch.tensor([[0.02, -0.01, -1.3, 0.25, 0.0]] * 3, device=device)
        crf = LA.crf_identity_raw(device).expand(3, 4).clone() + 0.35
        bad = [LA.apply_ppisp(r, vig, crf) for r in clean]
    else:
        raise ValueError(name)
    fits, _ = run_fitter(name, bad, clean, args)
    induced = [metric(LA.quantize(b), LA.quantize(c)) for b, c in zip(bad, clean)]
    left = [metric(LA.quantize(f), LA.quantize(c)) for f, c in zip(fits, clean)]
    rec = [(i - l) for i, l in zip(induced, left)]
    frac = [r / i if i > 1e-9 else float("nan") for r, i in zip(rec, induced)]
    ok = [f for f in frac if f == f]
    return {"induced_lpips_mean": statistics.fmean(induced),
            "residual_lpips_mean": statistics.fmean(left),
            "recovered_mean": statistics.fmean(rec),
            "recovered_fraction_mean": statistics.fmean(ok) if ok else float("nan"),
            "recovered_fraction_min": min(ok) if ok else float("nan"),
            "floor": RECOVERY_FLOOR,
            "passed": bool(ok) and statistics.fmean(ok) >= RECOVERY_FLOOR,
            "n_views": len(clean)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--fitters", default="affine,bilagrid_tv10,bilagrid_tv0,ppisp")
    ap.add_argument("--max-views", type=int, default=0, help="0 = all")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--net", default="vgg")
    ap.add_argument("--bilagrid-steps", type=int, default=LA.BILAGRID_STEPS)
    ap.add_argument("--ppisp-steps", type=int, default=500)
    ap.add_argument("--synthetic-control", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)

    dev = a.device if (a.device != "mps" or torch.backends.mps.is_available()) else "cpu"
    ps = pairs(Path(a.dump))
    if a.max_views:
        step = max(1, len(ps) // a.max_views)
        ps = ps[::step][:a.max_views]
    stems = [s for s, _, _, _ in ps]
    renders = [load_rgb(r).to(dev) for _, r, _, _ in ps]
    gts = [load_rgb(g).to(dev) for _, _, g, _ in ps]
    metric = make_metric(a.net, dev)

    base_l = [metric(r, g) for r, g in zip(renders, gts)]
    base_p = [psnr(r, g) for r, g in zip(renders, gts)]
    # C2, on the real data and through the real code path: an identity fitter
    # must return exactly 0.0, which catches a sign error in the delta.
    ident = LA.delta_lpips(metric, renders, [None] * len(renders), gts)
    if any(d != 0.0 for d in ident):
        raise RuntimeError(f"C2 identity control did not return 0: max {max(map(abs, ident))}")

    res = {"stage": "step3_ceiling", "scene": a.scene, "arm": a.arm, "tag": a.tag,
           "dump": str(a.dump), "device": dev, "net": a.net,
           "image_hw": list(renders[0].shape[:2]), "n_views": len(ps), "views": stems,
           "px_per_bilagrid_cell_bin":
               renders[0].shape[0] * renders[0].shape[1] / (16 * 16 * 8),
           "baseline": {"lpips_per_view": dict(zip(stems, base_l)),
                        "lpips_mean": statistics.fmean(base_l),
                        "psnr_per_view": dict(zip(stems, base_p)),
                        "psnr_mean": statistics.fmean(base_p)},
           "identity_control_c2": {"max_abs_delta": max(map(abs, ident)), "passed": True},
           "fitters": {}}

    for name in a.fitters.split(","):
        t0 = time.perf_counter()
        fits, info = run_fitter(name, renders, gts, a)
        qf = [LA.quantize(f) for f in fits]
        dl = LA.delta_lpips(metric, renders, qf, gts)
        fit_l = [b - d for b, d in zip(base_l, dl)]
        fit_p = [psnr(q, g) for q, g in zip(qf, gts)]
        dp = [f - b for f, b in zip(fit_p, base_p)]
        entry = {"delta_lpips_per_view": dict(zip(stems, dl)),
                 "delta_lpips_mean": statistics.fmean(dl),
                 "delta_lpips_min": min(dl), "delta_lpips_max": max(dl),
                 "lpips_after_mean": statistics.fmean(fit_l),
                 "delta_psnr_mean": statistics.fmean(dp),
                 "delta_psnr_min": min(dp), "delta_psnr_max": max(dp),
                 "psnr_after_mean": statistics.fmean(fit_p),
                 "mse_before_mean": statistics.fmean(i["mse_before"] for i in info),
                 "mse_after_mean": statistics.fmean(i["mse_after"] for i in info),
                 "n_params_per_view": info[0]["n_params"],
                 "wall_s": round(time.perf_counter() - t0, 1),
                 "info_first_view": info[0]}
        if a.synthetic_control:
            entry["synthetic_control_c1"] = synthetic_control(name, renders, metric, a, dev)
        res["fitters"][name] = entry
        c1 = entry.get("synthetic_control_c1")
        print(f"{a.scene}/{a.arm} {name:14s} dLPIPS {entry['delta_lpips_mean']:+.5f} "
              f"dPSNR {entry['delta_psnr_mean']:+.3f} dB  "
              f"mse {entry['mse_before_mean']:.5f}->{entry['mse_after_mean']:.5f}  "
              f"{entry['wall_s']}s", flush=True)
        if c1:
            print(f"    C1 recovered {c1['recovered_fraction_mean']:.3f} of induced "
                  f"{c1['induced_lpips_mean']:.4f}  passed={c1['passed']}", flush=True)

    out = LA.write_json(Path(a.out_root) / "step3" /
                        f"{a.scene}__{a.arm}{('__' + a.tag) if a.tag else ''}.json", res)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
