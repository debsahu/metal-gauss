#!/usr/bin/env python3
"""Where route (i)'s 18.96 ms/step goes: `_plane_depth` decomposed, at the production size.

The cost measurement (bench/results/plane_aux/throughput_pgeom.json) says plane-aux costs
20.7% more per step than centre depth at 500k splats / 1920x1440 -- four times the
pre-registered 5% threshold, against a plan that predicted "at 2.76 Mpx they should not".
That number says the marginal cost is real; it does not say where it is, and optimising
before measuring is precisely the mistake the plan made.

This script is versioned rather than a scratchpad one-liner because the Tier 0+1 numbers
came from scratchpad scripts and could not be reconciled from the record afterwards
(research/metal-gauss.md section 10).

Reports forward, backward and forward+backward separately, because the Tier 2 lesson
(section 10.2) is that the backward traversal was ~95% of a cost everyone had attributed
to the forward.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.runner import require_gpu_exclusive          # noqa: E402
from metal_gauss.geometry_loss import plane_depth_from_features  # noqa: E402


def timeit(fn, warmup: int, reps: int) -> float:
    """Median of `reps` after `warmup`, with an explicit sync around each block.

    CONTRIBUTING.md: Apple Silicon runs short bursts at boost clock, so the first
    measurement of anything is optimistic. bench/quick.py ramps for 2 s; the warm-up here
    is a count rather than a duration, so it is passed in and reported.
    """
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    ts = []
    for _ in range(reps):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.mps.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=1440)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    require_gpu_exclusive()
    dev = "mps"
    H, W = a.height, a.width
    g = torch.Generator().manual_seed(0)
    n_sum = torch.nn.functional.normalize(torch.randn(H, W, 3, generator=g), dim=-1).to(dev)
    n_sum[..., 2] = -n_sum[..., 2].abs()
    alpha = (torch.rand(H, W, generator=g) * 0.4 + 0.6).to(dev)
    n_sum = n_sum * alpha[..., None]
    off = (alpha * -2.5)[..., None].expand(H, W, 3).contiguous().to(dev)
    K = (1342.9, 1342.9, 966.4, 723.0)

    rows: dict[str, float] = {}

    # (1) the torch.cat that `_plane_depth` performs to build the (H,W,5) feature image,
    #     which plane_depth_from_features then slices straight back apart.
    def cat_only():
        torch.cat([n_sum, off[..., :1], alpha[..., None]], dim=-1)
    rows["cat_to_HW5_forward"] = timeit(cat_only, a.warmup, a.reps)

    feat = torch.cat([n_sum, off[..., :1], alpha[..., None]], dim=-1)

    # (2) the joint-finite sanitisation, which is a full read plus a full where.
    def sanitise():
        finite = torch.isfinite(feat).all(dim=-1)
        torch.where(finite[..., None], feat, torch.zeros_like(feat))
    rows["joint_finite_sanitise"] = timeit(sanitise, a.warmup, a.reps)

    # (3) the whole function, forward only.
    rows["plane_depth_forward"] = timeit(lambda: plane_depth_from_features(feat, *K),
                                         a.warmup, a.reps)

    # (4) the NORMAL output this trainer never uses. Like Brush (train.rs:1820-1834) we keep
    #     n_cam = normalize(n_sum/alpha) so one normal image feeds the prior-normal and
    #     consistency terms, and discard `_plane_normal`. It is computed every step.
    def normal_only():
        length = (n_sum * n_sum).sum(-1).clamp_min(1e-24).sqrt()
        nrm = n_sum / length[..., None]
        torch.where(alpha[..., None] > 0.5, nrm, torch.zeros_like(nrm))
    rows["unused_normal_output"] = timeit(normal_only, a.warmup, a.reps)

    # (5) forward + backward through the whole route (i) chain, which is what a training
    #     step actually pays. Section 10.2: the backward traversal was ~95% of a cost that
    #     had been attributed to the forward, so never report the forward alone.
    def fwd_bwd():
        ns = n_sum.detach().clone().requires_grad_(True)
        of = off.detach().clone().requires_grad_(True)
        f = torch.cat([ns, of[..., :1], alpha[..., None]], dim=-1)
        d, _, v = plane_depth_from_features(f, *K)
        (d * v).sum().backward()
    rows["route_i_forward_plus_backward"] = timeit(fwd_bwd, a.warmup, a.reps)

    out = {"height": H, "width": W, "megapixels": round(H * W / 1e6, 3),
           "warmup": a.warmup, "reps": a.reps, "unit": "ms (median)",
           "measured_marginal_ms_per_step": 18.959,
           "note": "measured_marginal_ms_per_step is the interleaved n=3 figure from "
                   "bench/results/plane_aux/throughput_pgeom.json, for comparison.",
           "rows": {k: round(v, 3) for k, v in rows.items()}}
    print(json.dumps(out, indent=2))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
