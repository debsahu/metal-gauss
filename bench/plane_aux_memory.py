#!/usr/bin/env python3
"""Peak MPS driver memory for `--depth-source center` vs `plane-aux`.

WHY THIS EXISTS RATHER THAN A COLUMN IN THE ARM REPORTS. Peak driver memory is one of the
battery columns Task 19 requires, and the plan's suggested source -- "the trainer's own
`[mem]` lines" -- CANNOT SUPPLY IT. `train.py:819-821` samples
`torch.mps.driver_allocated_memory()` every 200 steps and prints ONLY when it exceeds
20 GB. On a healthy run there are no `[mem]` lines at all: every P-GEOM arm in this task
produced zero. A harness that read peak memory from those lines would silently report
nothing, and `grade()` drops non-numeric values, so the column would have vanished from the
battery without anyone noticing -- the exact shape CLAUDE.md calls the failure this project
keeps repeating.

So the number is measured directly here, in-process, at the production configuration, by
sampling every step and keeping the maximum. What the arms' logs independently establish is
the coarse bound the threshold gives: zero `[mem]` lines means no arm exceeded 20 GB.

Runs a short prefix of a real training run per arm -- long enough to pass the peak, which
is at the splat cap, so `--no-grow` pins capacity from step 0 and the peak is reached
immediately rather than at 70% of a 30k schedule.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.runner import require_gpu_exclusive     # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--depth-dir"); ap.add_argument("--normal-dir")
    ap.add_argument("--init-ply")
    ap.add_argument("--max-resolution", type=int, default=1920)
    ap.add_argument("--budget", type=int, default=500_000)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    require_gpu_exclusive()
    from metal_gauss import train as T

    out = {}
    for src in ("center", "plane-aux"):
        argv = ["--colmap", a.colmap, "--images", a.images,
                "--max-resolution", str(a.max_resolution), "--steps", str(a.steps),
                "--budget", str(a.budget), "--no-grow", "--num-downscales", "0",
                "--eval-split-every", "8", "--eval-every", str(a.steps),
                "--seed", "42", "--flatten-loss-weight", "1.0",
                "--depth-loss-weight", "1.0", "--normal-loss-weight", "0.2",
                "--depth-source", src]
        for flag, val in (("--depth-dir", a.depth_dir), ("--normal-dir", a.normal_dir),
                          ("--init-ply", a.init_ply)):
            if val:
                argv += [flag, val]
        args = T.build_parser().parse_args(argv)

        peak = 0.0
        real_step = T.render_view

        def spy(*aa, **kw):
            nonlocal peak
            peak = max(peak, torch.mps.driver_allocated_memory() / 1e9)
            return real_step(*aa, **kw)

        T.render_view = spy
        try:
            T.train(args)
        finally:
            T.render_view = real_step
        # One more sample after the run: the last steps are where the cap has bound
        # longest, and empty_cache() runs every 50 steps, so the in-loop maximum can miss
        # a transient. Reported separately rather than folded in, so the two cannot be
        # confused.
        out[src] = {"peak_driver_gb_in_loop": round(peak, 3),
                    "driver_gb_after_run": round(torch.mps.driver_allocated_memory() / 1e9, 3)}
        print(src, out[src], flush=True)

    c, p = out["center"]["peak_driver_gb_in_loop"], out["plane-aux"]["peak_driver_gb_in_loop"]
    out["delta_gb"] = round(p - c, 3)
    out["ratio"] = round(p / c, 4) if c else None
    out["steps"] = a.steps
    out["note"] = ("in-loop peak sampled at every render_view call, --no-grow so capacity "
                   "is pinned at the budget from step 0. The trainer's own [mem] lines "
                   "only fire above 20 GB and are absent on every arm, which independently "
                   "bounds both arms under 20 GB.")
    print(json.dumps(out, indent=2))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
