#!/usr/bin/env python3
"""Task 22: what does the bilateral grid cost per training step, really?

    bench/bilagrid_steptime.py --colmap DIR --images DIR --out JSON [--steps 800]

The plan pre-registers a Metal slice kernel as warranted only if the torch
version measures above 5% of the fixed-capacity step. This measures it THROUGH
THE TRAINER, not on a synthetic tensor, at both resolutions that matter:

  * `--num-downscales 0`  -- the full-resolution step, which is what a "5% of
    step time" line is naturally read against;
  * `--num-downscales 2`  -- the SCHEDULE AVERAGE the production protocol
    actually runs, where a third of the steps are at 1/16 of the pixels. The two
    differ by more than a factor of two and quoting the wrong one overstates or
    understates the cost badly, so both are reported and neither is called "the"
    number.

`require_gpu_exclusive()` first: a contended wall-clock number is not slightly
wrong, it is meaningless, and this file exists because an earlier synthetic probe
in this task was taken while ANOTHER AGENT'S GPU test suite was live on the same
machine.
"""
from __future__ import annotations

import argparse, json, os, statistics, sys, time
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.runner import require_gpu_exclusive          # noqa: E402


def one(argv, tag, tmp):
    from metal_gauss.train import build_parser, train
    rep = str(tmp / f"{tag}.json")
    a = build_parser().parse_args(argv + ["--report", rep])
    # `train()` IS NOT THE CLI. `main()` resolves several defaults after parsing,
    # and a harness that calls `train()` on a bare parser namespace inherits None
    # for every one of them. This block crashed exactly there --
    #   TypeError: '>' not supported between instances of 'NoneType' and 'int'
    # at train.py:612 -- but ONLY on the --num-downscales 2 arms, because the
    # dereference sits inside `if args.num_downscales > 0`. The four
    # --num-downscales 0 arms had already written clean reports, so the block was
    # half-done and looked healthy until it wasn't.
    #
    # This DUPLICATES train.py:1182-1192 rather than refactoring it, deliberately:
    # the Task 22 training arms are in flight from a frozen snapshot of train.py,
    # and editing it now would mean the branch's final train.py is not the one
    # that produced the arms. tests/test_train_recipe.py and
    # tests/test_geometry_loss.py hand-set the same field, so this is the third
    # place that works around it -- extracting a `resolve_defaults(args)` used by
    # main() and by every harness is the real fix and is left for after the arms.
    if a.resolution_schedule is None:
        a.resolution_schedule = max(1, a.steps // 3)
    for k in ("resolution_schedule", "budget", "steps"):
        if getattr(a, k) is None:
            raise SystemExit(f"{k} is None after parsing; train() will crash on it")
    t0 = time.perf_counter()
    out = train(a)
    return {"tag": tag, "ms_per_step": out["metrics"]["ms_per_step"],
            "wall_s": out["metrics"]["wall_s"],
            "measured_wall_s": round(time.perf_counter() - t0, 1),
            "n_splats": out["metrics"]["n_splats"],
            "appearance": out["metrics"]["appearance"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--colmap", required=True); ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--budget", type=int, default=500000)
    ap.add_argument("--max-resolution", type=int, default=1920)
    ap.add_argument("--repeats", type=int, default=2)
    a = ap.parse_args()
    require_gpu_exclusive()
    tmp = Path(a.out).parent; tmp.mkdir(parents=True, exist_ok=True)

    base = ["--colmap", a.colmap, "--images", a.images,
            "--max-resolution", str(a.max_resolution), "--steps", str(a.steps),
            "--budget", str(a.budget), "--eval-split-every", "8",
            "--eval-every", str(a.steps), "--seed", "42",
            # the recipe arm's geometry weights: the grid's RELATIVE cost depends
            # on what else the step is doing, and the arm it will be judged in is R1
            "--flatten-loss-weight", "1.0", "--depth-loss-weight", "1.0",
            "--normal-loss-weight", "0.2", "--depth-normal-weight", "0.05"]
    rows = []
    for nd in (0, 2):
        for mode in ("off", "bilagrid"):
            for rep in range(a.repeats):
                tag = f"nd{nd}__{mode}__r{rep}"
                argv = base + ["--num-downscales", str(nd), "--appearance", mode]
                rows.append(one(argv, tag, tmp))
                print(f"  {tag:24s} {rows[-1]['ms_per_step']:8.2f} ms/step", flush=True)

    def ms(nd, mode):
        v = [r["ms_per_step"] for r in rows if r["tag"].startswith(f"nd{nd}__{mode}__")]
        return {"mean": statistics.fmean(v), "min": min(v), "max": max(v),
                "spread": max(v) - min(v), "n": len(v), "values": v}
    summary = {}
    for nd in (0, 2):
        o, b = ms(nd, "off"), ms(nd, "bilagrid")
        d = b["mean"] - o["mean"]
        summary[f"num_downscales_{nd}"] = {
            "off": o, "bilagrid": b, "delta_ms": d,
            "delta_pct_of_off": 100.0 * d / o["mean"],
            "over_5pct_line": bool(d > 0.05 * o["mean"]),
            "line_ms": 0.05 * o["mean"],
            "x_over_line": d / (0.05 * o["mean"])}
    out = {"kind": "bilagrid_steptime", "schema": 1,
           "config": {"steps": a.steps, "budget": a.budget,
                      "max_resolution": a.max_resolution, "repeats": a.repeats,
                      "colmap": a.colmap},
           "rows": rows, "summary": summary}
    Path(a.out).write_text(json.dumps(out, indent=2))
    for k, v in summary.items():
        print(f"{k}: off {v['off']['mean']:.2f} (spread {v['off']['spread']:.2f})  "
              f"bilagrid {v['bilagrid']['mean']:.2f} (spread {v['bilagrid']['spread']:.2f})  "
              f"delta {v['delta_ms']:+.2f} ms = {v['delta_pct_of_off']:+.1f}%  "
              f"5% line {v['line_ms']:.2f} ms -> {v['x_over_line']:.1f}x over")


if __name__ == "__main__":
    main()
