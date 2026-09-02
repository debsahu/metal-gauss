#!/usr/bin/env python3
"""Noise floors for the Tier 0+1 measurement protocol (plan Task 10, phase 2).

Reads B0a/B0b/B0c's report + splatstats + LPIPS JSONs and writes `floors.json`.

    B0a vs B0b  (SAME seed)      -> REPEAT floor
    B0a vs B0c  (different seed) -> SEED floor

An arm "moves" a metric only if it moves it by more than the REPEAT floor, and is "robust"
only if by more than the SEED floor. The repeat floor is the correct yardstick for the
paired same-seed A/Bs this protocol runs; grading a paired comparison against an unpaired
spread understates what counts as real.

This script must run -- and floors.json must exist -- BEFORE any treatment arm is graded.
Checkpoint D verifies that by file mtime.

It also refuses to proceed unless every arm was scored against a TSDF-ONLY reference cloud.
Scoring thin-axis against the seed that was trained on was an 11.6 deg error once, larger
than every recipe gain in CLAUDE.md's table, so it is checked rather than trusted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ARMS = ("B0a", "B0b", "B0c")


def load(out: Path, arm: str) -> dict:
    rep = json.loads((out / f"{arm}.json").read_text())
    st = json.loads((out / f"{arm}.stats.json").read_text())
    seed_cloud = str(st.get("seed_cloud") or "")
    if not seed_cloud.endswith("points3D.tsdf.txt"):
        sys.exit(f"{arm}: splatstats seed_cloud is {seed_cloud!r}, not the TSDF-only cloud. "
                 f"Thin-axis scored against the trained seed is not comparable to anything.")
    vals = {f"stats.{k}": v for k, v in st["metrics"].items() if isinstance(v, (int, float))}
    for k in ("psnr_masked", "psnr", "coverage", "ms_per_step", "n_splats"):
        if isinstance(rep["metrics"].get(k), (int, float)):
            vals[f"run.{k}"] = rep["metrics"][k]
    lp = out / f"{arm}.dump" / "lpips.json"
    if lp.exists():
        vals["run.lpips"] = json.loads(lp.read_text())["mean"]
    return {"seed": rep["resolved"]["seed"], "git": rep["env"]["git"],
            "seed_cloud": seed_cloud, "values": vals}


def main() -> None:
    out = Path(sys.argv[1])
    arms = {a: load(out, a) for a in ARMS}
    if arms["B0a"]["seed"] != arms["B0b"]["seed"]:
        sys.exit("B0a and B0b must share a seed; they are the REPEAT floor")
    if arms["B0c"]["seed"] == arms["B0a"]["seed"]:
        sys.exit("B0c must use a DIFFERENT seed; it is the SEED floor")
    keys = sorted(set(arms["B0a"]["values"]) & set(arms["B0b"]["values"]) & set(arms["B0c"]["values"]))
    floors = {}
    for k in keys:
        a, b, c = (arms[x]["values"][k] for x in ARMS)
        floors[k] = {"B0a": a, "B0b": b, "B0c": c,
                     "repeat_floor": abs(a - b), "seed_floor": abs(a - c)}
    doc = {"schema": 1,
           "note": "repeat_floor = |B0a - B0b| (same seed); seed_floor = |B0a - B0c|. "
                   "An arm moves a metric only if it moves it by more than repeat_floor.",
           "arms": {a: {k: v for k, v in arms[a].items() if k != "values"} for a in ARMS},
           "floors": floors}
    (out / "floors.json").write_text(json.dumps(doc, indent=1))
    print(f"wrote {out/'floors.json'} over {len(keys)} metrics")
    for k in ("run.psnr_masked", "run.lpips", "stats.thin_axis_angle_median_deg",
              "stats.on_seed_frac_1cm", "stats.opacity_p50"):
        if k in floors:
            f = floors[k]
            print(f"  {k:<38} repeat {f['repeat_floor']:.5g}   seed {f['seed_floor']:.5g}")


if __name__ == "__main__":
    main()
