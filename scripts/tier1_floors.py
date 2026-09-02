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

def load(out: Path, arm: str) -> dict:
    rep = json.loads((out / f"{arm}.json").read_text())
    seed_cloud = ""
    vals: dict = {}
    stats_path = out / f"{arm}.stats.json"
    if stats_path.exists():
        st = json.loads(stats_path.read_text())
        seed_cloud = str(st.get("seed_cloud") or "")
        # The reference must never be the cloud the trainer seeded from. Scoring thin-axis
        # against the trained seed was an 11.6 deg error once -- larger than every recipe
        # gain in CLAUDE.md's table -- so it is checked, not trusted.
        if Path(seed_cloud).name == "points3D.txt":
            sys.exit(f"{arm}: reference cloud is {seed_cloud!r}, which is the seed the "
                     f"trainer initialised from. Thin-axis scored against it is meaningless.")
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
    names = sys.argv[2:] or ["B0a", "B0b", "B0c"]
    if len(names) < 2:
        sys.exit("need at least two floor arms: the same-seed pair IS the repeat floor")
    arms = {a: load(out, a) for a in names}
    ra, rb = names[0], names[1]
    if arms[ra]["seed"] != arms[rb]["seed"]:
        sys.exit(f"{ra} and {rb} must share a seed; they are the REPEAT floor")
    sc = {a["seed_cloud"] for a in arms.values()}
    if len(sc) > 1:
        sys.exit(f"floor arms disagree about the reference cloud: {sc}")
    third = names[2] if len(names) > 2 else None
    if third and arms[third]["seed"] == arms[ra]["seed"]:
        sys.exit(f"{third} must use a DIFFERENT seed; it is the SEED floor")
    keys = sorted(set.intersection(*(set(arms[a]["values"]) for a in names)))
    floors = {}
    for k in keys:
        a, b = arms[ra]["values"][k], arms[rb]["values"][k]
        row = {ra: a, rb: b, "repeat_floor": abs(a - b)}
        if third:
            c = arms[third]["values"][k]
            row[third] = c
            row["seed_floor"] = abs(a - c)
        floors[k] = row
    doc = {"schema": 1,
           "note": "repeat_floor = |B0a - B0b| (same seed); seed_floor = |B0a - B0c|. "
                   "An arm moves a metric only if it moves it by more than repeat_floor.",
           "arms": {a: {k: v for k, v in arms[a].items() if k != "values"} for a in names},
           "floors": floors}
    (out / "floors.json").write_text(json.dumps(doc, indent=1))
    print(f"wrote {out/'floors.json'} over {len(keys)} metrics")
    for k in ("run.psnr_masked", "run.lpips", "stats.thin_axis_angle_median_deg",
              "stats.on_seed_frac_1cm", "stats.opacity_p50"):
        if k in floors:
            f = floors[k]
            sf = f"   seed {f['seed_floor']:.5g}" if "seed_floor" in f else ""
            print(f"  {k:<38} repeat {f['repeat_floor']:.5g}{sf}")


if __name__ == "__main__":
    main()
