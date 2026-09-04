#!/usr/bin/env python3
"""Task 19 steps 4-6: the plane-aux measurement protocol, floors first, one scene per run.

    scripts/plane_aux_arms.py --scene NAME --colmap DIR --images DIR --seed-cloud PATH
                              [--depth-dir DIR] [--normal-dir DIR] [--init-ply PATH]
                              [--max-resolution N] [--dn WEIGHT] --out DIR

PHASE ORDER IS LOAD-BEARING AND ENFORCED HERE, not by operator discipline. Floors are run,
scored, and WRITTEN before the treatment arm is scored -- the same rule
`run_tier1_arms.sh` enforces, for the same reason: nobody may choose a floor after seeing
a treatment number.

## The base arm's own floor, at n=3, because Tier 1's floors do not transfer

research/metal-gauss.md section 8.2's floor table is explicitly BASELINE-arm ("a recipe arm
may vary differently -- nothing here measures the spread of an arm that carries geometry
weights") and section 11.3b showed on-seed@1cm's within-implementation spread is 0.80%
against a 0.5% bar that had been assumed. So the base arm here -- R1p + `--depth-source
center` -- is run three times: twice at `--seed 42` and once at 43, per the plan's section
3. `scripts/tier1_floors.py` is NOT reused: it computes `repeat_floor = |A - B|` from a
PAIR, and section 8.2 is the record of an n=2 floor coming out 25-45x too small and taking
a day's conclusions with it.

## Grading, pre-registered here BEFORE any arm runs

  baseline  = MEAN of the three floor arms          (more robust than any single run)
  floor(m)  = max - min of the three floor arms     (the n=3 spread)
  delta(m)  = treatment - baseline
  paired    = treatment - F0                        (both --seed 42; REPORTED, not graded)

A metric MOVES only if |delta| > floor. Direction of "worse" per metric:

  on_seed_frac_1cm      worse = DOWN     must RISE by more than floor to pass
  thin_axis_angle_p50   worse = UP       must not rise by more than floor
  aspect_p50            worse = DOWN     must not fall by more than floor
  needle_frac           worse = UP       must not rise by more than floor
  psnr_masked           two-sided        must be WITHIN floor

`aspect_p50` and `needle_frac` are the anti-collapse columns and the reason this list is
not three metrics long: on the VOID row (section 8.1) thin-axis, opacity and dark fraction
all reported a destroyed reconstruction as HEALTHIER than baseline, and only in-plane
aspect (0.2957 -> 0.0659) and needle fraction (16.6% -> 56.8%) saw it.

Verdicts are computed per scene; the KEEP / OPT-IN / DROP decision needs both scenes and is
made in the report, not here.

Every guard in `plane_aux_throughput.py` applies and is imported from it rather than
re-typed: hand-rolled exclusivity checks, missing watchdogs (macOS has no `timeout`(1)),
empty-log liveness, the stale FileBaton lock, and `uv run --frozen` reverting fix_openmp.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("MG_ROOT") or Path(__file__).resolve().parents[1])
if not (ROOT / "metal_gauss" / "train.py").exists():
    raise SystemExit(f"MG_ROOT={ROOT} is not a metal-gauss checkout. Set MG_ROOT when "
                     f"running from a snapshot.")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from bench.runner import require_gpu_exclusive          # noqa: E402
from plane_aux_throughput import run_arm, summarise      # noqa: E402

SPLATSTATS = ROOT.parent.parent / "analyze" / "splatstats"

# metric -> +1 if HIGHER is better, -1 if LOWER is better, 0 if two-sided (want no change)
DIRECTION = {
    "stats.on_seed_frac_1cm": +1,
    "stats.on_seed_frac_2cm": +1,
    "stats.thin_axis_angle_p50": -1,
    "run.aspect_p50": +1,
    "run.needle_frac": -1,
    "run.psnr_masked": 0,
}
# The four geometry columns the keep/drop rule turns on. PSNR is a "within the floor"
# condition, not a geometry column, and is handled separately.
GEOMETRY_GATE = ("stats.on_seed_frac_1cm", "stats.thin_axis_angle_p50",
                 "run.aspect_p50", "run.needle_frac")


def peak_driver_gb(log: Path) -> float | None:
    vals = [float(m) for m in re.findall(r"\[mem\] driver ([0-9.]+) GB", log.read_text(errors="replace"))]
    return max(vals) if vals else None


def score(out: Path, tag: str, seed_cloud: str) -> None:
    """splatstats + LPIPS + backfill. Idempotent; each step skips if its artifact exists."""
    stats = out / f"{tag}.stats.json"
    if not stats.exists():
        r = subprocess.run(
            ["caffeinate", "-i", "uv", "run", "--frozen", "python", "scripts/splat_stats.py",
             str(out / f"{tag}.ply"), "--seed", seed_cloud, "--json", str(stats), "--quiet"],
            cwd=str(SPLATSTATS), capture_output=True, text=True)
        if not stats.exists():
            raise SystemExit(f"{tag}: splatstats wrote no JSON\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    lp = out / f"{tag}.dump" / "lpips.json"
    if not lp.exists():
        r = subprocess.run(["caffeinate", "-i", "uv", "run", "scripts/lpips_eval.py",
                            str(out / f"{tag}.dump")], cwd=str(ROOT),
                           capture_output=True, text=True)
        if not lp.exists():
            raise SystemExit(f"{tag}: lpips_eval wrote no JSON\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    subprocess.run([sys.executable, str(ROOT / "scripts/fix_openmp.py")],
                   capture_output=True, text=True)
    subprocess.run([str(ROOT / ".venv/bin/python"), "scripts/backfill_lpips.py",
                    str(out), tag], cwd=str(ROOT), capture_output=True, text=True)


def battery(out: Path, tag: str) -> dict:
    """Every column the plan requires, from the artifact that produced it -- never stdout."""
    rep = json.loads((out / f"{tag}.json").read_text())
    st = json.loads((out / f"{tag}.stats.json").read_text())
    ref = str(st.get("seed_cloud") or "")
    # Scoring thin-axis against the cloud the trainer seeded from was an 11.6 deg error
    # once -- larger than every recipe gain in CLAUDE.md's table. Checked, never trusted.
    if Path(ref).name in ("points3D.txt", "seed.ply"):
        raise SystemExit(f"{tag}: reference cloud is {ref!r}, which is (or may be) the seed "
                         f"the trainer initialised from. Use the TSDF-only cloud.")
    m, sh = rep["metrics"], rep["metrics"]["shape"]
    vals = {f"stats.{k}": v for k, v in st["metrics"].items() if isinstance(v, (int, float))}
    vals.update({
        "run.psnr_masked": m["psnr_masked"], "run.psnr": m["psnr"],
        "run.coverage": m["coverage"], "run.lpips": m.get("lpips"),
        "run.ms_per_step": m["ms_per_step"], "run.n_splats": m["n_splats"],
        "run.aspect_p50": sh["aspect_p50"], "run.needle_frac": sh["needle_frac"],
        "run.peak_driver_gb": peak_driver_gb(out / f"{tag}.log"),
    })
    return {"seed": rep["resolved"]["seed"], "git": rep["env"]["git"],
            "depth_source": rep["resolved"]["depth_source"], "seed_cloud": ref,
            "thin_axis_evaluated": st["metrics"].get("thin_axis_evaluated"),
            "values": {k: v for k, v in vals.items() if isinstance(v, (int, float))}}


def check_floors_match(prev: dict, dn: float, space: str) -> None:
    """Refuse a floors.json measured for a DIFFERENT configuration.

    Reusing floors is legitimate -- step 7 runs a second treatment against the same base --
    and it is also exactly how a floor from the wrong configuration gets applied without
    anyone noticing. research/metal-gauss.md section 8.2 is this project's record of that
    class of error: baseline-arm floors quoted for recipe arms, and an n=2 floor 25-45x too
    small. So the reuse is CHECKED, and the check reads the fields that define the
    configuration, not merely that the file exists.

    A missing key is treated as a mismatch, never as agreement: an older floors.json that
    predates a field cannot testify about it.
    """
    have_dn, have_space = prev.get("dn", "<absent>"), prev.get("depth_loss_space", "<absent>")
    if have_dn != dn or have_space != space:
        raise SystemExit(
            f"floors.json was measured at dn={have_dn} space={have_space}; this run is "
            f"dn={dn} space={space}. A floor for another configuration is not this arm's "
            f"floor. Re-measure, or drop --skip-floors.")


def verdict_for(metric: str, delta: float, floor: float) -> str:
    """IMPROVED / WORSENED / WITHIN FLOOR for one metric.

    Pure, so the whole grade is reproducible from the committed artifacts by
    `--regrade`, and so the rule itself is unit-tested rather than only exercised by an
    8-hour run. `abs(delta) > floor` is STRICT: a delta exactly equal to the floor has not
    cleared it.
    """
    sign = DIRECTION[metric]
    if abs(delta) <= floor:
        return "WITHIN FLOOR"
    if sign == 0:
        return "MOVED"
    return "IMPROVED" if sign * delta > 0 else "WORSENED"


def grade(scene: str, dn: float, t: dict, fl: dict) -> dict:
    """The pre-registered keep/drop rule, applied. See the module docstring."""
    rows, verdict = {}, {}
    for k in sorted(set(t["values"]) & set(fl)):
        base, floor = fl[k]["mean"], fl[k]["spread_n3"]
        d = t["values"][k] - base
        row = {"treatment": t["values"][k], "baseline_mean": base, "delta": d,
               "paired_vs_F0": t["values"][k] - fl[k]["F0"],
               "floor_spread_n3": floor, "moves": abs(d) > floor}
        if k in DIRECTION:
            row["direction"] = {1: "higher is better", -1: "lower is better",
                                0: "two-sided"}[DIRECTION[k]]
            row["verdict"] = verdict[k] = verdict_for(k, d, floor)
        rows[k] = row
    gate = {k: verdict.get(k) for k in GEOMETRY_GATE}
    psnr = verdict.get("run.psnr_masked")
    missing = [k for k, v in gate.items() if v is None] + ([] if psnr else ["run.psnr_masked"])
    if missing:
        # A gate column that is absent must never read as a pass. This is the shape of
        # failure CLAUDE.md calls the one this project keeps repeating: a check that reads
        # a condition something OTHER than the thing being checked could satisfy.
        raise SystemExit(f"{scene}: gate columns missing from the battery: {missing}")
    return {"schema": 1, "scene": scene, "dn": dn,
            "treatment": {k: v for k, v in t.items() if k != "values"},
            "scene_pass": (gate["stats.on_seed_frac_1cm"] == "IMPROVED"
                           and gate["stats.thin_axis_angle_p50"] != "WORSENED"
                           and gate["run.aspect_p50"] != "WORSENED"
                           and gate["run.needle_frac"] != "WORSENED"
                           and psnr == "WITHIN FLOOR"),
            "scene_drop": any(v == "WORSENED" for v in gate.values()),
            "falsifier_triggered_on_this_scene":
                (verdict.get("stats.on_seed_frac_1cm") == "WITHIN FLOOR"
                 and verdict.get("stats.thin_axis_angle_p50") == "WITHIN FLOOR"),
            "geometry_gate": gate, "psnr_verdict": psnr, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--seed-cloud", required=True)
    ap.add_argument("--depth-dir"); ap.add_argument("--normal-dir")
    ap.add_argument("--init-ply")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-resolution", type=int, default=1920)
    ap.add_argument("--budget", type=int, default=500_000)
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dn", type=float, default=0.0,
                    help="--depth-normal-weight. 0.0 is step 5's R1p (the pre-registered "
                         "one-variable arm); 0.05 is step 6's R1, which the plan gates "
                         "behind Task 20's bound.")
    ap.add_argument("--depth-loss-space", default="disparity", choices=["disparity", "metric"])
    ap.add_argument("--watchdog", type=float, default=14_400.0)
    ap.add_argument("--treatment-tag", default="P0",
                    help="tag for the treatment arm. Step 7 (metric vs disparity) reuses "
                         "the SAME floors with a different treatment, so it needs a "
                         "different tag and must not overwrite P0.")
    ap.add_argument("--treatment-depth-source", default="plane-aux",
                    choices=["center", "plane-aux"])
    ap.add_argument("--floors-only", action="store_true",
                    help="stop after phase 2. Used when a later treatment will reuse "
                         "these floors.")
    ap.add_argument("--skip-floors", action="store_true",
                    help="reuse the floors.json already in --out. REFUSES unless that "
                         "file exists AND its recorded dn / depth_loss_space match this "
                         "invocation: a floor measured for another configuration is not "
                         "this arm's floor, which is the whole reason Tier 3 measures its "
                         "own (section 8.2).")
    ap.add_argument("--regrade", action="store_true",
                    help="recompute grade.json from the artifacts already on disk. No "
                         "GPU, no training: the verdict is a pure function of the "
                         "reports and stats JSONs, and this proves it.")
    a = ap.parse_args()

    if a.regrade:
        out = Path(a.out)
        fl = json.loads((out / "floors.json").read_text())["floors"]
        doc = grade(a.scene, a.dn, battery(out, a.treatment_tag), fl)
        (out / "grade.json").write_text(json.dumps(doc, indent=2))
        print(json.dumps({k: doc[k] for k in
                          ("scene", "scene_pass", "scene_drop",
                           "falsifier_triggered_on_this_scene", "geometry_gate",
                           "psnr_verdict")}, indent=2))
        return

    require_gpu_exclusive()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    common = ["--colmap", a.colmap, "--images", a.images,
              "--max-resolution", str(a.max_resolution), "--steps", str(a.steps),
              "--budget", str(a.budget), "--num-downscales", "0",
              "--eval-split-every", "8", "--eval-every", "2500",
              "--depth-loss-space", a.depth_loss_space,
              "--flatten-loss-weight", "1.0", "--depth-loss-weight", "1.0",
              "--normal-loss-weight", "0.2", "--depth-normal-weight", str(a.dn)]
    for flag, val in (("--depth-dir", a.depth_dir), ("--normal-dir", a.normal_dir),
                      ("--init-ply", a.init_ply)):
        if val:
            common += [flag, val]

    # (tag, depth_source, seed). Two floors share the seed (the repeat pair); the third
    # uses seed+1 (the seed floor). The treatment shares the first floor's seed, so the
    # paired comparison exists even though grading is against the n=3 spread.
    floors = [("F0", "center", a.seed), ("F1", "center", a.seed), ("F2", "center", a.seed + 1)]
    treatment = (a.treatment_tag, a.treatment_depth_source, a.seed)

    if a.skip_floors:
        prev = json.loads((out / "floors.json").read_text())
        check_floors_match(prev, a.dn, a.depth_loss_space)
        fl = prev["floors"]
        print(f"=== PHASES 1-2 SKIPPED: reusing floors.json ({len(fl)} metrics) ===",
              flush=True)
    else:
        run_floor_phases(a, out, common, floors)
        fl = json.loads((out / "floors.json").read_text())["floors"]
    if a.floors_only:
        print("floors only: stopping before the treatment arm", flush=True)
        return

    print("=== PHASE 3: treatment arm ===", flush=True)
    tag, src, seed = treatment
    run_arm(tag, out, common + ["--seed", str(seed),
                                "--export", str(out / f"{tag}.ply"),
                                "--eval-dump", str(out / f"{tag}.dump")],
            ["--depth-source", src], a.watchdog)

    print("=== PHASE 4: score treatment and grade ===", flush=True)
    score(out, tag, a.seed_cloud)
    t = battery(out, tag)
    if t["depth_source"] != src:
        raise SystemExit(f"treatment arm reports depth_source={t['depth_source']!r}, asked "
                         f"for {src!r}: the flag did not survive resolve_depth_source.")
    doc = grade(a.scene, a.dn, t, fl)
    (out / f"grade_{tag}.json").write_text(json.dumps(doc, indent=2))
    (out / "grade.json").write_text(json.dumps(doc, indent=2))
    (out / "ALL_DONE").write_text("")
    print(json.dumps({k: doc[k] for k in
                      ("scene", "scene_pass", "scene_drop",
                       "falsifier_triggered_on_this_scene", "geometry_gate",
                       "psnr_verdict")}, indent=2))


def run_floor_phases(a, out: Path, common: list[str], floors) -> None:
    print(f"=== PHASE 1: floor arms ({a.scene}) ===", flush=True)
    for tag, src, seed in floors:
        run_arm(tag, out, common + ["--seed", str(seed),
                                    "--export", str(out / f"{tag}.ply"),
                                    "--eval-dump", str(out / f"{tag}.dump")],
                ["--depth-source", src], a.watchdog)

    print("=== PHASE 2: score floors, then WRITE floors.json ===", flush=True)
    for tag, _, _ in floors:
        score(out, tag, a.seed_cloud)
    arms = {tag: battery(out, tag) for tag, _, _ in floors}
    refs = {arms[t]["seed_cloud"] for t in arms}
    if len(refs) > 1:
        raise SystemExit(f"floor arms disagree about the reference cloud: {refs}")
    if arms["F0"]["seed"] != arms["F1"]["seed"]:
        raise SystemExit("F0 and F1 must share a seed: they are the repeat pair")
    if arms["F2"]["seed"] == arms["F0"]["seed"]:
        raise SystemExit("F2 must use a different seed: it is the seed floor")
    keys = sorted(set.intersection(*(set(arms[t]["values"]) for t in arms)))
    fl = {}
    for k in keys:
        v = [arms[t]["values"][k] for t in ("F0", "F1", "F2")]
        fl[k] = {"F0": v[0], "F1": v[1], "F2": v[2],
                 "mean": statistics.mean(v), "spread_n3": max(v) - min(v),
                 "repeat_pair_abs_diff": abs(v[0] - v[1])}
    (out / "floors.json").write_text(json.dumps(
        {"schema": 1, "scene": a.scene, "dn": a.dn,
         "depth_loss_space": a.depth_loss_space,
         "note": "floor = spread_n3 = max-min over the three base arms. repeat_pair is "
                 "|F0-F1| and is REPORTED ONLY -- an n=2 floor was 25-45x too small once "
                 "(research/metal-gauss.md section 8.2) and is not graded against.",
         "arms": {t: {k: v for k, v in arms[t].items() if k != "values"} for t in arms},
         "floors": fl}, indent=2))
    (out / "FLOORS_DONE").write_text("")
    print(f"  floors written: {len(fl)} metrics", flush=True)


if __name__ == "__main__":
    main()
