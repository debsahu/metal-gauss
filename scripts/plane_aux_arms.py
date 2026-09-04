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
import math
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
    "run.lpips": -1,
    "run.psnr_masked": 0,
}
# The four geometry columns the OLD (magnitude-blind) keep/drop rule turned on. Retained
# because `grade` still reports each column's IMPROVED / WORSENED / WITHIN FLOOR verdict
# and the tests assert every gate column has a declared direction. The DECISION no longer
# reads all four: aspect and needles moved to Band 1, where magnitude decides.
GEOMETRY_GATE = ("stats.on_seed_frac_1cm", "stats.thin_axis_angle_p50",
                 "run.aspect_p50", "run.needle_frac")

# ---------------------------------------------------------------------------------------
# THE TIER 3 KEEP/DROP RULE, adopted 2026-09-04, replacing "WORSENED anywhere = DROP".
#
# The old rule was magnitude-blind. research/metal-gauss.md section 12.4 records what that
# cost: P-GEOM's plane-aux arm moved in-plane aspect -2.5% and needles +0.6 pp -- 4.6x and
# 4.5x floors that are 0.0017 and 0.0013 wide -- and DROPped, while the VOID row of section
# 8.1 (aspect -78%, needles +40 pp, on-seed HALVED) produced the same one-word verdict.
# The section's own words: "Whether a magnitude-blind gate is the right instrument is a real
# question -- and it must be settled BEFORE the next arm, never after seeing this one."
#
#   Band 1  COLLAPSE     hard DROP, any one column, per-arm AND cumulative
#   Band 2  GEOMETRY     on-seed@1cm must RISE; thin-axis must not worsen
#   Band 3  PHOTOMETRIC  hard DROP on a >0.25 dB PSNR loss, or crossing the 24 dB gate
#
# A beyond-floor worsening that is below Band 1 is DRIFT: reported with sign and x floor,
# and it does NOT drop. It does block KEEP AS DEFAULT.
#
# EVERY BAND 1 THRESHOLD IS `sqrt(healthy x collapse)` in the column's natural space --
# geometric mean of the largest ADOPTED move (R1 or R1p vs B0a, whichever moved further in
# the worsening direction) and the smallest RECORDED collapse (the VOID row). It is a
# fence between two measured populations, not a round number.
# `tests/test_plane_aux_tier3_rule.py` re-derives all four from the section 8.1 table, so a
# later session cannot quietly retune one to make an arm pass.
COLLAPSE = {
    # column                     space  worse   threshold   derivation (section 8.1)
    "run.needle_frac":      {"space": "abs", "worse": +1, "threshold": 0.108},
    "run.aspect_p50":       {"space": "log", "worse": -1, "threshold": 0.346},
    "stats.on_seed_frac_1cm": {"space": "log", "worse": -1, "threshold": 0.185},
    "run.lpips":            {"space": "abs", "worse": +1, "threshold": 0.017},
}
# Band 2. Note what is NOT here: aspect and needles. They are Band 1's business now.
BAND2_GATE = ("stats.on_seed_frac_1cm", "stats.thin_axis_angle_p50")

# Band 3. 0.25 dB sits above the trainer's own cross-machine same-seed spread (0.115-0.220
# dB, section 8.2): a loss smaller than the delivery pipeline's own reproduction spread
# cannot be a product-visible regression on its own. 24.0 dB is CLAUDE.md's Stage 4 gate.
PSNR_DROP_DB = 0.25
STAGE4_PSNR_DB = 24.0

# Drift is scored over exactly the columns the rule grades, and only in the worsening
# direction. If it counted improvements, KEEP AS DEFAULT would be unreachable: Band 2
# REQUIRES on-seed to improve beyond its floor, which would itself be a drift column.
DRIFT_SCOPE = tuple(dict.fromkeys(tuple(COLLAPSE) + BAND2_GATE + ("run.psnr_masked",)))

# A hard needle is a splat whose minor in-plane half-axis is smaller than the rim
# displacement its OWN quantised orientation produces in the delivery format, so its
# orientation is undeliverable however well it was trained.
#
# Verified in the installed splat-transform (`@playcanvas/splat-transform`,
# `dist/index.mjs`, the SOG writer that emits the `252 + maxComp` tag): the quaternion is
# normalised, scaled by `+-sqrt(2)`, and its smallest three components stored as
# `255 * (q * 0.5 + 0.5)` in uint8. One uint8 step is therefore `sqrt(2)/255 = 0.0055459`
# in true component units; worst-case round-to-nearest error is `step/2` per component over
# three components, i.e. a quaternion perturbation of norm `(step/2)*sqrt(3)`; and a
# perturbation of norm e is a rotation of `2e`. That is 0.0096058 rad. 0.01 is the next
# round number at or above it.
#
# REPORTED ONLY. It is not a Band 1 column and not a gate column: it comes from a ply, not
# from the trainer's report, so an archived arm may or may not have it, and a gate that is
# sometimes absent is the failure shape this project keeps repeating.
HARD_NEEDLE_ASPECT = 0.01


def peak_driver_gb(log: Path) -> float | None:
    # A re-grade runs against the COMMITTED artifacts, which are the reports only -- the
    # multi-megabyte training logs are not in git. An absent log is "not measured here",
    # never zero: the column then falls out of the battery on both sides and is not graded.
    if not log.exists():
        return None
    vals = [float(m) for m in re.findall(r"\[mem\] driver ([0-9.]+) GB", log.read_text(errors="replace"))]
    return max(vals) if vals else None


def report_path(out: Path, tag: str) -> Path:
    """`<tag>.json` as a run writes it, or `<tag>.report.json` as the repo commits it.

    The two are not two formats: the committed file is a trimmed SUBSET, and `resolved`,
    `env.git` and `metrics` are byte-identical between them (checked across all ten Task 19
    arms). Accepting both is what lets `--regrade` reproduce a verdict from the repository
    alone, with no scratch directory and no GPU -- which is the only form of reproducibility
    that survives the scratch directory being cleaned.
    """
    for name in (f"{tag}.json", f"{tag}.report.json"):
        if (out / name).exists():
            return out / name
    raise SystemExit(f"{tag}: no report at {out}/{tag}.json or {out}/{tag}.report.json")


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
    rep = json.loads(report_path(out, tag).read_text())
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
        # REPORTED, never gated. `Dlog aspect = Dlog smid - Dlog smax`, so aspect already
        # IS this differential and a collapse test on the halves would double-count it.
        # They are here because MAGNITUDE is what separates Task 19 from the VOID row --
        # same shape, 1/35 the size -- and the raw millimetres say that directly.
        "run.smid_p50_mm": sh.get("smid_p50_mm"),
        "run.smax_p50_mm": sh.get("smax_p50_mm"),
        "run.hard_needle_frac": (sh.get("hard_needle_frac")
                                 if sh.get("hard_needle_frac") is not None
                                 else hard_needle_from_sidecar(out, tag)),
        "run.peak_driver_gb": peak_driver_gb(out / f"{tag}.log"),
    })
    return {"seed": rep["resolved"]["seed"], "git": rep["env"]["git"],
            "depth_source": rep["resolved"]["depth_source"], "seed_cloud": ref,
            "resolved": rep["resolved"],
            "thin_axis_evaluated": st["metrics"].get("thin_axis_evaluated"),
            "values": {k: v for k, v in vals.items() if isinstance(v, (int, float))}}


def hard_needle_from_sidecar(out: Path, tag: str) -> float | None:
    """`frac(aspect < HARD_NEEDLE_ASPECT)`, from `<tag>.shape.json` if `scripts/ply_shape.py`
    has been run over that arm's ply.

    It cannot come from the report for arms trained before the column existed, and it is
    NOT a gate column, so absence is legitimate and silent -- the one case where a missing
    column is allowed to be silent, because nothing decides on it.

    The sidecar is REFUSED unless it names the ply it read and reproduces that arm's
    `aspect_p50` and `needle_frac`: a shape file computed from some other ply would put a
    number in the battery that describes a different reconstruction.
    """
    f = out / f"{tag}.shape.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    if d.get("verified_against_report") is not True:
        raise SystemExit(f"{f} does not record a passing cross-check against {tag}'s "
                         f"report. Re-run scripts/ply_shape.py; an unverified shape "
                         f"sidecar may describe a different ply.")
    return d["hard_needle_frac"]


def floors_from_reports(out: Path, tags=("F0", "F1", "F2")) -> dict:
    """Rebuild the floor table from the floor arms' artifacts, for columns floors.json
    predates (smid/smax, hard needles, LPIPS on an older run).

    NOT a substitute for floors.json, which is the phase-ordered record written BEFORE any
    treatment was scored. It is checked against it: see `merge_extended_floors`.
    """
    arms = {t: battery(out, t) for t in tags}
    keys = sorted(set.intersection(*(set(arms[t]["values"]) for t in tags)))
    fl = {}
    for k in keys:
        v = [arms[t]["values"][k] for t in tags]
        fl[k] = {"F0": v[0], "F1": v[1], "F2": v[2],
                 "mean": statistics.mean(v), "spread_n3": max(v) - min(v),
                 "repeat_pair_abs_diff": abs(v[0] - v[1])}
    return fl


def merge_extended_floors(committed: dict, rebuilt: dict) -> dict:
    """Committed floors, plus the columns only the rebuild has -- and a hard refusal if the
    two DISAGREE anywhere they overlap.

    The overlap check is the point, not the merge. `floors.json` was written under the
    phase order that makes the whole protocol trustworthy (floors scored and frozen before
    any treatment number existed); recomputing it from the same reports must reproduce it
    exactly, and if it does not, something about the artifacts has changed underneath and
    no verdict computed from them means anything. Adding a column must never be a way to
    quietly re-measure a floor.
    """
    for k, v in committed.items():
        if k not in rebuilt:
            continue
        for field in ("F0", "F1", "F2", "mean", "spread_n3"):
            a, b = v.get(field), rebuilt[k].get(field)
            if a is None or b is None or abs(a - b) > 1e-12 * max(1.0, abs(a)):
                raise SystemExit(
                    f"floors.json and the floor arms' own reports disagree on "
                    f"{k}.{field}: {a!r} vs {b!r}. The committed floors are the frozen "
                    f"record; a rebuild that does not reproduce them means the artifacts "
                    f"moved, and nothing graded against them is meaningful.")
    return {**{k: v for k, v in rebuilt.items() if k not in committed}, **committed}


FLOOR_CONFIG_KEYS = ("depth_normal_weight", "depth_loss_space", "depth_source",
                     "flatten_loss_weight", "depth_loss_weight", "normal_loss_weight",
                     "budget", "steps", "max_resolution", "num_downscales")


def floor_configs(out: Path, tags=("F0", "F1", "F2")) -> list[dict]:
    """The floor arms' OWN resolved settings, from the reports that produced them.

    Not from a summary field in floors.json. `_run_report` records `vars(args)` AFTER
    resolution, so this is what each arm actually ran; a summary field is a claim about it.
    It also means the check works on a floors.json written before the summary field existed
    -- which is the case here, since the arm queue is executing a snapshot that predates it,
    and hand-editing a measurement artifact to satisfy a guard is not an option.
    """
    return [json.loads((out / f"{t}.json").read_text())["resolved"] for t in tags]


def check_floors_match(configs: list[dict], base_depth_source: str, dn: float,
                       space: str) -> None:
    """Refuse floors that were not measured on THIS arm's base configuration.

    Reusing floors is legitimate -- step 7 grades a second treatment against the same base
    -- and it is also exactly how a floor from the wrong configuration gets applied without
    anyone noticing. research/metal-gauss.md section 8.2 is this project's record of that
    class of error: baseline-arm floors quoted for recipe arms, and an n=2 floor 25-45x too
    small.

    Two things are checked, and the first is the one a summary field cannot do:

      1. the floor arms AGREE WITH EACH OTHER on every configuration key. Three runs that
         differ in a flag are not a repeat measurement of anything, and their spread is not
         a noise floor -- it is a treatment effect wearing one.
      2. they match the requested base. `depth_source` is included deliberately: if the
         winning depth source is `plane-aux`, floors measured on `center` are NOT its
         floors, and the honest answer is to re-measure rather than reuse.

    A missing key is a mismatch, never agreement.
    """
    if not configs:
        raise SystemExit("no floor arm reports to check")
    want = dict(zip(FLOOR_CONFIG_KEYS,
                    [dn, space, base_depth_source] + [None] * (len(FLOOR_CONFIG_KEYS) - 3)))
    ref = configs[0]
    for i, c in enumerate(configs[1:], start=1):
        diff = {k: (ref.get(k, "<absent>"), c.get(k, "<absent>")) for k in FLOOR_CONFIG_KEYS
                if ref.get(k, "<absent>") != c.get(k, "<absent>")}
        if diff:
            raise SystemExit(
                f"floor arms 0 and {i} differ in configuration: {diff}. Three runs that "
                f"differ in a flag are not a repeat measurement, and their spread is a "
                f"treatment effect, not a noise floor.")
    for k, v in want.items():
        if v is None:
            continue
        have = ref.get(k, "<absent>")
        if have != v:
            raise SystemExit(
                f"floors were measured with {k}={have}; this arm's base is {k}={v}. A "
                f"floor for another configuration is not this arm's floor. Re-measure, or "
                f"drop --skip-floors.")


PRIMARY_TAG = "P0"


def write_grade(out: Path, tag: str, doc: dict) -> None:
    """Write `grade_<tag>.json` always, and `grade.json` ONLY for the primary treatment.

    A DEFECT THIS FIXES, caught in flight rather than by review. `--regrade` wrote
    `grade.json` unconditionally, so re-grading the step-7 arm (`--treatment-tag M0`)
    OVERWROTE the step-5 scene verdict with the metric-space one -- and `--summary` reads
    `grade.json`. The cross-scene decision would then have been computed from the wrong
    arm's verdicts and reported as the plane-aux result, with nothing erroring: both files
    are well-formed grades of real arms, differing only in which arm they grade.

    The scene decision belongs to the PRE-REGISTERED arm (P0). Everything else gets its own
    file and cannot silently take its place.
    """
    (out / f"grade_{tag}.json").write_text(json.dumps(doc, indent=2))
    if tag == PRIMARY_TAG:
        (out / "grade.json").write_text(json.dumps(doc, indent=2))


def grade_filename(tag: str) -> str:
    """`grade.json` for the PRIMARY arm, `grade_<tag>.json` for every other.

    Mirrors `write_grade` exactly, and for the same reason: the scene verdict belongs to
    the pre-registered arm, and a second arm's cross-scene decision must be computable
    WITHOUT borrowing its filename. `--regrade --treatment-tag M0` overwriting `grade.json`
    is a defect this repo already shipped once (see `write_grade`); a summary that could
    only read `grade.json` would push someone toward re-creating it by hand.
    """
    return "grade.json" if tag == PRIMARY_TAG else f"grade_{tag}.json"


def collect_scenes(root: Path, scenes_csv: str, tag: str = PRIMARY_TAG) -> dict[str, dict]:
    """Load exactly the named scenes' grade.json, and refuse anything else.

    THIS IS NOT DEFENSIVENESS, IT IS A NEAR-MISS MADE STRUCTURAL. `--summary` originally
    globbed every subdirectory of --out that contained a grade.json. Two SYNTHETIC smoke
    directories -- fixtures written to test the grader itself, one of them a fabricated
    pass/regress pair -- were sitting in that same tree. Globbing would have fed invented
    numbers into the cross-scene decision, and because DROP is checked first and is not
    overridable, a fabricated regression would have produced a DROP that looked exactly
    like a measured one. Nothing would have errored.

    So the caller NAMES the scenes. A named scene that is missing is an error, and an
    UNNAMED grade.json found in the tree is ALSO an error rather than a silent inclusion --
    the second half is the one that matters, because it is the half a glob gets wrong.
    """
    want = [s for s in (x.strip() for x in scenes_csv.split(",")) if s]
    if not want:
        raise SystemExit("--summary requires --scenes (e.g. --scenes pgeom,arkit). A glob "
                         "over --out would count any stray grade.json, including the "
                         "grader's own synthetic test fixtures, as a measured scene.")
    fname = grade_filename(tag)
    found = {d.name for d in root.iterdir() if d.is_dir() and (d / fname).exists()}
    missing, extra = sorted(set(want) - found), sorted(found - set(want))
    if missing:
        raise SystemExit(f"--scenes named {missing} but no {fname} for them under {root}")
    if extra:
        raise SystemExit(f"unnamed {fname} under {root}: {extra}. Name them in --scenes "
                         f"or move them out; a stray one is not silently ignored.")
    per = {n: json.loads((root / n / fname).read_text()) for n in want}
    for n, g in per.items():
        # The directory name is not evidence about what was measured; the report is.
        if g.get("scene") != n:
            raise SystemExit(f"{n}/grade.json says scene={g.get('scene')!r}: the directory "
                             f"and the report disagree about what was measured.")
    return per


# ============================================================ the three bands


def collapse_delta(metric: str, value: float, reference: float) -> float:
    """How far `value` sits from `reference` TOWARD WORSE, in the column's natural space.

    Positive = worse, always, whichever direction the column runs. A sign error here
    inverts every Band 1 test -- an arm that HALVED on-seed would read as a large
    improvement and no collapse would ever fire -- so the sign is a test of its own.
    """
    spec = COLLAPSE[metric]
    if spec["space"] == "log":
        if value <= 0.0 or reference <= 0.0:
            raise SystemExit(f"{metric}: log-space column needs positive values, got "
                             f"value={value!r} reference={reference!r}")
        d = math.log(value) - math.log(reference)
    else:
        d = value - reference
    return spec["worse"] * d


def _collapse_side(values: dict, reference: dict, what: str) -> dict:
    row = {}
    for col in COLLAPSE:
        if col not in values:
            raise SystemExit(f"Band 1 column {col} is missing from the treatment battery. "
                             f"A collapse column that was never measured must never read "
                             f"as 'did not collapse'.")
        if col not in reference:
            raise SystemExit(f"Band 1 column {col} is missing from the {what}. An anchor "
                             f"or baseline that predates a column cannot testify about "
                             f"that column.")
        d = collapse_delta(col, values[col], reference[col])
        thr = COLLAPSE[col]["threshold"]
        row[col] = {"value": values[col], "reference": reference[col], "delta": d,
                    "threshold": thr, "x_threshold": d / thr,
                    "space": COLLAPSE[col]["space"], "fired": d > thr}
    return row


def band1(t_values: dict, base_values: dict, anchor_values: dict) -> dict:
    """Band 1 -- COLLAPSE. Hard DROP; any ONE column; per-arm AND cumulative.

    Per-arm is against this arm's own re-measured base. Cumulative is against the scene's
    FROZEN Tier 3 anchor, and it is the half that stops the rule ratcheting: four accepted
    8 pp needle drifts are a 32 pp collapse that no single arm ever fired on. Without it a
    magnitude rule is a licence to walk anywhere, in small steps.

    Comparison is STRICT: a delta exactly equal to the threshold has not fired.
    """
    per = _collapse_side(t_values, base_values, "baseline")
    cum = _collapse_side(t_values, anchor_values, "anchor")
    pf = [k for k, v in per.items() if v["fired"]]
    cf = [k for k, v in cum.items() if v["fired"]]
    return {"per_arm": per, "cumulative": cum, "per_arm_fired": pf,
            "cumulative_fired": cf, "fired": bool(pf or cf)}


def band2(verdicts: dict) -> str:
    """Band 2 -- GEOMETRY GATE, unchanged from the pre-registered rule except in SCOPE.

        PASS          on-seed@1cm IMPROVED beyond floor, thin-axis p50 not WORSENED
        FAIL          either column WORSENED beyond floor
        WITHIN FLOOR  neither worsened, but on-seed did not rise either

    Aspect and needles are deliberately NOT read here. They were two of the four columns of
    the old gate, and moving them to Band 1 -- where a 2.5% move and a 78% collapse get
    different answers -- is the entire change.
    """
    missing = [k for k in BAND2_GATE if verdicts.get(k) is None]
    if missing:
        raise SystemExit(f"Band 2 columns missing from the battery: {missing}. An absent "
                         f"gate column must never read as a pass.")
    on_seed, thin = (verdicts[k] for k in BAND2_GATE)
    if on_seed == "WORSENED" or thin == "WORSENED":
        return "FAIL"
    if on_seed == "IMPROVED":
        return "PASS"
    return "WITHIN FLOOR"


def band3(psnr_treatment: float, psnr_baseline: float) -> dict:
    """Band 3 -- PHOTOMETRIC. Hard DROP on a PSNR LOSS greater than 0.25 dB, or on falling
    below the 24 dB Stage 4 gate from at or above it.

    One-sided by construction: the rule says "falls by", and the old two-sided "must be
    WITHIN floor" condition is what made every Tier 3 arm unable to PASS whatever its
    geometry did. A gain is not a regression.

    Both comparisons are strict.
    """
    loss = psnr_baseline - psnr_treatment
    crossed = psnr_baseline >= STAGE4_PSNR_DB > psnr_treatment
    return {"baseline": psnr_baseline, "treatment": psnr_treatment, "loss_db": loss,
            "allowance_db": PSNR_DROP_DB, "exceeds_allowance": loss > PSNR_DROP_DB,
            "crossed_stage4_gate": crossed,
            "baseline_above_stage4": psnr_baseline >= STAGE4_PSNR_DB,
            "fired": bool(loss > PSNR_DROP_DB or crossed)}


def drift_columns(rows: dict, verdicts: dict, band1_detail: dict,
                  band2_verdict: str | None = None,
                  band3_fired: bool = False) -> list[dict]:
    """Beyond floor, below Band 1, and WORSE. Reported with sign and x floor; never a DROP.

    Two exclusions carry the definition:
      * IMPROVEMENTS are not drift. Band 2 REQUIRES on-seed to improve beyond its floor, so
        counting any beyond-floor move would make KEEP AS DEFAULT unreachable by
        construction -- a rule with an unreachable branch is a broken rule.
      * A column that FIRED Band 1 is a COLLAPSE, not a drift. Reporting it as drift would
        make a hard DROP read as adoptable-with-caveats.
    """
    fired = set(band1_detail["per_arm_fired"]) | set(band1_detail["cumulative_fired"])
    out = []
    for k in DRIFT_SCOPE:
        if k in fired or k not in rows or k not in verdicts:
            continue
        d = rows[k]["delta"]
        if DIRECTION.get(k, 0) == 0:
            # two-sided column (PSNR): only a FALL is a worsening
            worse = verdicts[k] == "MOVED" and d < 0
        else:
            worse = verdicts[k] == "WORSENED"
        if not worse:
            continue
        fl = rows[k]["floor_spread_n3"]
        # A Band 2 column that WORSENED is why the scene failed, not a harmless drift. It
        # still satisfies the literal definition ("beyond floor, below Band 1"), so it is
        # reported rather than hidden -- but flagged, because a list whose entries mean
        # "adoptable with caveats" and "this is the DROP" at the same time is precisely the
        # shape of check CLAUDE.md warns about, aimed at a human reader.
        caused_fail = bool(band2_verdict == "FAIL" and k in BAND2_GATE
                           and verdicts.get(k) == "WORSENED")
        # Same reasoning for Band 3: on the VOID row masked PSNR falls 1.03 dB, which IS
        # the Band 3 firing, and listing it unqualified as "drift, does not DROP" would
        # describe the drop as a caveat.
        caused_b3 = bool(band3_fired and k == "run.psnr_masked")
        out.append({"metric": k, "delta": d, "floor_spread_n3": fl,
                    "x_floor": abs(d) / fl if fl else float("inf"), "sign": "worse",
                    "caused_band2_fail": caused_fail, "caused_band3_fire": caused_b3})
    return out


ANCHOR_CONFIG_KEYS = ("budget", "steps", "max_resolution", "num_downscales")


def load_anchor(path: Path, scene: str) -> dict:
    """The scene's frozen Tier 3 anchor entry, or an error.

    A missing scene is an ERROR and never an empty anchor: an empty anchor makes every
    cumulative check vacuous while still writing a well-formed grade, which is precisely
    the shape of failure the other guards in this file exist to stop.
    """
    doc = json.loads(path.read_text())
    entry = doc.get("scenes", {}).get(scene)
    if not entry:
        raise SystemExit(f"no frozen Tier 3 anchor for scene {scene!r} in {path}. The "
                         f"cumulative half of Band 1 cannot be evaluated without one, and "
                         f"an absent anchor must not silently become a vacuous check. "
                         f"Anchors present: {sorted(doc.get('scenes', {}))}")
    return entry


def check_anchor_applies(scene: str, anchor_entry: dict, resolved: dict) -> None:
    """The anchor is re-measured only when scene, budget or resolution changes -- so a run
    that changed one of those must not be graded against the old anchor.

    `steps` and `num_downscales` are checked too: both change what a 30k arm's geometry
    columns settle at, and neither is named in the sentence above, which is exactly why
    they are the ones that would slip through.

    `depth_source` is deliberately NOT checked. The anchor is a frozen SCENE baseline; if a
    treatment is ever adopted as the default base, the new floors move with it and the
    anchor must not, or the ratchet the cumulative check exists to catch becomes invisible.
    """
    want = anchor_entry.get("config") or {}
    for k in ANCHOR_CONFIG_KEYS:
        a, b = want.get(k, "<absent>"), resolved.get(k, "<absent>")
        if a != b:
            raise SystemExit(
                f"{scene}: the frozen anchor was measured at {k}={a!r} and this arm ran at "
                f"{k}={b!r}. An anchor for another configuration is not this scene's "
                f"anchor -- re-measure it (and say so in tier3_anchor.json) rather than "
                f"grading a ratchet against a fiction.")


def combined_verdict(per_scene: dict[str, dict]) -> dict:
    """The cross-scene outcome class, from the three-band rule.

        DROP                        -- Band 1 fires anywhere, or Band 2 FAILs, or Band 3
                                       fires. Checked FIRST and not overridable.
        KEEP AS DEFAULT             -- Band 2 PASSes on every scene with NO drift column
                                       on any of them.
        OPT-IN, DEFAULT-CANDIDATE   -- Band 2 PASSes on every scene, drift present.
                                       Promotable to default only by a blind visual A/B --
                                       NOT by this grader, which has no view of the render.
        OPT-IN                      -- PASSes on one scene, WITHIN FLOOR on the other.
        NOT ADOPTED                 -- nothing passed and nothing regressed.

    DROP IS CHECKED FIRST AND IS NOT OVERRIDABLE, and the drop set is recomputed here from
    the bands rather than read from each scene's `scene_drop`. A pass on one scene and a
    collapse on another is not an opt-in; an implementation that reached an opt-in branch
    first would turn a collapse into a recommendation.

    THE FALSIFIER IS REPORTED SEPARATELY AND CAN BE INCOMPLETE. As written it requires both
    on-seed@1cm and thin-axis to stay inside the floor on both scenes AT BOTH dn SETTINGS.
    Step 6 (dn = 0.05) is gated behind Task 20, so `falsifier_complete` is False whenever
    only one dn setting has been measured. A partial falsifier is not a falsification.
    """
    scenes = sorted(per_scene)
    def _drops(s):
        g = per_scene[s]
        return bool(g.get("band1_fired") or g.get("band2") == "FAIL"
                    or g.get("band3_fired"))
    drops = [s for s in scenes if _drops(s)]
    passes = [s for s in scenes if per_scene[s].get("band2") == "PASS" and s not in drops]
    within = [s for s in scenes if per_scene[s].get("band2") == "WITHIN FLOOR"
              and s not in drops]
    drifting = {s: per_scene[s].get("drift") or [] for s in scenes}
    any_drift = any(drifting[s] for s in scenes)
    dns = sorted({per_scene[s]["dn"] for s in scenes})
    fals = [s for s in scenes if per_scene[s]["falsifier_triggered_on_this_scene"]]

    if drops:
        decision = "DROP"
    elif len(passes) == len(scenes):
        decision = "KEEP AS DEFAULT" if not any_drift else "OPT-IN, DEFAULT-CANDIDATE"
    elif passes and len(passes) + len(within) == len(scenes):
        decision = "OPT-IN"
    else:
        decision = "NOT ADOPTED (no scene passed, none regressed)"

    return {"schema": 2, "rule": "tier3-three-band-2026-09-04",
            "scenes": scenes, "decision": decision,
            "promotion_requires":
                ("a blind visual A/B on a rendered view; this grader cannot promote a "
                 "default-candidate, because nothing it measures looks at the render"
                 if decision == "OPT-IN, DEFAULT-CANDIDATE" else None),
            "passed_on": passes, "within_floor_on": within, "regressed_on": drops,
            "drift_on": {s: [d["metric"] for d in drifting[s]] for s in scenes
                         if drifting[s]},
            "dn_settings_measured": dns,
            "falsifier_at_measured_dn": len(fals) == len(scenes),
            "falsifier_scenes": fals,
            "falsifier_complete": len(dns) >= 2,
            "per_scene": {s: {k: per_scene[s].get(k) for k in
                              ("band1_fired", "band2", "band3_fired", "scene_pass",
                               "scene_drop", "geometry_gate", "psnr_verdict", "dn")}
                          | {"drift": [d["metric"] for d in drifting[s]]}
                          for s in scenes}}


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


def grade(scene: str, dn: float, t: dict, fl: dict, anchor_values: dict,
          anchor_entry: dict, resolved: dict) -> dict:
    """The THREE-BAND Tier 3 keep/drop rule, applied to one scene. See the COLLAPSE block.

    Pure, so the whole grade is reproducible from the committed artifacts by `--regrade`,
    and so the rule is unit-tested rather than exercised once by an eight-hour run.

    `anchor_values` and `anchor_entry` are passed separately on purpose: the first is the
    frozen per-column baseline the cumulative check reads, the second carries the
    provenance and configuration the anchor is only valid for. They are cross-checked
    against each other here, so passing a mismatched pair is an error rather than a silent
    grade against the wrong numbers.
    """
    check_anchor_applies(scene, anchor_entry, resolved)
    declared = anchor_entry.get("values")
    if declared is not None and declared != anchor_values:
        raise SystemExit(f"{scene}: anchor_values disagree with anchor_entry['values']. "
                         f"Two claims about the same frozen anchor must not differ.")

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

    base_vals = {k: fl[k]["mean"] for k in fl}
    b1 = band1(t["values"], base_vals, anchor_values)
    b2 = band2(verdict)
    b3 = band3(t["values"]["run.psnr_masked"], fl["run.psnr_masked"]["mean"])
    drift = drift_columns(rows, verdict, b1, b2, b3["fired"])
    drop = bool(b1["fired"] or b2 == "FAIL" or b3["fired"])
    return {"schema": 2, "rule": "tier3-three-band-2026-09-04", "scene": scene, "dn": dn,
            "treatment": {k: v for k, v in t.items() if k != "values"},
            "band1": b1, "band1_fired": b1["fired"],
            "band2": b2, "band3": b3, "band3_fired": b3["fired"],
            "drift": drift,
            "scene_drop": drop,
            "scene_pass": (b2 == "PASS" and not drop),
            "falsifier_triggered_on_this_scene":
                (verdict.get("stats.on_seed_frac_1cm") == "WITHIN FLOOR"
                 and verdict.get("stats.thin_axis_angle_p50") == "WITHIN FLOOR"),
            "geometry_gate": gate, "psnr_verdict": psnr,
            "anchor": {"values": anchor_values,
                       "config": anchor_entry.get("config"),
                       "source": anchor_entry.get("source")},
            "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="")
    ap.add_argument("--colmap", default=""); ap.add_argument("--images", default="")
    ap.add_argument("--seed-cloud", default="")
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
    ap.add_argument("--floors-depth-source", default="center",
                    choices=["center", "plane-aux"],
                    help="the depth source the reused floors were measured on. Must be "
                         "stated, and must match: floors measured on `center` are not the "
                         "floors of a `plane-aux` base.")
    ap.add_argument("--floors-depth-loss-space", default="",
                    choices=["", "disparity", "metric"],
                    help="the loss space the reused floors were measured in, when it "
                         "DIFFERS from this arm's. Step 7 compares metric against a "
                         "disparity BASE, so the base's floors are the disparity ones. "
                         "Defaults to --depth-loss-space, so the ordinary case must still "
                         "match and the divergence has to be stated on the command line.")
    ap.add_argument("--skip-floors", action="store_true",
                    help="reuse the floors.json already in --out. REFUSES unless that "
                         "file exists AND its recorded dn / depth_loss_space match this "
                         "invocation: a floor measured for another configuration is not "
                         "this arm's floor, which is the whole reason Tier 3 measures its "
                         "own (section 8.2).")
    ap.add_argument("--scenes", default="",
                    help="comma-separated scene directory names --summary must find, "
                         "e.g. 'pgeom,arkit'. REQUIRED for --summary: see collect_scenes.")
    ap.add_argument("--summary", action="store_true",
                    help="read every <scene>/grade.json under --out and emit the "
                         "cross-scene KEEP / OPT-IN / DROP decision. No GPU.")
    ap.add_argument("--anchor", default=str(ROOT / "bench/results/plane_aux/tier3_anchor.json"),
                    help="the FROZEN Tier 3 anchor Band 1's cumulative half reads. Not the "
                         "floors: floors are re-measured per arm and would let the rule "
                         "ratchet, which is the failure the anchor exists to stop.")
    ap.add_argument("--regrade", action="store_true",
                    help="recompute grade.json from the artifacts already on disk. No "
                         "GPU, no training: the verdict is a pure function of the "
                         "reports and stats JSONs, and this proves it.")
    a = ap.parse_args()

    if a.summary:
        # `--out` is the PARENT holding one subdirectory per scene. Reads only grade.json
        # files, so it needs no GPU and no reports.
        root = Path(a.out)
        if not root.is_dir():
            raise SystemExit(f"--out {root} is not a directory")
        per = collect_scenes(root, a.scenes, a.treatment_tag)
        v = combined_verdict(per)
        v["arm"] = a.treatment_tag
        name = ("combined_verdict.json" if a.treatment_tag == PRIMARY_TAG
                else f"combined_verdict_{a.treatment_tag}.json")
        (Path(a.out) / name).write_text(json.dumps(v, indent=2))
        print(json.dumps(v, indent=2))
        return

    if a.regrade:
        out = Path(a.out)
        if not a.scene:
            raise SystemExit("--regrade needs --scene: the frozen anchor is per scene, and "
                             "the directory name is not evidence about what was measured.")
        committed = json.loads((out / "floors.json").read_text())["floors"]
        fl = merge_extended_floors(committed, floors_from_reports(out))
        t = battery(out, a.treatment_tag)
        entry = load_anchor(Path(a.anchor), a.scene)
        doc = grade(a.scene, a.dn, t, fl, entry["values"], entry, t["resolved"])
        write_grade(out, a.treatment_tag, doc)
        print(json.dumps({k: doc[k] for k in
                          ("scene", "rule", "band1_fired", "band2", "band3_fired",
                           "scene_pass", "scene_drop",
                           "falsifier_triggered_on_this_scene", "geometry_gate",
                           "psnr_verdict")} | {"drift": [d["metric"] for d in doc["drift"]]},
                         indent=2))
        return

    for req in ("scene", "colmap", "images", "seed_cloud"):
        if not getattr(a, req):
            raise SystemExit(f"--{req.replace('_', '-')} is required to run arms "
                             f"(it is optional only for --summary / --regrade)")
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
        # The floors describe the BASE arm, and step 7's base is the disparity arm while
        # its treatment is the metric one -- so the space to check the floors against is
        # the BASE's, not the treatment's. It must be STATED (it defaults to the
        # treatment's, so the ordinary case still has to match) rather than inferred.
        check_floors_match(floor_configs(out), a.floors_depth_source, a.dn,
                           a.floors_depth_loss_space or a.depth_loss_space)
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
    entry = load_anchor(Path(a.anchor), a.scene)
    doc = grade(a.scene, a.dn, t, fl, entry["values"], entry, t["resolved"])
    write_grade(out, tag, doc)
    (out / "ALL_DONE").write_text("")
    print(json.dumps({k: doc[k] for k in
                      ("scene", "rule", "band1_fired", "band2", "band3_fired",
                       "scene_pass", "scene_drop",
                       "falsifier_triggered_on_this_scene", "geometry_gate",
                       "psnr_verdict")} | {"drift": [d["metric"] for d in doc["drift"]]},
                     indent=2))


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
