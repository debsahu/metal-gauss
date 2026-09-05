#!/usr/bin/env python3
"""Task 22 grading: score the arms, then apply BOTH rules and report both.

    bench/bilagrid_verdict.py --out-dir DIR --scene pg|ark --json OUT [--score]

TWO RULES APPLY AND THEY ASK DIFFERENT QUESTIONS. Reporting one would be
reporting half the result, so both are computed and neither is suppressed:

  * THE PLAN'S TASK-22 RULE, which is LPIPS-primary and is what this task was
    commissioned against: keep only if held-out LPIPS improves on P-GEOM by more
    than the base arm's measured n=3 floor AND by at least Task 21's materiality
    threshold 0.015, with no geometry column worse beyond its own floor. A
    PSNR-only improvement does not count.
  * THE OPERATOR'S THREE-BAND RULE (`3cfd8f3`), in bench/tier3_bands.py. It is
    implemented from the COMMITTED derivation in research/metal-gauss.md s13.6
    rather than imported: the existing implementation is in an UNCOMMITTED
    working file on another implementer's open branch (see that module's
    docstring), and two of its four thresholds are re-derived by its tests
    rather than merely restated.

AND THERE IS A CATEGORY TENSION BETWEEN THEM THAT I AM NOT GOING TO RESOLVE
SILENTLY. Band 2 REQUIRES on-seed@1cm to RISE beyond its floor. That was written
for a GEOMETRY lever (the dn neighbour gate). An appearance model is a
PHOTOMETRIC lever with no mechanism by which on-seed should rise, so a
geometry-neutral appearance model reads "WITHIN FLOOR" -> not adopted, no matter
what LPIPS does. Whether that is the intended reading for a photometric item is
an operator question. Both verdicts are printed side by side and the tension is
named in the output; this file does not pick.

Every number is read from a report JSON, a splatstats JSON or an lpips.json --
never from stdout.
"""
from __future__ import annotations

import argparse, json, os, statistics, subprocess, sys
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.tier3_bands import (                                       # noqa: E402
    COLLAPSE, DIRECTION, GEOMETRY_GATE, band1, band2, band3, verdict_for,
)

MATERIALITY = 0.015          # Task 21's pre-registered threshold
SPLATSTATS = Path("/Users/debsahu/Workspace/slam/analyze/splatstats")


def score(out: Path, tag: str, seed_cloud: str) -> None:
    """splatstats + LPIPS. Idempotent, and each step asserts its ARTIFACT rather
    than its exit status: a scorer that printed a number and wrote nothing is a
    state this project has been in."""
    stats = out / f"{tag}.stats.json"
    if not stats.exists():
        subprocess.run(["caffeinate", "-i", "uv", "run", "--frozen", "python",
                        "scripts/splat_stats.py", str(out / f"{tag}.ply"),
                        "--seed", seed_cloud, "--json", str(stats), "--quiet"],
                       cwd=str(SPLATSTATS), capture_output=True, text=True)
        if not stats.exists():
            raise SystemExit(f"{tag}: splatstats wrote no JSON")
    lp = out / f"{tag}.dump" / "lpips.json"
    if not lp.exists():
        subprocess.run([str(ROOT / ".venv/bin/python"), "scripts/lpips_eval.py",
                        str(out / f"{tag}.dump")], cwd=str(ROOT),
                       capture_output=True, text=True)
        if not lp.exists():
            raise SystemExit(f"{tag}: lpips_eval wrote no JSON")


def values(out: Path, tag: str) -> dict:
    rep = json.loads((out / f"{tag}.json").read_text())
    st = json.loads((out / f"{tag}.stats.json").read_text())
    lp = json.loads((out / f"{tag}.dump" / "lpips.json").read_text())
    m, sh = rep["metrics"], (rep["metrics"].get("shape") or {})
    v = {f"stats.{k}": x for k, x in st["metrics"].items() if isinstance(x, (int, float))}
    v.update({"run.psnr_masked": m["psnr_masked"], "run.psnr": m["psnr"],
              "run.lpips": lp["mean"], "run.ms_per_step": m["ms_per_step"],
              "run.n_splats": m["n_splats"], "run.aspect_p50": sh.get("aspect_p50"),
              "run.needle_frac": sh.get("needle_frac"),
              "run.hard_needle_frac": sh.get("hard_needle_frac")})
    return {"tag": tag, "seed": rep["resolved"].get("seed"),
            "appearance": rep["resolved"].get("appearance"),
            "git": (rep.get("env") or {}).get("git"),
            "seed_cloud": st.get("seed_cloud"),
            "wall_s": m.get("wall_s"),
            "appearance_state": m.get("appearance"),
            "lpips_n": lp["n"],
            "resolved": rep["resolved"],
            "values": {k: x for k, x in v.items() if isinstance(x, (int, float))}}


def table(arms: list) -> dict:
    keys = sorted(set.intersection(*(set(a["values"]) for a in arms)))
    return {k: {"values": [a["values"][k] for a in arms],
                "mean": statistics.mean(a["values"][k] for a in arms),
                "spread_n3": max(a["values"][k] for a in arms)
                             - min(a["values"][k] for a in arms),
                "repeat_pair_abs_diff": abs(arms[0]["values"][k] - arms[1]["values"][k])}
            for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True); ap.add_argument("--scene", required=True)
    ap.add_argument("--json", required=True); ap.add_argument("--seed-cloud", required=True)
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    out = Path(a.out_dir)
    base_tags = [f"{a.scene}_base_{s}" for s in "abc"]
    bila_tags = [f"{a.scene}_bila_{s}" for s in "abc"]
    if a.score:
        for t in base_tags + bila_tags:
            score(out, t, a.seed_cloud)

    base = [values(out, t) for t in base_tags]
    bila = [values(out, t) for t in bila_tags]
    # EVERY ARM MUST HAVE BEEN SCORED AGAINST THE SAME REFERENCE CLOUD, or the
    # geometry columns are not comparable. CLAUDE.md records an 11.6 deg error
    # from exactly this -- larger than every recipe gain it was used to judge.
    refs = {x["seed_cloud"] for x in base + bila}
    if len(refs) != 1:
        raise SystemExit(f"arms disagree about the reference cloud: {refs}")
    modes = {x["appearance"] for x in base}, {x["appearance"] for x in bila}
    if modes != ({"off"}, {"bilagrid"}):
        raise SystemExit(f"arms are not the configurations they claim: {modes}")

    fl, tr = table(base), table(bila)
    base_vals = {k: v["mean"] for k, v in fl.items()}
    t_vals = {k: v["mean"] for k, v in tr.items()}

    rows, verdict = {}, {}
    for k in sorted(set(t_vals) & set(base_vals)):
        d = t_vals[k] - base_vals[k]
        floor = fl[k]["spread_n3"]
        row = {"base_mean": base_vals[k], "base_spread_n3": floor,
               "treatment_mean": t_vals[k], "treatment_spread_n3": tr[k]["spread_n3"],
               "delta": d, "paired_same_seed": tr[k]["values"][0] - fl[k]["values"][0],
               "moves_beyond_base_floor": abs(d) > floor,
               "moves_beyond_both_floors": abs(d) > max(floor, tr[k]["spread_n3"])}
        if k in DIRECTION:
            row["verdict"] = verdict[k] = verdict_for(k, d, floor)
        rows[k] = row

    b1 = band1(t_vals, base_vals)      # per-arm only; see tier3_bands.band1
    b2 = band2(verdict)
    b3 = band3(t_vals["run.psnr_masked"], base_vals["run.psnr_masked"])
    # Beyond floor, below Band 1, and WORSE -- reported, never a DROP.
    drift = sorted(k for k, v in rows.items()
                   if v.get("verdict") == "WORSENED"
                   and not b1["per_arm"].get(k, {}).get("fired"))
    three_band_drop = bool(b1["fired"] or b2 == "FAIL" or b3["fired"])

    # ---- the plan's LPIPS-primary rule
    dl = base_vals["run.lpips"] - t_vals["run.lpips"]      # POSITIVE = improvement
    lp_floor = fl["run.lpips"]["spread_n3"]
    geom_worse = [k for k in GEOMETRY_GATE if verdict.get(k) == "WORSENED"]
    plan_keep = bool(dl > lp_floor and dl >= MATERIALITY and not geom_worse)
    plan = {"delta_lpips_improvement": dl, "base_lpips_floor_n3": lp_floor,
            "treatment_lpips_floor_n3": tr["run.lpips"]["spread_n3"],
            "materiality": MATERIALITY,
            "beats_floor": dl > lp_floor, "beats_materiality": dl >= MATERIALITY,
            "geometry_columns_worsened": geom_worse,
            "verdict": "KEEP" if plan_keep else "DROP",
            "note": ("A PSNR-only improvement does not count and an improvement on lego "
                     "does not count. On ARKitScenes the 0.015 materiality is NOT applied "
                     "as a decision -- Task 21's own post-hoc ceiling there is 0.00888, so "
                     "grading it against 0.015 would be grading it against a bar its own "
                     "ceiling forbids. That scene is reported and must not REGRESS.")}

    doc = {"kind": "bilagrid_verdict", "schema": 1, "scene": a.scene,
           "reference_cloud": refs.pop(),
           "arms": {"base": base, "bilagrid": bila},
           "floors": {"base": fl, "bilagrid": tr},
           "rows": rows,
           "three_band": {"rule": "tier3-three-band-2026-09-04 (imported from "
                                  "scripts/dn_gate_arms.py, not retyped)",
                          "band1": b1, "band2": b2, "band3": b3,
                          "drift_worsened_below_band1": drift, "drop": three_band_drop,
                          "verdict": "KEEP" if not three_band_drop and b2 == "PASS"
                                     else "DROP",
                          "category_tension":
                              "Band 2 REQUIRES on-seed@1cm to RISE. It was written for a "
                              "GEOMETRY lever; an appearance model is a PHOTOMETRIC one "
                              "with no mechanism by which on-seed should rise, so a "
                              "geometry-neutral result reads WITHIN FLOOR -> not adopted "
                              "whatever LPIPS does. Flagged, not resolved here."},
           "plan_rule": plan}
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(doc, indent=2))

    print(f"\n=== {a.scene} ===")
    print(f"  LPIPS         {base_vals['run.lpips']:.5f} -> {t_vals['run.lpips']:.5f}  "
          f"improvement {dl:+.5f}  base floor {lp_floor:.5f}  materiality {MATERIALITY}")
    print(f"  masked PSNR   {base_vals['run.psnr_masked']:.4f} -> "
          f"{t_vals['run.psnr_masked']:.4f}  ({t_vals['run.psnr_masked'] - base_vals['run.psnr_masked']:+.4f} dB, "
          f"floor {fl['run.psnr_masked']['spread_n3']:.4f})")
    for k in GEOMETRY_GATE:
        print(f"  {k:28s} {base_vals[k]:.5f} -> {t_vals[k]:.5f}  {verdict.get(k)}")
    print(f"  ms/step       {base_vals['run.ms_per_step']:.2f} -> "
          f"{t_vals['run.ms_per_step']:.2f}")
    print(f"  BAND1 fired {b1['fired']}  BAND2 {b2}  BAND3 fired {b3['fired']} "
          f"(loss {b3['loss_db']:+.4f} dB)")
    print(f"  three-band verdict: {doc['three_band']['verdict']}")
    print(f"  plan rule verdict : {plan['verdict']}")


if __name__ == "__main__":
    main()
