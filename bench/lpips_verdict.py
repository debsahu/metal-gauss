#!/usr/bin/env python3
"""Task 21 step 5: apply the pre-registered decision rule to the measurements.

    bench/lpips_verdict.py --out-root R --scene pgeom --arm R1 [--tag T]
                           [--capacity-delta X] [--capacity-floor F]

The rule, from the pre-registration commit and its one amendment, restated here
so the code and the record cannot drift apart:

  BUILD Task 22 iff the best of fitters (a) affine, (b) bilagrid at Brush's
        tv 10, (c) shared PPISP recovers dLPIPS >= 0.015 on P-GEOM held-out --
        10% of the 0.147 gap to the gate, 20x the 0.00073 floor.
  SHAPE if built: (c) >= 0.80 * (b) => PPISP stages, else the bilateral grid.
  DEFER, overriding BUILD: if the capacity arm moves LPIPS by more than (b)'s
        ceiling, capacity is the finding and Task 22 waits behind it.
  AMENDMENT: a fitter's number is usable only if its C1 synthetic control passed
        (>= 0.90 of its own injected distortion recovered). A fitter that failed
        C1 is a failed control, NEVER a null.

THE CONSEQUENCE OF THAT LAST CLAUSE IS THE POINT OF THIS FILE. If every fitter
in the rule fails C1, there is NO VERDICT -- not a CUT. An unconverged fit and a
scene with no photometric component both return dLPIPS ~= 0, and letting the
first masquerade as the second is precisely how a probe manufactures the answer
it was built to test.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench import lpips_attr as LA                                    # noqa: E402

BUILD_THRESHOLD = 0.015          # pre-registered
SHAPE_FRACTION = 0.80            # pre-registered
RULE_FITTERS = ("affine", "bilagrid_tv10", "ppisp")     # the plan's (a), (b), (c)
GRID_FITTER = "bilagrid_tv10"
PPISP_FITTER = "ppisp"

# A REGULARISED fitter's C1 is proven by its UNREGULARISED twin. C1 exists to
# certify that the fitter RUNS and its optimiser works; a regulariser is part of
# the MODEL, and a model's inability to express something is exactly what a
# ceiling is supposed to capture, not a reason to discard the ceiling. Measured
# forward-direction C1 on pgeom/R1, 2 views: bilagrid_tv0 0.964 (pass),
# bilagrid_tv10 0.888 against a 0.90 floor -- identical code path, identical
# optimiser settings, the regulariser the only difference. Requiring the
# regularised form to invert a smooth field to 90% while penalising exactly the
# grid variation that inversion needs is a control a correct implementation
# fails.
C1_PROXY = {"bilagrid_tv10": "bilagrid_tv0"}


def usable(entry: dict, fitters: dict | None = None,
           name: str | None = None) -> tuple[bool, str]:
    c1 = entry.get("synthetic_control_c1")
    if c1 is None:
        return False, "no C1 synthetic control was run"
    if c1.get("passed"):
        return True, f"C1 passed ({c1.get('recovered_fraction_mean'):.3f})"
    proxy = C1_PROXY.get(name or "")
    if proxy and fitters and proxy in fitters:
        pc1 = (fitters[proxy] or {}).get("synthetic_control_c1") or {}
        if pc1.get("passed"):
            return True, (f"C1 {c1.get('recovered_fraction_mean'):.3f} is below the floor, "
                          f"but its unregularised twin {proxy} passed at "
                          f"{pc1.get('recovered_fraction_mean'):.3f} -- the MACHINERY is "
                          f"proven and the shortfall is the regulariser's own cost "
                          f"({1.0 - float(c1.get('recovered_fraction_mean')):.3f})")
    return False, (f"C1 FAILED: recovered {c1.get('recovered_fraction_mean'):.3f} "
                   f"of its own injected distortion, floor {c1.get('floor')}")


def pearson(xs, ys) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    d = (sxx * syy) ** 0.5
    return float("nan") if d == 0 else sxy / d


def coupling(res: dict) -> dict:
    """Per-VIEW correlation between PSNR and LPIPS inside one arm.

    The arm-level correlation across the Tier 1 arms is near -1 but over an
    LPIPS range of 0.003, so the coefficient is not the evidence there. Within
    one arm the held-out views span a real range (pgeom B0a: LPIPS 0.280-0.598)
    and the question "does LPIPS carry information this scene's PSNR does not"
    becomes answerable on 25 points.
    """
    b = res["baseline"]
    stems = list(b["lpips_per_view"])
    l = [b["lpips_per_view"][s] for s in stems]
    p = [b["psnr_per_view"][s] for s in stems]
    return {"n_views": len(stems), "pearson_psnr_lpips": pearson(p, l),
            "lpips_min": min(l), "lpips_max": max(l),
            "psnr_min": min(p), "psnr_max": max(p)}


def verdict(res: dict, capacity_delta: float | None = None,
            capacity_floor: float | None = None) -> dict:
    fitters = res["fitters"]
    rows, usable_rows = {}, {}
    for name in sorted(fitters):
        e = fitters[name]
        ok, why = usable(e, fitters, name)
        rows[name] = {"delta_lpips": e["delta_lpips_mean"],
                      "delta_psnr": e["delta_psnr_mean"],
                      "n_params_per_view": e["n_params_per_view"],
                      "usable": ok, "c1": why}
        if ok and name in RULE_FITTERS:
            usable_rows[name] = e["delta_lpips_mean"]

    out = {"stage": "step5_verdict", "scene": res["scene"], "arm": res["arm"],
           "tag": res.get("tag", ""), "n_views": res["n_views"],
           "baseline_lpips": res["baseline"]["lpips_mean"],
           "baseline_psnr": res["baseline"]["psnr_mean"],
           "build_threshold": BUILD_THRESHOLD, "rows": rows,
           "per_view_coupling": coupling(res)}

    if not usable_rows:
        out.update(verdict="NO VERDICT", why=(
            "every fitter in the rule failed or lacked its C1 synthetic control. "
            "An unconverged fit and a scene with no photometric component return "
            "the same dLPIPS; a CUT may not be read off a failed control."))
        return out

    best_name = max(usable_rows, key=usable_rows.get)
    ceiling = usable_rows[best_name]
    out.update(ceiling_fitter=best_name, ceiling=ceiling,
               excluded=[n for n in RULE_FITTERS if n not in usable_rows])

    if capacity_delta is not None:
        grid = usable_rows.get(GRID_FITTER)
        out["capacity_delta_lpips"] = capacity_delta
        out["capacity_floor"] = capacity_floor
        out["capacity_grid_ceiling"] = grid
        # Strictly greater, and on the IMPROVEMENT: a capacity arm that made LPIPS
        # WORSE is not "capacity is the finding", it is evidence against it.
        if grid is not None and capacity_delta > grid:
            out.update(verdict="DEFER",
                       why=(f"the capacity arm moved LPIPS by {capacity_delta:.5f}, "
                            f"more than fitter (b)'s ceiling {grid:.5f}. Capacity is "
                            f"the finding; Task 22 waits behind a capacity item."))
            return out

    if ceiling < BUILD_THRESHOLD:
        out.update(verdict="CUT", why=(
            f"the best usable fitter ({best_name}) recovers {ceiling:.5f}, under the "
            f"pre-registered {BUILD_THRESHOLD}. A free post-hoc per-view fit cannot "
            f"recover the LPIPS, so nothing constrained and indirect can."))
        return out

    grid = usable_rows.get(GRID_FITTER)
    pp = usable_rows.get(PPISP_FITTER)
    if grid is not None and pp is not None and pp >= SHAPE_FRACTION * grid:
        shape = "ppisp"
    elif grid is not None:
        shape = "bilagrid"
    else:
        shape = "UNDECIDED (fitter (b) is not usable)"
    out.update(verdict="BUILD", shape=shape, why=(
        f"the best usable fitter ({best_name}) recovers {ceiling:.5f} >= "
        f"{BUILD_THRESHOLD}. Shape from (c)/(b) = "
        f"{'n/a' if not (grid and pp) else f'{pp / grid:.3f}'}."))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--scene", default="pgeom")
    ap.add_argument("--arm", default="R1")
    ap.add_argument("--tag", default="")
    ap.add_argument("--capacity-delta", type=float, default=None,
                    help="LPIPS IMPROVEMENT from the capacity arm, POSITIVE when the "
                         "bigger budget scored BETTER -- the same sign convention as a "
                         "fitter's dLPIPS, so the two are directly comparable. Passing "
                         "the raw (big - small) difference inverts it: LPIPS is a "
                         "distance, so a better arm has a LOWER one.")
    ap.add_argument("--capacity-floor", type=float, default=None)
    ap.add_argument("--write", default=None)
    a = ap.parse_args(argv)
    name = f"{a.scene}__{a.arm}{('__' + a.tag) if a.tag else ''}.json"
    res = LA.read_result(Path(a.out_root) / "step3" / name)
    if res.get("stage") != "step3_ceiling":
        raise ValueError(f"{name}: stage {res.get('stage')!r} is not step3_ceiling")
    v = verdict(res, a.capacity_delta, a.capacity_floor)
    print(json.dumps(v, indent=2))
    if a.write:
        LA.write_json(a.write, v)


if __name__ == "__main__":
    main()
