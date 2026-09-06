#!/usr/bin/env python3
"""Shape columns from an exported ply, including the HARD-NEEDLE fraction.

    scripts/ply_shape.py ARM.ply --report ARM.report.json --out ARM.shape.json

WHY THIS EXISTS AND NOT A REPORT FIELD. `hard_needle_frac` was added to
`metal_gauss.train.shape_metrics` on 2026-09-04, so every arm trained before that -- Task
19's ten among them -- has a ply and no column. Retraining eight hours of GPU to recover a
reported-only statistic would be absurd; the ply already holds every scale.

WHAT A HARD NEEDLE IS. `aspect = smid / smax < 0.01`. The threshold is derived from the
delivery format, not chosen: splat-transform's SOG writer normalises the rotation
quaternion, scales it by +-sqrt(2), and stores the smallest three components as
`255 * (q * 0.5 + 0.5)` in uint8. One step is sqrt(2)/255 = 0.0055459 in true component
units, the worst-case round-to-nearest error is step/2 per component over three, and a
quaternion perturbation of norm e is a rotation of 2e -- 0.0096058 rad. Below that aspect,
a splat's minor in-plane half-axis is smaller than the rim displacement its own quantised
orientation produces, so its orientation is undeliverable however well it was trained.

IT IS REPORTED, NEVER GATED. See `scripts/plane_aux_arms.py`: a column that is present for
some arms and absent for others cannot be a gate without becoming the exact failure this
project keeps repeating.

THE CROSS-CHECK IS THE POINT, not the convenience. This file re-implements, in numpy over a
hand-parsed ply, what the trainer computed in torch on the GPU. Two things could be wrong
-- the ply field order, and the median convention -- and both fail SILENTLY, producing a
plausible number for a different quantity. So it recomputes `aspect_p50`, `needle_frac`,
`smid_p50_mm` and `smax_p50_mm` too, and REFUSES TO WRITE unless all four reproduce the
report the trainer wrote. A shape file with `verified_against_report: true` is the only
kind `plane_aux_arms.battery` will read.

`torch.median` returns the LOWER of the two middle values on an even count; `np.median`
averages them. At 500,000 splats that difference is real, and it is exactly the kind of
silent disagreement the cross-check exists to catch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HARD_NEEDLE_ASPECT = 0.01
NEEDLE_ASPECT = 0.1


def read_ply_scales(path: Path) -> np.ndarray:
    """`scale_0..2` from a binary_little_endian float32 ply, as an (N, 3) float64 array.

    Parses the header rather than assuming the INRIA layout: the offsets are computed from
    the declared property order, so a ply with SH bands, extra fields or a different order
    reads correctly, and one this parser cannot handle raises instead of returning
    plausible garbage from the wrong byte offsets.
    """
    with path.open("rb") as f:
        if f.readline().strip() != b"ply":
            raise SystemExit(f"{path}: not a ply")
        fmt, count, props = None, None, []
        while True:
            line = f.readline()
            if not line:
                raise SystemExit(f"{path}: header ended without end_header")
            tok = line.split()
            if not tok:
                continue
            if tok[0] == b"format":
                fmt = tok[1]
            elif tok[0] == b"element":
                if tok[1] == b"vertex":
                    count = int(tok[2])
                elif count is not None:
                    break            # a second element: vertex properties are complete
            elif tok[0] == b"property" and count is not None:
                if len(tok) != 3:
                    raise SystemExit(f"{path}: list properties are not supported: {line!r}")
                props.append((tok[1].decode(), tok[2].decode()))
            elif tok[0] == b"end_header":
                break
        if fmt != b"binary_little_endian":
            raise SystemExit(f"{path}: format is {fmt!r}; only binary_little_endian is "
                             f"supported (an ascii ply would be parsed wrong, silently)")
        if any(t != "float" for t, _ in props):
            raise SystemExit(f"{path}: mixed property widths are not supported; "
                             f"types seen: {sorted({t for t, _ in props})}")
        names = [n for _, n in props]
        want = ["scale_0", "scale_1", "scale_2"]
        missing = [w for w in want if w not in names]
        if missing:
            raise SystemExit(f"{path}: no {missing} in the vertex properties")
        raw = np.fromfile(f, dtype="<f4", count=count * len(names))
    if raw.size != count * len(names):
        raise SystemExit(f"{path}: body holds {raw.size} floats, header declares "
                         f"{count * len(names)}")
    rows = raw.reshape(count, len(names))
    return np.stack([rows[:, names.index(w)] for w in want], axis=1).astype(np.float64)


def lower_median(x: np.ndarray) -> float:
    """`torch.median`'s convention: the lower of the two middle values on an even count.

    `np.median` averages them, and at 500,000 splats that is a different number. Matching
    the reference is what makes the cross-check meaningful rather than approximately true.
    """
    return float(np.partition(x, (x.size - 1) // 2)[(x.size - 1) // 2])


# TWO CONVENTIONS ARE IN USE IN THIS PROJECT'S OWN ARTIFACTS, and they are not
# interchangeable past ~5 significant figures. Established 2026-09-04 by recomputing three
# Tier 1 arms both ways and matching to 10 digits:
#
#   `lower`    what the TRAINER writes into `metrics.shape` -- `torch.median`, float32.
#              Task 19's ten arms are all of this kind.
#   `average`  what Tier 1's `collected.json` holds under `ply.*` -- `np.median`, float64.
#              research/metal-gauss.md section 8.1's aspect and needle figures are these.
#
# On B0a they differ by 1.9e-6 relative on `smid_p50_mm` -- far too small to move any
# verdict, and far too large to be float noise, so a cross-check tight enough to catch a
# misread field WILL trip over it. Naming the convention is the fix; guessing is not.
MEDIANS = {"lower": lower_median, "average": lambda x: float(np.median(x))}


def shape_from_ply(path: Path, median: str = "lower") -> dict:
    med = MEDIANS[median]
    s = np.sort(np.exp(read_ply_scales(path)), axis=1)
    aspect = s[:, 1] / np.maximum(s[:, 2], 1e-12)
    return {"n_splats": int(s.shape[0]),
            "median_convention": median,
            "aspect_p50": med(aspect),
            # FRACTIONS, so both conventions agree exactly on these two -- they are counts,
            # not order statistics. The hard-needle column is therefore comparable across
            # Tier 1 and Task 19 even though their medians are not.
            "needle_frac": float((aspect < NEEDLE_ASPECT).mean()),
            "hard_needle_frac": float((aspect < HARD_NEEDLE_ASPECT).mean()),
            "smid_p50_mm": med(s[:, 1]) * 1000.0,
            "smax_p50_mm": med(s[:, 2]) * 1000.0}


# Relative tolerances, chosen against the two error sources rather than by taste.
#
# The report stores the trainer's float32 values, so every column carries ~6e-8 of relative
# storage error: `needle_frac` 0.157376 comes back as 0.1573760062456131, a 6.2e-9 relative
# difference that a 1e-9 tolerance rejects. That is a false alarm, and this file's first run
# raised it on 8 of 10 real arms.
#
# The tolerance still has to be sharp enough to catch what it exists to catch. On
# `needle_frac` the smallest REAL disagreement is one misclassified splat: at 500,000
# splats that is 2e-6 relative, 20x above the 1e-7 bar below. On the medians, one step of
# the sorted array is far larger still. So both bars sit between float32 storage noise and
# the smallest meaningful error, with an order of magnitude of margin either side.
CHECK_TOL = {"aspect_p50": 1e-6, "needle_frac": 1e-7, "smid_p50_mm": 1e-6,
             "smax_p50_mm": 1e-6}


# The cross-check is only evidence if it actually compared something. These two columns
# are the ones that pin the two silent failure modes -- field order and median convention --
# so a "verification" that did not reach both is not a verification.
REQUIRED_CHECKS = ("aspect_p50", "needle_frac")


def reference_shape(report: Path, arm: str | None) -> dict:
    """The shape columns to reproduce, from an arm's own report or from a `collected.json`.

    THE SECOND SOURCE IS NOT A CONVENIENCE. `metrics.shape` was added to the trainer
    mid-batch, so Tier 1's B0a, B0b, B0c, F1 and R1_openweights -- the VOID row among them
    -- have plys and no shape block. Their ply-derived columns live in that batch's
    `collected.json` under `ply.*` keys, which is where research/metal-gauss.md section 8.1's
    aspect and needle figures came from. Checking against those is what makes a Tier 1
    hard-needle number provenance-bearing rather than merely computed.
    """
    doc = json.loads(report.read_text())
    if arm is None:
        return doc["metrics"]["shape"]
    if arm not in doc:
        raise SystemExit(f"{report}: no arm {arm!r}; it holds {sorted(doc)[:12]}")
    row = doc[arm]
    got = {k[len("ply."):]: v for k, v in row.items() if k.startswith("ply.")}
    if not got:
        raise SystemExit(f"{report}: arm {arm!r} carries no `ply.*` shape columns, so there "
                         f"is nothing to cross-check against.")
    return got


def convention_hint(ply: Path | None, sh: dict, used: str) -> str:
    """If the OTHER median convention would have matched, say so.

    A 2e-6 disagreement is otherwise a mystery that reads like a corrupt ply. It is not
    diagnosis by guessing: the alternative is recomputed and compared, so the hint is only
    printed when it is true.
    """
    if ply is None:
        return ""
    other = "average" if used == "lower" else "lower"
    try:
        alt = shape_from_ply(ply, other)
    except SystemExit:
        return ""
    if all(abs(sh[k] - alt[k]) <= CHECK_TOL[k] * max(1.0, abs(sh[k]))
           for k in CHECK_TOL if k in sh):
        return (f"  --> the reference DOES match the {other!r} median convention. "
                f"Re-run with --median {other}.\n")
    return ""


def cross_check(got: dict, report: Path, arm: str | None = None,
                ply: Path | None = None) -> dict:
    """Every column the reference already recorded must reproduce, or nothing is written.

    Returns the columns it ACTUALLY compared. An empty return is not a pass: `main` refuses
    to stamp `verified_against_report` unless `REQUIRED_CHECKS` were among them. Without
    that, a report with no shape block would sail through with zero comparisons and be
    written out as verified -- a check satisfied by something other than the thing being
    checked, which is the failure this project keeps repeating.
    """
    sh = reference_shape(report, arm)
    hint = convention_hint(ply, sh, got.get("median_convention", "lower"))
    bad = {}
    for k, tol in CHECK_TOL.items():
        if k not in sh:
            continue
        a, b = sh[k], got[k]
        if abs(a - b) > tol * max(1.0, abs(a)):
            bad[k] = {"report": a, "recomputed": b, "tol": tol}
    if bad:
        raise SystemExit(
            f"ply-recomputed shape does not reproduce {report}: {json.dumps(bad, indent=2)}\n"
            f"{hint}"
            f"Either the ply is not this arm's, the field order was read wrong, or the "
            f"median convention differs. A hard-needle number computed alongside a "
            f"disagreement describes some other reconstruction.")
    return {k: sh[k] for k in CHECK_TOL if k in sh}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ply")
    ap.add_argument("--report", required=True,
                    help="the arm's report JSON; its metrics.shape block is the reference "
                         "the recomputation must reproduce. Not optional: an unverified "
                         "shape file is refused by plane_aux_arms.battery.")
    ap.add_argument("--arm", default=None,
                    help="read the reference from a batch `collected.json` under this arm's "
                         "`ply.*` keys instead of from `metrics.shape`. Needed for the Tier "
                         "1 arms, whose reports predate the trainer's shape block.")
    ap.add_argument("--median", default="lower", choices=sorted(MEDIANS),
                    help="`lower` = torch.median, what the trainer's metrics.shape holds "
                         "(Task 19). `average` = np.median, what Tier 1's collected.json "
                         "ply.* columns hold. See MEDIANS; the cross-check will tell you "
                         "if you picked the wrong one.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ply, rep, out = Path(a.ply), Path(a.report), Path(a.out)
    got = shape_from_ply(ply, a.median)
    checked = cross_check(got, rep, a.arm, ply)
    unchecked = [k for k in REQUIRED_CHECKS if k not in checked]
    if unchecked:
        raise SystemExit(
            f"{rep} supplied no reference for {unchecked}, so nothing pinned the ply field "
            f"order or the median convention. Refusing to write a shape file stamped as "
            f"verified when the verification did not happen. Pass --arm to read a batch "
            f"collected.json, or point --report at a report that carries metrics.shape.")
    doc = {"schema": 1, "ply": str(ply), "report": str(rep), "arm": a.arm,
           "hard_needle_aspect": HARD_NEEDLE_ASPECT,
           "verified_against_report": True,
           "reproduced": checked, **got}
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({k: doc[k] for k in
                      ("ply", "n_splats", "median_convention", "aspect_p50", "needle_frac",
                       "hard_needle_frac", "smid_p50_mm", "smax_p50_mm")}, indent=2))


if __name__ == "__main__":
    main()
