"""Tier 3 three-band keep/drop rule, implemented from its COMMITTED derivation.

THE RULE (`3cfd8f3`, the operator's 2026-09-04 amendment that replaced the
magnitude-blind "WORSENED anywhere = DROP"):

    Band 1  COLLAPSE     hard DROP, any one column
    Band 2  GEOMETRY     on-seed@1cm must RISE; thin-axis must not worsen
    Band 3  PHOTOMETRIC  hard DROP on a >0.25 dB PSNR loss, or crossing the 24 dB gate

WHY THIS IS IMPLEMENTED HERE RATHER THAN IMPORTED, which was the first choice.
An implementation exists at `scripts/dn_gate_arms.py`, but it is on
`feat/dn-neighbour-gate` (PR #4, open, owned by another implementer) AND, checked
2026-09-04, the band code is in that branch's UNCOMMITTED WORKING FILE: the
pushed commit a302cdb has 682 lines and no `band1` at all, while the local file
has 1651 and is `M`. Vendoring from a file another agent is actively editing
would be a dependency on something that changes underneath us, which is the one
thing this project's long-run rules forbid outright.

So the source of truth used here is the COMMITTED derivation in
`research/metal-gauss.md` s13.6, which is what that code itself cites. Every
Band-1 threshold is `sqrt(healthy x collapse)` in the column's natural space with
the adopted arm chosen PER COLUMN, and tests/test_tier3_bands.py RE-DERIVES the
two the note publishes the inputs for, rather than only asserting the constants:

    needle_frac  sqrt(2.8962 pp x 40.1558 pp)   = 10.784 pp  -> 0.108
    aspect_p50  -sqrt(0.07974   x 1.50112)      = -0.34598   -> 0.346

The other two (on-seed 0.185, LPIPS 0.017) are taken as published: the note gives
their thresholds but not the arm values they were derived from, so they are
constants with a stated provenance and NOT independently checked here. That
distinction is recorded rather than blurred.

They are conventions with a derivation, not measurements.
"""
from __future__ import annotations

import math

# +1 higher is better, -1 lower is better, 0 two-sided.
DIRECTION = {
    "stats.on_seed_frac_1cm": +1,
    "stats.on_seed_frac_2cm": +1,
    "stats.thin_axis_angle_p50": -1,
    "run.aspect_p50": +1,
    "run.needle_frac": -1,
    "run.lpips": -1,
    "run.psnr_masked": 0,
}
GEOMETRY_GATE = ("stats.on_seed_frac_1cm", "stats.thin_axis_angle_p50",
                 "run.aspect_p50", "run.needle_frac")
BAND2_GATE = ("stats.on_seed_frac_1cm", "stats.thin_axis_angle_p50")

COLLAPSE = {
    "run.needle_frac":        {"space": "abs", "worse": +1, "threshold": 0.108},
    "run.aspect_p50":         {"space": "log", "worse": -1, "threshold": 0.346},
    "stats.on_seed_frac_1cm": {"space": "log", "worse": -1, "threshold": 0.185},
    "run.lpips":              {"space": "abs", "worse": +1, "threshold": 0.017},
}
PSNR_DROP_DB = 0.25
STAGE4_PSNR_DB = 24.0


def verdict_for(metric: str, delta: float, floor: float) -> str:
    """IMPROVED / WORSENED / WITHIN FLOOR, in the column's own direction."""
    d = DIRECTION.get(metric, 0)
    if abs(delta) <= floor:
        return "WITHIN FLOOR"
    if d == 0:
        return "MOVED"
    return "IMPROVED" if (delta * d) > 0 else "WORSENED"


def collapse_delta(metric: str, value: float, reference: float) -> float:
    """How far `value` sits from `reference` TOWARD WORSE, in the column's space.

    POSITIVE = WORSE, always, whichever way the column runs. A sign error inverts
    every Band 1 test -- an arm that HALVED on-seed would read as an improvement
    and no collapse could ever fire -- so the sign has a test of its own.
    """
    spec = COLLAPSE[metric]
    if spec["space"] == "log":
        if value <= 0.0 or reference <= 0.0:
            raise ValueError(f"{metric}: log-space column needs positive values, "
                             f"got value={value!r} reference={reference!r}")
        d = math.log(value) - math.log(reference)
    else:
        d = value - reference
    return spec["worse"] * d


def band1(t_values: dict, base_values: dict) -> dict:
    """Band 1 -- COLLAPSE. Hard DROP; any ONE column. Comparison is STRICT: a
    delta exactly equal to the threshold has NOT fired.

    Only the per-arm half is computed here. The cumulative half needs a FROZEN
    scene anchor from a previous Tier 3 arm; this is the first appearance arm on
    either scene, so the anchor would be this arm's own floor mean and the
    cumulative delta would equal the per-arm delta exactly. Reporting that as a
    cumulative check that passed would be reporting a tautology as evidence.
    """
    row = {}
    for col, spec in COLLAPSE.items():
        if col not in t_values:
            raise ValueError(f"Band 1 column {col} missing from the treatment battery. "
                             f"A collapse column never measured must never read as "
                             f"'did not collapse'.")
        if col not in base_values:
            raise ValueError(f"Band 1 column {col} missing from the baseline.")
        d = collapse_delta(col, t_values[col], base_values[col])
        row[col] = {"value": t_values[col], "reference": base_values[col], "delta": d,
                    "threshold": spec["threshold"], "space": spec["space"],
                    "x_threshold": d / spec["threshold"], "fired": d > spec["threshold"]}
    fired = [k for k, v in row.items() if v["fired"]]
    return {"per_arm": row, "per_arm_fired": fired, "fired": bool(fired),
            "cumulative": None,
            "cumulative_note":
                "NOT COMPUTED. The cumulative half needs a frozen anchor from a previous "
                "Tier 3 arm on this scene; this is the first, so the anchor would be this "
                "arm's own floor mean and the check would be vacuous by construction. "
                "Absent, not passed."}


def band2(verdicts: dict) -> str:
    """PASS if on-seed@1cm IMPROVED beyond floor and thin-axis did not worsen;
    FAIL if either worsened; WITHIN FLOOR otherwise.

    Aspect and needles are deliberately NOT read here -- moving them to Band 1,
    where a 2.5% move and a 78% collapse get different answers, IS the amendment.
    """
    missing = [k for k in BAND2_GATE if verdicts.get(k) is None]
    if missing:
        raise ValueError(f"Band 2 columns missing: {missing}. An absent gate column "
                         f"must never read as a pass.")
    on_seed, thin = (verdicts[k] for k in BAND2_GATE)
    if on_seed == "WORSENED" or thin == "WORSENED":
        return "FAIL"
    return "PASS" if on_seed == "IMPROVED" else "WITHIN FLOOR"


def band3(psnr_treatment: float, psnr_baseline: float) -> dict:
    """Hard DROP on a PSNR LOSS greater than 0.25 dB, or on falling below the
    24 dB Stage 4 gate from at or above it.

    ONE-SIDED by construction: the rule says "falls by". A gain is not a
    regression, and the older two-sided "must be WITHIN floor" reading is what
    made every Tier 3 arm unable to PASS whatever its geometry did. Both
    comparisons are strict.
    """
    loss = psnr_baseline - psnr_treatment
    return {"baseline": psnr_baseline, "treatment": psnr_treatment, "loss_db": loss,
            "allowance_db": PSNR_DROP_DB, "exceeds_allowance": loss > PSNR_DROP_DB,
            "crossed_stage4_gate": psnr_baseline >= STAGE4_PSNR_DB > psnr_treatment,
            "baseline_above_stage4": psnr_baseline >= STAGE4_PSNR_DB,
            "fired": bool(loss > PSNR_DROP_DB
                          or psnr_baseline >= STAGE4_PSNR_DB > psnr_treatment)}
