"""Tier 3 three-band keep/drop rule, implemented from its COMMITTED derivation.

THE RULE (`3cfd8f3`, the operator's 2026-09-04 amendment that replaced the
magnitude-blind "WORSENED anywhere = DROP"):

    Band 1  COLLAPSE     hard DROP, any one column, per-arm AND cumulative
    Band 2  GEOMETRY     on-seed@1cm must RISE; thin-axis must not worsen
    Band 3  PHOTOMETRIC  hard DROP on a >0.25 dB PSNR loss, or crossing the 24 dB gate
    DRIFT                beyond floor, below Band 1, and WORSE -- reported, never a DROP

THIS FILE IS NOW THE ONLY COPY OF THE RULE THAT TASK 20 AND TASK 22 SHARE.
It began as Task 22's re-implementation from the committed derivation, written
because Task 20's copy was then in an uncommitted working file on another
branch. Both are now committed, they were DIFFERENTIALLY AUDITED -- 4,079 checks
over both implementations, including at each threshold and +/-1e-12 either side,
ZERO disagreements -- and the audit's recommendation was to keep the THRESHOLD
PROVENANCE from this file and the CUMULATIVE HALF from Task 20's. That is what
happened here: the provenance block below is unchanged, and `band1`'s cumulative
half, `_collapse_side`, `transfers_between_scenes`, `threshold_relative`,
`DRIFT_SCOPE`, `ANCHOR_CONFIG_KEYS` and `drift_columns` came from
`scripts/dn_gate_arms.py`, which now IMPORTS them from here rather than
redefining them.

Collapsing the two copies costs the ability to RE-RUN that differential audit --
there is only one implementation left to differ from. The replacement safeguard
is `tests/mutants_dn_gate_grade.py`, which mutates THIS file and requires each
mutant to kill a NAMED test.

A THIRD implementation still exists, in `scripts/plane_aux_arms.py` (Task 19).
It is deliberately NOT rewired: `scripts/mutate_tier3_rule.py` mutates that file
by literal string substitution and hard-fails on an anchor that is not unique, so
editing it breaks the audit battery's anchors in the direction where mutants stop
being well-defined rather than where a test goes red. Retargeting that battery is
separate work, and this is a recorded deferral rather than an oversight.

WHERE THE THRESHOLDS COME FROM. The source of truth is the COMMITTED derivation
in `research/metal-gauss.md` s13.6. Every Band-1 threshold is
`sqrt(healthy x collapse)` in the column's natural space with the adopted arm
chosen PER COLUMN, and tests/test_tier3_bands.py RE-DERIVES the two the note
publishes the inputs for, rather than only asserting the constants:

    needle_frac  sqrt(2.8962 pp x 40.1558 pp)   = 10.784 pp  -> 0.108
    aspect_p50  -sqrt(0.07974   x 1.50112)      = -0.34598   -> 0.346

The other two (on-seed 0.185, LPIPS 0.017) are taken as published: the note gives
their thresholds but not the arm values they were derived from, so they are
constants with a stated provenance and NOT independently checked here. That
distinction is recorded rather than blurred.

They are conventions with a derivation, not measurements.

ERRORS ARE `ValueError`, NEVER `SystemExit`. Task 20's copy raised `SystemExit`,
which was right for a script and wrong for a library: it is not catchable by
`except Exception`, it cannot be asserted on without importing the CLI's
conventions into a unit test, and a caller that wanted to report several bad
columns at once could not. The CLI boundary is where a refusal becomes an exit --
`scripts/dn_gate_arms.main` converts a `ValueError` from this module into a
`SystemExit` carrying the same message, so the operator-visible behaviour is
unchanged and `tests/test_dn_gate_grade.py` asserts that conversion.
"""
from __future__ import annotations

import math

#: Column names, so the two thin-axis statistics can never be confused for each other.
ON_SEED_1CM = "stats.on_seed_frac_1cm"
#: splatstats' DEFAULT thin-axis column: gated to splats within `thin_axis_gate_tolerance_m`
#: (0.05) of the seed. Two arms scored this way are measured over DIFFERENT POPULATIONS.
THIN_AXIS_GATED = "stats.thin_axis_angle_p50"
#: The same statistic from a second splatstats run with `--thin-axis-gate -1`, i.e. over
#: EVERY splat. See `BAND2_GATE_UNGATED` for why a band must read this one.
THIN_AXIS_UNGATED = "stats.thin_axis_angle_p50_ungated"

# +1 higher is better, -1 lower is better, 0 two-sided.
DIRECTION = {
    ON_SEED_1CM: +1,
    "stats.on_seed_frac_2cm": +1,
    THIN_AXIS_GATED: -1,
    THIN_AXIS_UNGATED: -1,
    "run.aspect_p50": +1,
    "run.needle_frac": -1,
    "run.lpips": -1,
    "run.psnr_masked": 0,
}
GEOMETRY_GATE = (ON_SEED_1CM, THIN_AXIS_GATED,
                 "run.aspect_p50", "run.needle_frac")
BAND2_GATE = (ON_SEED_1CM, THIN_AXIS_GATED)

#: THE GATED THIN-AXIS COLUMN IS A COMPOSITION STATISTIC, NOT AN ORIENTATION ONE, whenever
#: two arms admit different numbers of splats. Established by Task 22 on 2026-09-05, BEFORE
#: the Task 20 arms were graded, so it is not post-hoc: Task 19's headline -2.35 deg
#: REVERSED to +0.78 deg WORSE on ARKitScenes once the two arms were compared at equal
#: admitted population, and the apparent gain tracked admitted-population excess at
#: r = -0.995. splatstats' default gate admits only splats within 5 cm of the seed --
#: often ~230k of 500k -- so the excess is large and the confound is live at every arm.
#: A grader that reads thin-axis therefore reads the UNGATED column and REPORTS BOTH.
BAND2_GATE_UNGATED = (ON_SEED_1CM, THIN_AXIS_UNGATED)
GEOMETRY_GATE_UNGATED = (ON_SEED_1CM, THIN_AXIS_UNGATED,
                         "run.aspect_p50", "run.needle_frac")

COLLAPSE = {
    "run.needle_frac":        {"space": "abs", "worse": +1, "threshold": 0.108},
    "run.aspect_p50":         {"space": "log", "worse": -1, "threshold": 0.346},
    ON_SEED_1CM:              {"space": "log", "worse": -1, "threshold": 0.185},
    "run.lpips":              {"space": "abs", "worse": +1, "threshold": 0.017},
}
PSNR_DROP_DB = 0.25
STAGE4_PSNR_DB = 24.0

#: Every column DRIFT may be reported on: the four collapse columns, the Band 2 pair in
#: both thin-axis forms, and masked PSNR. `dict.fromkeys` dedupes while keeping the order.
DRIFT_SCOPE = tuple(dict.fromkeys(tuple(COLLAPSE) + BAND2_GATE + BAND2_GATE_UNGATED
                                  + ("run.psnr_masked",)))

#: What must match between a frozen anchor and the arm being graded. `steps` and
#: `num_downscales` are in here because both change what a 30k arm's geometry columns
#: settle at, and neither is named in the anchor's own re-measure sentence -- which is
#: exactly why they are the two that would slip through.
ANCHOR_CONFIG_KEYS = ("budget", "steps", "max_resolution", "num_downscales")


def verdict_for(metric: str, delta: float, floor: float) -> str:
    """IMPROVED / WORSENED / WITHIN FLOOR, in the column's own direction.

    `abs(delta) > floor` is STRICT: a delta exactly equal to the floor has not cleared it.
    """
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


def transfers_between_scenes(spec: dict) -> bool:
    """A LOG threshold is a ratio and means the same relative change on any baseline; an
    ABSOLUTE one does not. DERIVED from the space rather than declared, so the two cannot
    drift apart -- and so the grade can say which of the four transfer.

    This is the machine-readable form of Task 20's pre-registration section 5:
    `needle_frac +10.8 pp` is 71% relative on P-GEOM's 0.1516 baseline and ~46% on
    P-MASK's 0.2348, so quoting it as a constant across scenes is quoting two different
    rules.
    """
    return spec["space"] == "log"


def threshold_relative(spec: dict, reference: float):
    """The threshold as a FRACTION OF THE BASELINE, in one comparable form for both
    spaces, so a reader never has to convert between pp and dlog to see how big it is."""
    if spec["space"] == "abs":
        return (spec["threshold"] / reference) if reference else None
    t = spec["threshold"]
    return (1.0 - math.exp(-t)) if spec["worse"] < 0 else (math.exp(t) - 1.0)


def _collapse_side(values: dict, reference: dict, what: str) -> dict:
    """One side of Band 1 -- every collapse column against one reference dict."""
    row = {}
    for col, spec in COLLAPSE.items():
        if col not in values:
            raise ValueError(f"Band 1 column {col} is missing from the treatment "
                             f"battery. A collapse column that was never measured must "
                             f"never read as 'did not collapse'.")
        if col not in reference:
            raise ValueError(f"Band 1 column {col} is missing from the {what}. An anchor "
                             f"or baseline that predates a column cannot testify about "
                             f"that column.")
        d = collapse_delta(col, values[col], reference[col])
        thr = spec["threshold"]
        row[col] = {"value": values[col], "reference": reference[col], "delta": d,
                    "threshold": thr, "x_threshold": d / thr, "space": spec["space"],
                    # Pre-registration section 5: state the scene's own baseline beside
                    # the threshold rather than applying a constant from another scene.
                    "scene_baseline": reference[col],
                    "threshold_relative_to_baseline": threshold_relative(spec,
                                                                         reference[col]),
                    "threshold_transfers_between_scenes": transfers_between_scenes(spec),
                    "fired": d > thr}
    return row


#: What `band1` says when it was given no anchor. Asserted verbatim by
#: tests/test_tier3_bands.py: an absent cumulative half must read ABSENT, never PASSED.
NO_ANCHOR_NOTE = (
    "NOT COMPUTED. The cumulative half needs a frozen anchor from a previous Tier 3 arm "
    "on this scene, and none was supplied. Absent, not passed.")


def band1(t_values: dict, base_values: dict, anchor_values: dict | None = None,
          self_anchored: bool = False) -> dict:
    """Band 1 -- COLLAPSE. Hard DROP; any ONE column; per-arm AND cumulative.

    Per-arm is against this arm's own re-measured base. Cumulative is against the scene's
    FROZEN anchor, and it is the half that stops the rule ratcheting: four accepted 8 pp
    needle drifts are a 32 pp collapse that no single arm ever fired on.

    Comparison is STRICT: a delta exactly equal to the threshold has NOT fired.

    `anchor_values=None` IS THE HONEST ANSWER FOR AN ARM WITH NO PREDECESSOR, not a
    convenience default. Task 22's appearance arm was the first Tier 3 arm on either
    scene, so its only available anchor would have been its own floor mean and the
    cumulative delta would have equalled the per-arm delta exactly. Reporting that as a
    cumulative check that passed would be reporting a tautology as evidence, so the half
    is returned as `None` with `NO_ANCHOR_NOTE` and the caller cannot mistake it.

    VACUITY IS MEASURED, NOT READ OFF A FLAG. When an anchor IS given, on a self-anchored
    scene's FIRST arm the anchor is the floor mean and the cumulative delta equals the
    per-arm delta exactly, so the check decides nothing -- but on the second arm the
    floors have moved and it starts deciding. A harness that reported vacuity from
    `self_anchored` would go on saying so forever, exactly when the check begins to
    matter, which is why `self_anchored` is RECORDED here and never used to decide.
    """
    per = _collapse_side(t_values, base_values, "baseline")
    pf = [k for k, v in per.items() if v["fired"]]
    if anchor_values is None:
        return {"per_arm": per, "per_arm_fired": pf, "cumulative_fired": [],
                "fired": bool(pf), "cumulative": None,
                "anchor_is_self_anchored": self_anchored,
                "cumulative_check_vacuous": None,
                "cumulative_note": NO_ANCHOR_NOTE}
    cum = _collapse_side(t_values, anchor_values, "anchor")
    vacuous = all(abs(anchor_values[c] - base_values[c])
                  <= 1e-12 * max(1.0, abs(base_values[c])) for c in COLLAPSE)
    note = ("VACUOUS BY CONSTRUCTION on this arm: the anchor IS this arm's own floor mean, "
            "so the cumulative delta equals the per-arm delta exactly and the cumulative "
            "half decides nothing. This is NOT a cumulative check that was made and "
            "passed. It becomes a real check for the SECOND Tier 3 arm on this scene."
            if vacuous else
            "The anchor differs from this arm's floor mean, so the cumulative half is a "
            "real check.")
    decomposition = {
        c: {"everything_but_the_gate": collapse_delta(c, base_values[c],
                                                      anchor_values[c]),
            "the_gate": collapse_delta(c, t_values[c], base_values[c])}
        for c in COLLAPSE}
    cf = [k for k, v in cum.items() if v["fired"]]
    return {"per_arm": per, "cumulative": cum, "per_arm_fired": pf,
            "cumulative_fired": cf, "fired": bool(pf or cf),
            "anchor_is_self_anchored": self_anchored,
            "cumulative_check_vacuous": vacuous, "cumulative_note": note,
            # Pre-registration section 5: a frozen anchor may differ from the arm being
            # graded in ways that are NOT the treatment (the loss chain, --export-every,
            # the MACHINE), so a cumulative firing must never be read as a treatment
            # effect without this decomposition.
            "decomposition": decomposition,
            "decomposition_note":
                "(base - anchor) is everything-but-the-treatment; (treatment - base) is "
                "the treatment. They sum to the cumulative delta in the column's own "
                "space. If the cumulative check fires while the per-arm one does not, "
                "what is in question is the anchor's applicability, not the treatment."}


def band2(verdicts: dict, *, lever: str = "geometry", gate: tuple = BAND2_GATE) -> str:
    """Band 2 -- GEOMETRY GATE, in one of two forms.

    lever="geometry" (the original, `3cfd8f3`)
        PASS          on-seed@1cm IMPROVED beyond floor, thin-axis not WORSENED
        FAIL          either column WORSENED beyond floor
        WITHIN FLOOR  neither worsened, but on-seed did not rise either

    lever="photometric" (AMENDMENT 1, operator, 2026-09-04, pre-registered in
    commit 99b0c92 BEFORE any Task 22 arm existed)
        PASS          neither column WORSENED beyond floor
        FAIL          either column WORSENED beyond floor

    WHY THE FORM INVERTS FOR A PHOTOMETRIC LEVER. Band 2 exists to confirm that a
    GEOMETRY lever actually helps geometry; it was derived on plane-aux and
    metric-space, both of which move splat positions. An appearance model has no
    mechanism by which on-seed should RISE, so requiring it to rise is a CATEGORY
    ERROR rather than a bar -- it returns DROP for every possible photometric
    result, including a perfect one. The do-no-harm form preserves what the band
    is FOR: catching a lever that buys its headline metric by damaging the
    reconstruction. That failure mode is entirely live for a bilateral grid -- one
    that absorbs error the geometry should have fixed shows up as on-seed FALLING
    while LPIPS improves, and the inverted form still catches it. The amendment
    NARROWS the band's scope; it does not weaken its purpose.

    `gate` names WHICH TWO COLUMNS. It defaults to `BAND2_GATE`, whose thin-axis half is
    splatstats' GATED column, because a caller with only one splatstats run has nothing
    else to offer. A caller that scored the arm twice passes `BAND2_GATE_UNGATED` -- see
    that constant for why a delta between gated columns is a composition statistic. The
    parameter exists rather than a second function so the truth table above is shared.

    Aspect and needles are deliberately NOT read here -- moving them to Band 1,
    where a 2.5% move and a 78% collapse get different answers, IS `3cfd8f3`.
    """
    if lever not in ("geometry", "photometric"):
        raise ValueError(f"lever must be 'geometry' or 'photometric', got {lever!r}")
    missing = [k for k in gate if verdicts.get(k) is None]
    if missing:
        raise ValueError(f"Band 2 columns missing: {missing}. An absent gate column "
                         f"must never read as a pass.")
    on_seed, thin = (verdicts[k] for k in gate)
    if on_seed == "WORSENED" or thin == "WORSENED":
        return "FAIL"
    if lever == "photometric":
        return "PASS"
    return "PASS" if on_seed == "IMPROVED" else "WITHIN FLOOR"


def band3(psnr_treatment: float, psnr_baseline: float,
          scene_psnr_floor: float | None = None) -> dict:
    """Hard DROP on a PSNR LOSS greater than 0.25 dB, or on falling below the
    24 dB Stage 4 gate from at or above it.

    ONE-SIDED by construction: the rule says "falls by". A gain is not a
    regression, and the older two-sided "must be WITHIN floor" reading is what
    made every Tier 3 arm unable to PASS whatever its geometry did. Both
    comparisons are strict.

    AMENDMENT 2 (`7c738b8`, recorded 2026-09-05 before the arm it affects existed).
    `scene_psnr_floor` is the scene's OWN n>=3 masked-PSNR floor. When it is at or above
    the 0.25 dB threshold, this band returns **INDETERMINATE** -- neither FIRED nor PASS:

        Band 3 returns INDETERMINATE on any scene whose own n>=3 masked-PSNR floor is
        >= the 0.25 dB threshold. On such a scene Band 3 contributes nothing to the
        verdict, and the outcome is decided by Band 1, Band 2 and the task's primary
        evidence alone. The verdict must state INDETERMINATE explicitly, with the
        scene's floor beside the threshold, and must not be reported as a pass.

    WHY. 3cfd8f3 derived 0.25 dB as a PRODUCT-VISIBILITY bar standing ABOVE the noise --
    "a loss smaller than the pipeline's own reproduction spread cannot be a product-
    visible regression". Where the scene's reproduction spread IS the bar, that premise
    fails and the conclusion does not follow: the band would hard-DROP a treatment for a
    movement two identical runs also produce. Measured on P-MASK, whose same-seed repeat
    pair F0/F1 differ by 0.2475 dB on masked PSNR.

    IT INTRODUCES NO NEW CONSTANT. The trigger compares two quantities the protocol
    already measures. `max(0.25, k x floor)` was considered and REJECTED: `k` would be a
    number chosen after seeing 0.2475.

    `scene_psnr_floor=None` means the floor was not supplied, and the band is live -- the
    pre-amendment behaviour, which is what Task 22's already-scored P-GEOM arm was graded
    under and what this deliberately does not disturb. `status` is always one of
    FIRED / PASS / INDETERMINATE, so no reader has to infer the third state from `fired`
    being False.
    """
    loss = psnr_baseline - psnr_treatment
    crossed = psnr_baseline >= STAGE4_PSNR_DB > psnr_treatment
    indeterminate = (scene_psnr_floor is not None
                     and scene_psnr_floor >= PSNR_DROP_DB)
    fired = bool(loss > PSNR_DROP_DB or crossed)
    out = {"baseline": psnr_baseline, "treatment": psnr_treatment, "loss_db": loss,
           "allowance_db": PSNR_DROP_DB, "exceeds_allowance": loss > PSNR_DROP_DB,
           "crossed_stage4_gate": crossed,
           "baseline_above_stage4": psnr_baseline >= STAGE4_PSNR_DB,
           "scene_psnr_floor_n3": scene_psnr_floor,
           "indeterminate": indeterminate,
           "fired": False if indeterminate else fired}
    out["status"] = ("INDETERMINATE" if indeterminate
                     else ("FIRED" if fired else "PASS"))
    if indeterminate:
        out["indeterminate_note"] = (
            f"AMENDMENT 2 (7c738b8): this scene's own n>=3 masked-PSNR floor is "
            f"{scene_psnr_floor:.6g} dB, at or above the {PSNR_DROP_DB} dB threshold, so "
            f"the threshold sits inside the scene's own reproduction noise and has no "
            f"discriminating power. Band 3 contributes NOTHING to this scene's verdict. "
            f"This is NOT a pass: it would have {'FIRED' if fired else 'passed'} had it "
            f"been evaluated, and that is recorded rather than acted on.")
        out["would_have_fired"] = fired
    return out


def drift_columns(rows: dict, verdicts: dict, band1_detail: dict,
                  band2_verdict=None, band3_fired: bool = False,
                  gate: tuple = BAND2_GATE, scope: tuple = DRIFT_SCOPE) -> list:
    """Beyond floor, below Band 1, and WORSE. Reported with sign and x floor; never a DROP.

    Two exclusions carry the definition:
      * IMPROVEMENTS are not drift. Band 2 REQUIRES on-seed to improve beyond its floor,
        so counting any beyond-floor move would make KEEP AS DEFAULT unreachable by
        construction -- and a rule with an unreachable branch is a broken rule.
      * A column that FIRED Band 1 is a COLLAPSE, not a drift. Reporting it as drift would
        make a hard DROP read as adoptable-with-caveats.
    """
    fired = set(band1_detail["per_arm_fired"]) | set(band1_detail["cumulative_fired"])
    out = []
    for k in scope:
        if k in fired or k not in rows or k not in verdicts:
            continue
        d = rows[k]["delta"]
        if DIRECTION.get(k, 0) == 0:
            worse = verdicts[k] == "MOVED" and d < 0      # two-sided: only a FALL is bad
        else:
            worse = verdicts[k] == "WORSENED"
        if not worse:
            continue
        fl = rows[k]["floor_spread_n3"]
        # A Band 2 column that WORSENED is why the scene failed, not a harmless drift; it
        # satisfies the literal definition, so it is reported rather than hidden -- but
        # flagged, because a list whose entries mean "adoptable with caveats" and "this is
        # the DROP" at once is precisely the check-shape CLAUDE.md warns about.
        caused_fail = bool(band2_verdict == "FAIL" and k in gate
                           and verdicts.get(k) == "WORSENED")
        caused_b3 = bool(band3_fired and k == "run.psnr_masked")
        out.append({"metric": k, "delta": d, "floor_spread_n3": fl,
                    "x_floor": abs(d) / fl if fl else None, "sign": "worse",
                    "caused_band2_fail": caused_fail, "caused_band3_fire": caused_b3})
    return out
