"""The three-band rule, and the two thresholds that can be RE-DERIVED rather than
merely asserted.

A test that only restates a constant cannot fail for the reason you care about --
someone edits the constant and the test with it. Two of the four Band-1
thresholds have their inputs published in research/metal-gauss.md s13.6, so those
are recomputed from the arm values here. The other two are constants with a
stated provenance, and this file says so rather than pretending otherwise.
"""
from __future__ import annotations

import math

import pytest

from bench.tier3_bands import (
    COLLAPSE, DIRECTION, PSNR_DROP_DB, STAGE4_PSNR_DB,
    band1, band2, band3, collapse_delta, verdict_for,
)


def test_needle_threshold_is_recomputed_from_the_published_arm_values():
    """research/metal-gauss.md s13.6: adopted (R1p - B0a) +2.8962 pp, collapse
    (VOID - B0a) +40.1558 pp, sqrt = 10.78 pp, published +10.8 pp. Fails if the
    constant is edited without the derivation moving with it."""
    assert math.isclose(math.sqrt(2.8962 * 40.1558) / 100.0,
                        COLLAPSE["run.needle_frac"]["threshold"], rel_tol=3e-3)


def test_aspect_threshold_is_recomputed_from_the_published_arm_values():
    """s13.6: adopted (R1 - B0a) dlog -0.07974, collapse dlog -1.50112,
    -sqrt|.| = -0.3460, published -0.346. Note the anchor arm DIFFERS from
    needle's (R1 here, R1p there) and that is intended, not a slip: a uniform
    anchor is UNDEFINED for on-seed, which R1 improved."""
    assert math.isclose(math.sqrt(0.07974 * 1.50112),
                        COLLAPSE["run.aspect_p50"]["threshold"], rel_tol=3e-3)


def test_the_two_UNDERIVED_thresholds_are_declared_as_such():
    """Provenance, asserted. s13.6 publishes on-seed's and LPIPS's thresholds but
    not the arm values behind them, so they cannot be recomputed here. Pinning
    them anyway means a silent edit is caught; claiming they were derived would
    be a false provenance."""
    assert COLLAPSE["stats.on_seed_frac_1cm"]["threshold"] == 0.185
    assert COLLAPSE["run.lpips"]["threshold"] == 0.017


def test_collapse_delta_is_POSITIVE_FOR_WORSE_in_every_column():
    """A sign error inverts every Band 1 test: an arm that HALVED on-seed would
    read as a large improvement and no collapse could ever fire. Each column is
    given a value that is unambiguously worse and must return positive."""
    assert collapse_delta("run.needle_frac", 0.30, 0.16) > 0        # more needles
    assert collapse_delta("run.aspect_p50", 0.07, 0.30) > 0         # aspect collapsed
    assert collapse_delta("stats.on_seed_frac_1cm", 0.04, 0.09) > 0  # on-seed halved
    assert collapse_delta("run.lpips", 0.45, 0.39) > 0              # LPIPS rose
    # ...and unambiguously BETTER values must return negative, or the test above
    # is satisfied by a function that returns positive unconditionally.
    assert collapse_delta("run.needle_frac", 0.10, 0.16) < 0
    assert collapse_delta("run.aspect_p50", 0.40, 0.30) < 0
    assert collapse_delta("stats.on_seed_frac_1cm", 0.12, 0.09) < 0
    assert collapse_delta("run.lpips", 0.35, 0.39) < 0


def test_band1_comparison_is_strict_at_the_threshold():
    """"Exactly at the threshold has not fired." Fails on `>=`."""
    # BASE 0.0 ON PURPOSE. The obvious fixture -- base 0.16, treatment 0.16+0.108
    # -- CANNOT express "exactly at the threshold" in binary floating point: the
    # subtraction returns 0.10800000000000001, which is strictly greater, and the
    # test failed on a correct implementation. Differencing from zero is exact.
    base = {"run.needle_frac": 0.0, "run.aspect_p50": 0.30,
            "stats.on_seed_frac_1cm": 0.09, "run.lpips": 0.39}
    exact = dict(base, **{"run.needle_frac": 0.108})
    assert band1(exact, base)["per_arm"]["run.needle_frac"]["delta"] == 0.108
    assert not band1(exact, base)["fired"]
    over = dict(base, **{"run.needle_frac": 0.108 + 1e-9})
    assert band1(over, base)["per_arm_fired"] == ["run.needle_frac"]


def test_band1_refuses_a_column_it_never_measured():
    """An absent collapse column must never read as "did not collapse"."""
    base = {"run.needle_frac": 0.16, "run.aspect_p50": 0.30,
            "stats.on_seed_frac_1cm": 0.09, "run.lpips": 0.39}
    with pytest.raises(ValueError, match="never measured|missing"):
        band1({k: v for k, v in base.items() if k != "run.lpips"}, base)


def test_band1_does_not_claim_a_cumulative_check_it_did_not_make():
    """The cumulative half needs a frozen anchor from a PREVIOUS Tier 3 arm. With
    none, it must be reported ABSENT rather than passed -- a tautology reported
    as evidence is the failure this project keeps repeating."""
    base = {"run.needle_frac": 0.16, "run.aspect_p50": 0.30,
            "stats.on_seed_frac_1cm": 0.09, "run.lpips": 0.39}
    b = band1(base, base)
    assert b["cumulative"] is None
    assert "Absent, not passed" in b["cumulative_note"]


@pytest.mark.parametrize("on_seed,thin,want", [
    ("IMPROVED", "WITHIN FLOOR", "PASS"),
    ("IMPROVED", "IMPROVED", "PASS"),
    ("WITHIN FLOOR", "WITHIN FLOOR", "WITHIN FLOOR"),
    ("WITHIN FLOOR", "WORSENED", "FAIL"),
    ("WORSENED", "IMPROVED", "FAIL"),
])
def test_band2_truth_table(on_seed, thin, want):
    """Band 2 REQUIRES on-seed to RISE; "not worse" is WITHIN FLOOR, not PASS.
    That distinction decides this task's verdict, so it is enumerated."""
    assert band2({"stats.on_seed_frac_1cm": on_seed,
                  "stats.thin_axis_angle_p50": thin}) == want


def test_band2_refuses_a_missing_gate_column():
    with pytest.raises(ValueError, match="missing"):
        band2({"stats.on_seed_frac_1cm": "IMPROVED"})


def test_band3_is_ONE_SIDED_and_a_gain_is_not_a_regression():
    """The older two-sided reading is what made every Tier 3 arm unable to PASS
    whatever its geometry did. Fails if a PSNR GAIN fires the band."""
    assert not band3(23.0, 22.0)["fired"]                 # +1.0 dB gain
    assert not band3(22.60 - 0.25, 22.60)["fired"]        # exactly the allowance
    assert band3(22.60 - 0.2501, 22.60)["fired"]
    # the Stage 4 crossing is independent of the 0.25 allowance
    assert band3(23.95, 24.05)["fired"], "crossing 24 dB must fire on a 0.10 dB loss"
    assert not band3(23.95, 23.99)["fired"], "below the gate already, small loss"


def test_verdict_for_respects_each_columns_direction():
    """Fails if a lower-is-better column is graded as higher-is-better -- which
    would make a WORSE thin-axis read as IMPROVED."""
    assert verdict_for("stats.thin_axis_angle_p50", -2.0, 0.5) == "IMPROVED"
    assert verdict_for("stats.thin_axis_angle_p50", +2.0, 0.5) == "WORSENED"
    assert verdict_for("stats.on_seed_frac_1cm", +0.02, 0.001) == "IMPROVED"
    assert verdict_for("stats.on_seed_frac_1cm", -0.02, 0.001) == "WORSENED"
    assert verdict_for("run.lpips", -0.02, 0.001) == "IMPROVED"
    assert verdict_for("run.psnr_masked", -2.0, 0.1) == "MOVED"   # two-sided
    assert verdict_for("stats.on_seed_frac_1cm", 0.0005, 0.001) == "WITHIN FLOOR"


def test_every_gated_column_has_a_direction():
    """Fails if a column is added to a gate without saying which way it runs --
    `verdict_for` would silently grade it two-sided and it could never WORSEN."""
    from bench.tier3_bands import BAND2_GATE, GEOMETRY_GATE
    for k in set(GEOMETRY_GATE) | set(BAND2_GATE) | set(COLLAPSE):
        assert DIRECTION.get(k) is not None, k
        if k in COLLAPSE:
            assert DIRECTION[k] != 0, f"{k} is a collapse column and cannot be two-sided"


# ------------------------- AMENDMENT 1 (operator, 2026-09-04, commit 99b0c92)

@pytest.mark.parametrize("on_seed,thin,geom,photo", [
    ("IMPROVED",     "WITHIN FLOOR", "PASS",         "PASS"),
    ("IMPROVED",     "IMPROVED",     "PASS",         "PASS"),
    ("WITHIN FLOOR", "WITHIN FLOOR", "WITHIN FLOOR", "PASS"),      # the ONLY change
    ("WITHIN FLOOR", "IMPROVED",     "WITHIN FLOOR", "PASS"),      # the ONLY change
    ("WITHIN FLOOR", "WORSENED",     "FAIL",         "FAIL"),
    ("WORSENED",     "IMPROVED",     "FAIL",         "FAIL"),
    ("WORSENED",     "WORSENED",     "FAIL",         "FAIL"),
])
def test_the_photometric_form_changes_EXACTLY_the_did_not_rise_cells(
        on_seed, thin, geom, photo):
    """AMENDMENT 1 inverts Band 2 for a photometric lever from "on-seed must
    IMPROVE" to "on-seed must NOT WORSEN". This enumerates the whole truth table
    under BOTH forms, so the amendment's blast radius is asserted rather than
    described: the only cells that move are the two where on-seed did not rise and
    nothing worsened.

    Fails if the amendment is implemented as "photometric always passes Band 2",
    which would delete the do-no-harm property that is the entire reason the
    inverted form is acceptable."""
    v = {"stats.on_seed_frac_1cm": on_seed, "stats.thin_axis_angle_p50": thin}
    assert band2(v) == geom
    assert band2(v, lever="photometric") == photo


def test_the_photometric_form_still_FAILS_on_damage_which_is_the_whole_point():
    """The failure mode the inverted band must still catch: a grid that buys its
    LPIPS by absorbing error the geometry should have fixed, which shows as
    on-seed FALLING. If this ever passes, the amendment has become "photometric
    levers skip Band 2" and the band is decorative."""
    damaged = {"stats.on_seed_frac_1cm": "WORSENED",
               "stats.thin_axis_angle_p50": "IMPROVED"}
    assert band2(damaged, lever="photometric") == "FAIL"
    # ...and the control: an undamaged geometry-neutral result must PASS, or the
    # assertion above would be satisfiable by a form that fails everything.
    neutral = {"stats.on_seed_frac_1cm": "WITHIN FLOOR",
               "stats.thin_axis_angle_p50": "WITHIN FLOOR"}
    assert band2(neutral, lever="photometric") == "PASS"


def test_an_unknown_lever_is_rejected_rather_than_defaulted():
    """A typo'd lever must not silently fall through to the geometry form, which
    would return DROP for a photometric arm and look like a real verdict."""
    v = {"stats.on_seed_frac_1cm": "IMPROVED", "stats.thin_axis_angle_p50": "IMPROVED"}
    with pytest.raises(ValueError, match="lever must be"):
        band2(v, lever="photometrics")


# ------------------- AMENDMENT 2 (Task 20, commit 7c738b8, 2026-09-05)

def test_band3_is_INDETERMINATE_where_its_threshold_sits_inside_the_scenes_own_floor():
    """AMENDMENT 2, pre-registered in `7c738b8` BEFORE the arm it affects existed.

    3cfd8f3 derived 0.25 dB as a PRODUCT-VISIBILITY bar standing ABOVE the noise. On
    P-MASK the same-seed repeat pair F0/F1 differ by 0.2475 dB, so on that scene the bar
    IS the noise and has no discriminating power: it would hard-DROP a treatment for a
    movement two identical runs also produce.

    CATCHES the three ways this could be implemented wrongly, each of which reads as a
    reasonable result:
      * INDETERMINATE reported as a PASS -- the rule says explicitly it is neither;
      * INDETERMINATE reported as FIRED, i.e. a DROP on a scene the band cannot judge;
      * the amendment applied on a scene whose floor is BELOW the threshold, which would
        silently retire Band 3 everywhere.
    """
    # THE TRIGGER IS THE n>=3 FLOOR, NOT THE n=2 REPEAT PAIR, and this test was wrong
    # about that on its first run -- worth recording, because the amendment's own
    # motivating number is the pair difference. P-MASK's F0/F1 differ by 0.2475 dB, which
    # is what made the amendment necessary, but 0.2475 < 0.25 so IT DOES NOT ITSELF
    # TRIGGER. The amendment says so explicitly ("the floor itself still requires n=3 and
    # is not quoted from two samples -- section 8.2") and argues the n=3 floor must reach
    # >= 0.2475 by monotonicity, not that it must reach 0.25. Asserted both ways here so
    # nobody re-derives the rule from the headline number.
    assert band3(24.6, 25.0, 0.2475)["status"] == "FIRED", (
        "the n=2 repeat-pair difference is not the n=3 floor and must not trigger")

    # A loss that WOULD have fired, on a scene whose own n=3 floor is >= 0.25 dB.
    b = band3(24.6, 25.0, 0.2600)
    assert b["status"] == "INDETERMINATE"
    assert b["fired"] is False, "an INDETERMINATE band must not DROP the scene"
    assert b["indeterminate"] is True
    assert b["would_have_fired"] is True, "what it would have said is recorded, not acted on"
    assert b["scene_psnr_floor_n3"] == 0.2600
    # "with the scene's floor beside the threshold" -- the rule's own words.
    assert "0.26" in b["indeterminate_note"] and str(PSNR_DROP_DB) in b["indeterminate_note"]

    # The SAME loss on a scene whose floor is below the threshold is untouched: P-GEOM's
    # floor is 0.142204, and the amendment's own Scope section says its verdict stands.
    live = band3(24.6, 25.0, 0.142204)
    assert live["status"] == "FIRED" and live["fired"] is True
    assert live["indeterminate"] is False

    # No floor supplied = the pre-amendment behaviour, which is what Task 22's already
    # scored P-GEOM arm was graded under.
    assert band3(24.6, 25.0)["status"] == "FIRED"
    assert band3(24.9, 25.0)["status"] == "PASS"


def test_the_INDETERMINATE_trigger_is_at_or_above_the_threshold_and_introduces_no_constant():
    """The rule is `floor >= 0.25`, not `>`, and not `k x floor`. `max(0.25, k x floor)`
    was CONSIDERED AND REJECTED in the amendment: `k` would be a number chosen after
    seeing 0.2475, which is the retuning this project forbids. So the only constant in
    play is the 0.25 that was already there -- asserted here so a later `k` cannot be
    slipped in without this failing."""
    assert band3(24.6, 25.0, PSNR_DROP_DB)["status"] == "INDETERMINATE"     # exactly at
    assert band3(24.6, 25.0, PSNR_DROP_DB - 1e-12)["status"] == "FIRED"     # just below
    # A scene whose floor is huge does not become a DROP or a PASS, whatever the loss.
    assert band3(30.0, 25.0, 1.0)["status"] == "INDETERMINATE"   # a large GAIN
    assert band3(10.0, 25.0, 1.0)["status"] == "INDETERMINATE"   # a catastrophic loss


def test_a_scene_that_is_INDETERMINATE_is_never_silently_a_pass():
    """The amendment's operative sentence: "must not be reported as a pass". A reader --
    or a downstream harness -- that only looks at `fired` sees False, which is what a PASS
    also looks like. `status` is the field that separates them, so it must exist on EVERY
    return and take exactly one of three values."""
    for args in ((24.6, 25.0), (24.6, 25.0, 0.10), (24.6, 25.0, 0.30),
                 (25.5, 25.0), (23.95, 24.05, 0.30)):
        out = band3(*args)
        assert out["status"] in ("FIRED", "PASS", "INDETERMINATE")
        assert (out["status"] == "FIRED") == out["fired"]


# --------------------------- the two thin-axis columns are not interchangeable

def test_the_UNGATED_thin_axis_column_is_a_DIFFERENT_column_with_its_own_direction():
    """splatstats' default gate admits only splats within 5 cm of the seed, so two arms
    can be scored over different POPULATIONS and a delta between their gated thin-axis
    medians is a COMPOSITION statistic wearing an orientation statistic's name. Task 22
    measured that on 2026-09-05: Task 19's -2.35 deg reversed to +0.78 deg WORSE at equal
    population, tracking admitted-population excess at r = -0.995.

    CATCHES the two columns being conflated -- same name, or a missing direction, either
    of which would make the ungated column grade two-sided and never able to WORSEN."""
    from bench.tier3_bands import (BAND2_GATE_UNGATED, GEOMETRY_GATE_UNGATED,
                                   THIN_AXIS_GATED, THIN_AXIS_UNGATED)
    assert THIN_AXIS_GATED != THIN_AXIS_UNGATED
    assert DIRECTION[THIN_AXIS_UNGATED] == DIRECTION[THIN_AXIS_GATED] == -1
    assert BAND2_GATE_UNGATED == ("stats.on_seed_frac_1cm", THIN_AXIS_UNGATED)
    assert set(GEOMETRY_GATE_UNGATED) - set(BAND2_GATE_UNGATED) == {"run.aspect_p50",
                                                                    "run.needle_frac"}
    # Every column of BOTH gates has a direction, for the reason above.
    for k in set(BAND2_GATE_UNGATED) | set(GEOMETRY_GATE_UNGATED):
        assert DIRECTION.get(k) is not None, k


def test_band2_reads_whichever_thin_axis_column_it_is_GIVEN_and_refuses_the_other(
):
    """CATCHES the gate parameter being accepted and ignored -- an argument sink. The two
    columns carry different verdicts here, so a band2 that read the hard-coded default
    would return PASS where the ungated column says FAIL."""
    from bench.tier3_bands import BAND2_GATE_UNGATED, THIN_AXIS_GATED, THIN_AXIS_UNGATED
    v = {"stats.on_seed_frac_1cm": "IMPROVED",
         THIN_AXIS_GATED: "IMPROVED", THIN_AXIS_UNGATED: "WORSENED"}
    assert band2(v) == "PASS"                                   # the gated column
    assert band2(v, gate=BAND2_GATE_UNGATED) == "FAIL"          # the ungated one
    # ...and an arm scored only once cannot be graded on the ungated gate by accident.
    with pytest.raises(ValueError, match="thin_axis_angle_p50_ungated"):
        band2({"stats.on_seed_frac_1cm": "IMPROVED", THIN_AXIS_GATED: "IMPROVED"},
              gate=BAND2_GATE_UNGATED)


def test_hard_needle_frac_is_REPORTED_and_appears_in_NO_BAND_of_the_unified_rule():
    """`run.hard_needle_frac` is a DELIVERY statement -- the fraction of splats whose
    orientation splat-transform's 8-bit smallest-three quaternion cannot represent -- and
    it is the WEAKEST of the four candidate collapse columns on the only scale a collapse
    test cares about: adopted-vs-collapse log separation 5.0x, against aspect's 18.8x
    (research/metal-gauss.md s13.6; derived at length in
    tests/test_plane_aux_tier3_rule.py::
    test_the_hard_needle_column_is_a_DELIVERY_STATEMENT_not_a_collapse_discriminator).

    THAT DERIVATION IS PINNED AGAINST `scripts/plane_aux_arms.py` AND NOTHING PINNED IT
    HERE. `bench/tier3_bands.py` is now the rule Task 20 and Task 22 both grade with, so a
    promotion of this column to a gate would have gone uncaught in the one module that
    matters. `DRIFT_SCOPE` is included deliberately: drift is reported, not decided on, but
    a column in it is one edit from a band.
    """
    from bench.tier3_bands import (BAND2_GATE_UNGATED, DRIFT_SCOPE,
                                   GEOMETRY_GATE_UNGATED)
    from bench.tier3_bands import BAND2_GATE, COLLAPSE, GEOMETRY_GATE
    for where, names in (("COLLAPSE", COLLAPSE), ("GEOMETRY_GATE", GEOMETRY_GATE),
                         ("BAND2_GATE", BAND2_GATE), ("DRIFT_SCOPE", DRIFT_SCOPE),
                         ("GEOMETRY_GATE_UNGATED", GEOMETRY_GATE_UNGATED),
                         ("BAND2_GATE_UNGATED", BAND2_GATE_UNGATED)):
        assert "run.hard_needle_frac" not in set(names), where
    # The control: the columns that ARE gates are present, or the loop above is satisfied
    # by a module whose gates are all empty.
    assert "run.needle_frac" in COLLAPSE and "run.aspect_p50" in COLLAPSE
