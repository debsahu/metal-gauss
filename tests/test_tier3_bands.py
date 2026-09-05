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
