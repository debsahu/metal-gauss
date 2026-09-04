"""The REPLACEMENT Tier 3 keep/drop rule, unit-tested.

The rule this replaces was magnitude-blind: any beyond-floor worsening of any of four
geometry columns was a DROP, so a 2.5% relative aspect move (4.6x a 0.0017 floor) and the
VOID row's 78% collapse were the same verdict. research/metal-gauss.md section 12.4 records
that as "a real question ... it must be settled BEFORE the next arm, never after seeing
this one".

The replacement is three bands. These tests need no GPU and no artifacts, and every
threshold in the rule is re-derived here from the section 8.1 table rather than trusted as
a constant.
"""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from plane_aux_arms import (  # noqa: E402
    COLLAPSE, HARD_NEEDLE_ASPECT, band1, band2, band3, collapse_delta,
    check_anchor_applies, drift_columns, grade, combined_verdict, load_anchor,
)

ON_SEED = "stats.on_seed_frac_1cm"
THIN = "stats.thin_axis_angle_p50"
ASPECT = "run.aspect_p50"
NEEDLE = "run.needle_frac"
PSNR = "run.psnr_masked"
LPIPS = "run.lpips"

# THE TIER 1 ARMS, AT FULL PRECISION, FROM THEIR OWN ARTIFACTS -- not from
# research/metal-gauss.md section 8.1's 3-4 sf table, which is a restatement of them.
# `bench/results/plane_aux/tier1_void_row.json` carries them into the repository because
# the run directory is session scratchpad and will not survive it.
TIER1 = json.loads((Path(__file__).resolve().parents[1] /
                    "bench/results/plane_aux/tier1_void_row.json").read_text())["arms"]
B0A = TIER1["B0a"]["values"]
R1 = TIER1["R1"]["values"]
R1P = TIER1["R1p"]["values"]
VOID = TIER1["R1_openweights"]["values"]      # the pre-fix open-weight-path arm


# --------------------------------------------------------------- threshold provenance

def test_every_collapse_threshold_is_REDERIVED_from_the_section_8_1_table():
    """The four Band-1 thresholds are `sqrt(healthy x collapse)` in each column's natural
    space, anchored on the largest ADOPTED move (R1 or R1p vs B0a, whichever moved further
    in the WORSENING direction) and on the VOID row (the smallest recorded collapse).

    Would catch a threshold typed in wrong, or one silently retuned later to make an arm
    pass. A constant with no derivation behind it is exactly what a future session would
    nudge; this makes nudging it fail a test that names the source table.

    Note which arm anchors which column: aspect is R1 (0.2730 is further from 0.2957 than
    R1p's 0.2749), the other three are R1p. The brief called all four "R1p-vs-B0a"; the
    rule is "largest adopted worsening", and for aspect that is R1.
    """
    for col, spec in COLLAPSE.items():
        healthy = max(collapse_delta(col, arm[col], B0A[col]) for arm in (R1, R1P))
        collapse = collapse_delta(col, VOID[col], B0A[col])
        assert healthy > 0, f"{col}: no adopted arm worsened it; the anchor is undefined"
        assert collapse > healthy, f"{col}: the VOID row must be worse than any adopted arm"
        derived = math.sqrt(healthy * collapse)
        # The pre-registered constants are 2-3 significant figures; the artifacts derive
        # 0.107842 / 0.345971 / 0.184386 / 0.017349. Agreement is 0.15%-2.05%, i.e. the
        # constants ARE these numbers rounded. The bar is 3%: tight enough that a typo or a
        # retune fails, loose enough that 2 sf rounding does not.
        assert abs(derived - spec["threshold"]) <= 0.03 * spec["threshold"], (
            f"{col}: the rule uses {spec['threshold']}, the Tier 1 artifacts derive "
            f"{derived:.6f}")


def test_the_derivation_uses_the_LARGEST_ADOPTED_move_per_column_and_they_DIFFER():
    """Which adopted arm anchors which column is not uniform, and asserting that keeps the
    rule honest about it: aspect is anchored on R1, the other three on R1p. The brief
    described all four as "R1p-vs-B0a".

    Would catch a derivation hard-wired to one arm, which changes the aspect threshold by
    4% and silently makes the rule 4% looser on the column Task 19 actually moves.
    """
    anchors = {}
    for col in COLLAPSE:
        anchors[col] = max(("R1", "R1p"),
                           key=lambda a: collapse_delta(col, TIER1[a]["values"][col],
                                                        B0A[col]))
    assert anchors[ASPECT] == "R1"
    assert {anchors[c] for c in (NEEDLE, ON_SEED, LPIPS)} == {"R1p"}


def test_the_VOID_row_is_the_pre_fix_arm_and_NOT_one_of_the_adopted_ones():
    """Provenance, asserted rather than assumed. The upper anchor must be the arm section
    8.1 struck through -- a different `depth_normal_weight`, a different commit -- and not,
    say, R1 mislabelled. Would catch the anchor file being rebuilt from the wrong arm."""
    v, r1 = TIER1["R1_openweights"], TIER1["R1"]
    assert v["git"] != r1["git"], "the VOID row is a DIFFERENT binary, pre aux-weight fix"
    assert v["values"][ON_SEED] < 0.55 * B0A[ON_SEED], "on-seed HALVED is the signature"
    assert v["values"][NEEDLE] > 3 * B0A[NEEDLE]


def test_collapse_delta_is_SIGNED_toward_worse_and_uses_the_declared_space():
    """`collapse_delta` returns a POSITIVE number when the column got worse, in log space
    for the ratio columns and absolute for the rest.

    Would catch a sign flip, which would invert every Band-1 test: an arm that HALVED
    on-seed would read as a large improvement and never fire.
    """
    assert collapse_delta(NEEDLE, 0.30, 0.20) == pytest.approx(+0.10)   # needles up = worse
    assert collapse_delta(NEEDLE, 0.10, 0.20) == pytest.approx(-0.10)
    assert collapse_delta(LPIPS, 0.42, 0.40) == pytest.approx(+0.02)    # LPIPS up = worse
    # aspect and on-seed are log-space and DOWN is worse
    assert collapse_delta(ASPECT, 0.10, 0.20) == pytest.approx(math.log(2.0))
    assert collapse_delta(ASPECT, 0.40, 0.20) == pytest.approx(-math.log(2.0))
    assert collapse_delta(ON_SEED, 0.04, 0.08) == pytest.approx(math.log(2.0))


def test_the_log_thresholds_mean_what_the_table_says_they_mean():
    """-0.346 is -29% and -0.185 is -17%. Would catch a threshold applied as a RATIO
    (0.346) where the rule means a LOG ratio, which is a 4x difference in strictness."""
    assert math.exp(-COLLAPSE[ASPECT]["threshold"]) == pytest.approx(0.71, abs=0.005)
    assert math.exp(-COLLAPSE[ON_SEED]["threshold"]) == pytest.approx(0.83, abs=0.005)


# ------------------------------------------------------------------- Band 1: collapse

def _anchor(**over):
    a = {NEEDLE: 0.1516, ASPECT: 0.3126, ON_SEED: 0.0719, LPIPS: 0.4012}
    a.update(over)
    return a


def _base(**over):
    b = {NEEDLE: 0.1516, ASPECT: 0.3126, ON_SEED: 0.0719, LPIPS: 0.4012, PSNR: 22.57,
         THIN: 30.35}
    b.update(over)
    return b


def test_the_hard_needle_column_SEES_the_VOID_row_far_more_loudly_than_needle_frac():
    """`frac(aspect < 0.01)` on the archived plys. It is REPORTED, not gated -- but if it
    could not separate VOID from the adopted recipe there would be no case for reporting it.

    Measured: baseline 0.700%, R1p 1.548%, VOID 31.770%. VOID is 20.5x the adopted arm on
    this column against 2.9x on `needle_frac`, so it is a sharper instrument for exactly
    the pathology the rule exists to catch -- which is the claim, and it is testable.
    """
    hn = "run.hard_needle_frac"
    b, adopted, void = B0A[hn], R1P[hn], VOID[hn]
    assert void > 0.10, "the prediction under test was >= 10% on the VOID row"
    assert void / adopted > 3 * (VOID[NEEDLE] / R1P[NEEDLE]), \
        "hard needles must separate VOID from the adopted recipe MORE sharply than the " \
        "existing needle column, or the column adds nothing"
    assert adopted / b < 3.0, "and it must not scream on an arm we adopted"


def test_THE_VOID_ROW_fires_band1_on_FOUR_INDEPENDENT_COLUMNS():
    """THE ROW THE RULE EXISTS TO CATCH. research/metal-gauss.md section 8.1: the pre-fix
    recipe reached a similar thin-axis by destroying the splats, and thin-axis, opacity and
    dark fraction all called it HEALTHIER than baseline.

    A replacement rule that lets VOID through is not a loosening, it is a hole. Four
    columns is the assertion, not one: the rule must not depend on any single column
    surviving a future schema change.
    """
    d = band1(VOID, B0A, B0A)
    assert d["fired"], "the VOID row must be a hard DROP"
    assert set(d["per_arm_fired"]) == {NEEDLE, ASPECT, ON_SEED, LPIPS}
    for col in (NEEDLE, ASPECT, ON_SEED, LPIPS):
        assert d["per_arm"][col]["x_threshold"] > 3.0, col
    # and the separation that makes the threshold a fence rather than a line: every ADOPTED
    # arm sits below 0.3x on every column, so nothing about this is close.
    for arm in ("R1", "R1p"):
        a = band1(TIER1[arm]["values"], B0A, B0A)
        assert not a["fired"], arm
        assert max(v["x_threshold"] for v in a["per_arm"].values()) < 0.3, arm


def test_task19_sized_moves_do_NOT_fire_band1():
    """The whole point of the replacement. P-GEOM's P0 moved aspect -2.5% and needles
    +0.6 pp -- 4.6x and 4.5x their floors, and DROPs under the old rule. Band 1 must not
    fire on them, or the replacement changes nothing.

    Would catch thresholds set at the floor rather than between healthy and collapse.
    """
    t = {NEEDLE: 0.15738, ASPECT: 0.30462, ON_SEED: 0.09788, LPIPS: 0.40086}
    d = band1(t, _base(), _anchor())
    assert not d["fired"]
    assert d["per_arm"][ASPECT]["x_threshold"] < 0.1


def test_band1_fires_on_ANY_ONE_column_alone():
    """Would catch an `all()` where the rule says `any one column`."""
    for col, bad in ((NEEDLE, 0.1516 + 0.11), (ASPECT, 0.3126 * math.exp(-0.35)),
                     (ON_SEED, 0.0719 * math.exp(-0.19)), (LPIPS, 0.4012 + 0.018)):
        t = {NEEDLE: 0.1516, ASPECT: 0.3126, ON_SEED: 0.0719, LPIPS: 0.4012}
        t[col] = bad
        d = band1(t, _base(), _anchor())
        assert d["fired"] and d["per_arm_fired"] == [col], col


def test_the_band1_comparison_is_STRICT_at_the_threshold():
    """A delta exactly equal to the threshold has NOT fired. Would catch `>=`.

    THIS TEST DID NOT BIND WHEN FIRST WRITTEN, and the `>=` mutant survived it. It built
    the boundary as `0.1516 + 0.108` and differenced against `0.1516`, which in binary
    floating point is 0.10799999999999998 -- strictly BELOW the threshold, so both `>` and
    `>=` answered "not fired" and the test passed for the wrong reason. That is the exact
    failure mode CLAUDE.md names: a rule tested only where it cannot bind.

    Fixed by referencing 0.0, where `thr - 0.0 == thr` is exact -- and by ASSERTING the
    exactness, so a future edit that reintroduces rounding fails here instead of silently
    going slack again.
    """
    thr = COLLAPSE[NEEDLE]["threshold"]
    base, anchor = _base(**{NEEDLE: 0.0}), _anchor(**{NEEDLE: 0.0})
    exact = {NEEDLE: thr, ASPECT: 0.3126, ON_SEED: 0.0719, LPIPS: 0.4012}
    d = band1(exact, base, anchor)
    assert d["per_arm"][NEEDLE]["delta"] == thr, \
        "the fixture must sit EXACTLY on the threshold or it cannot test strictness"
    assert not d["fired"]
    over = dict(exact); over[NEEDLE] = math.nextafter(thr, 1.0)
    assert over[NEEDLE] > thr
    assert band1(over, base, anchor)["fired"]


def test_CUMULATIVE_band1_catches_a_RATCHET_that_no_single_arm_fires_on():
    """THE REASON THE FROZEN ANCHOR EXISTS. Four accepted 8 pp needle drifts are a 32 pp
    collapse, and each one is inside the 10.8 pp per-arm threshold against ITS OWN
    re-measured base.

    This is the test the per-arm check cannot pass on its own: the treatment sits 8 pp
    above a base that has itself already drifted 24 pp from the frozen anchor. Would catch
    an implementation that grades cumulative against the floor mean -- i.e. that computes
    the per-arm number twice and calls one of them cumulative.
    """
    anchor = _anchor()
    drifted_base = _base(**{NEEDLE: anchor[NEEDLE] + 0.24})
    t = {NEEDLE: drifted_base[NEEDLE] + 0.08, ASPECT: 0.3126, ON_SEED: 0.0719,
         LPIPS: 0.4012}
    d = band1(t, drifted_base, anchor)
    assert d["per_arm_fired"] == [], "the per-arm step is inside its threshold"
    assert d["cumulative_fired"] == [NEEDLE]
    assert d["fired"], "cumulative alone must be enough to DROP"


def test_cumulative_and_per_arm_AGREE_when_the_anchor_IS_the_base():
    """The degenerate case that holds for Task 19 itself -- its floors ARE the anchor -- so
    a difference between the two columns there would be an implementation artefact."""
    t = {NEEDLE: 0.15738, ASPECT: 0.30462, ON_SEED: 0.09788, LPIPS: 0.40086}
    d = band1(t, _base(), _anchor())
    for col in COLLAPSE:
        assert d["per_arm"][col]["delta"] == pytest.approx(d["cumulative"][col]["delta"])


def test_a_band1_column_MISSING_from_the_treatment_is_an_ERROR_not_a_pass():
    """THE FAILURE THIS PROJECT KEEPS REPEATING. A collapse column that was never measured
    must not read as "did not collapse"."""
    t = {NEEDLE: 0.1516, ASPECT: 0.3126, ON_SEED: 0.0719}      # no LPIPS
    with pytest.raises(SystemExit, match=LPIPS):
        band1(t, _base(), _anchor())


def test_a_band1_column_MISSING_from_the_ANCHOR_is_an_ERROR_not_a_pass():
    """An anchor that predates a column cannot testify about that column's ratchet."""
    t = {NEEDLE: 0.1516, ASPECT: 0.3126, ON_SEED: 0.0719, LPIPS: 0.4012}
    a = _anchor(); a.pop(LPIPS)
    with pytest.raises(SystemExit, match="anchor"):
        band1(t, _base(), a)


# --------------------------------------------------------------- Band 2: geometry gate

def test_band2_requires_on_seed_to_RISE_and_thin_axis_not_to_WORSEN():
    assert band2({ON_SEED: "IMPROVED", THIN: "IMPROVED"}) == "PASS"
    assert band2({ON_SEED: "IMPROVED", THIN: "WITHIN FLOOR"}) == "PASS"
    assert band2({ON_SEED: "IMPROVED", THIN: "WORSENED"}) == "FAIL"
    assert band2({ON_SEED: "WORSENED", THIN: "IMPROVED"}) == "FAIL"
    assert band2({ON_SEED: "WITHIN FLOOR", THIN: "IMPROVED"}) == "WITHIN FLOOR"
    assert band2({ON_SEED: "WITHIN FLOOR", THIN: "WITHIN FLOOR"}) == "WITHIN FLOOR"


def test_band2_does_NOT_read_aspect_or_needles():
    """The one substantive change to Band 2: the anti-collapse pair left the gate for
    Band 1, where magnitude decides. Would catch a copy of the old four-column gate."""
    assert band2({ON_SEED: "IMPROVED", THIN: "IMPROVED",
                  ASPECT: "WORSENED", NEEDLE: "WORSENED"}) == "PASS"


def test_a_missing_band2_verdict_is_an_ERROR():
    with pytest.raises(SystemExit, match=THIN):
        band2({ON_SEED: "IMPROVED"})


# ---------------------------------------------------------------- Band 3: photometric

def test_band3_fires_on_a_psnr_LOSS_larger_than_0_25_dB():
    """0.25 dB is above the trainer's own cross-machine same-seed spread (0.115-0.220 dB,
    section 8.2): a loss inside that spread cannot be a product-visible regression."""
    assert band3(22.57 - 0.26, 22.57)["fired"]
    assert not band3(22.57 - 0.24, 22.57)["fired"]
    assert not band3(22.57 - 0.25, 22.57)["fired"], "strict: exactly 0.25 has not fired"


def test_band3_does_NOT_fire_on_a_psnr_GAIN():
    """Would catch `abs(delta) > 0.25`, i.e. the old two-sided PSNR condition surviving
    into a band whose text says 'falls by'."""
    assert not band3(22.57 + 5.0, 22.57)["fired"]


def test_band3_fires_when_a_scene_that_was_ABOVE_24_dB_drops_below_it():
    """The Stage 4 delivery gate. A 0.10 dB loss is inside the 0.25 dB allowance and is
    still a DROP if it crosses 24 dB."""
    d = band3(23.99, 24.05)
    assert d["fired"] and d["crossed_stage4_gate"]
    assert not band3(23.99, 23.95)["fired"], "a scene already below 24 cannot cross it"
    assert not band3(24.00, 24.05)["fired"], "strict: exactly 24.0 is not below the gate"


def test_band3_on_the_real_arkit_numbers_does_not_fire():
    """arkit baseline 24.7602, P0 -0.103 dB -> 24.657, still above 24. Would catch a gate
    written against the BASELINE crossing rather than the treatment."""
    assert not band3(24.7602 - 0.103, 24.7602)["fired"]


# ------------------------------------------------------------------------------ drift

def test_drift_is_WORSENING_ONLY_or_KEEP_AS_DEFAULT_is_unreachable():
    """Band 2 REQUIRES on-seed to improve beyond its floor. If drift counted any
    beyond-floor move, every Band-2 pass would carry drift and the KEEP AS DEFAULT class
    could never be reached. Drift is worsening only.

    Would catch `moves == True` used as the drift predicate.
    """
    verdict = {ON_SEED: "IMPROVED", THIN: "IMPROVED", ASPECT: "IMPROVED",
               NEEDLE: "IMPROVED", LPIPS: "IMPROVED", PSNR: "MOVED"}
    rows = {k: {"delta": +1.0, "floor_spread_n3": 0.001} for k in verdict}
    rows[PSNR]["delta"] = +1.0                      # PSNR UP is not a worsening
    assert drift_columns(rows, verdict, {"per_arm_fired": [], "cumulative_fired": []}) == []


def test_a_worsened_column_below_the_collapse_threshold_is_DRIFT():
    verdict = {ON_SEED: "IMPROVED", THIN: "IMPROVED", ASPECT: "WORSENED",
               NEEDLE: "WORSENED", LPIPS: "WITHIN FLOOR", PSNR: "WITHIN FLOOR"}
    rows = {k: {"delta": -0.008, "floor_spread_n3": 0.0017} for k in verdict}
    d = drift_columns(rows, verdict, {"per_arm_fired": [], "cumulative_fired": []})
    assert {x["metric"] for x in d} == {ASPECT, NEEDLE}
    assert all(x["x_floor"] > 1.0 and "sign" in x for x in d)


def test_a_column_that_FIRED_band1_is_a_COLLAPSE_not_a_drift():
    """Drift is defined as 'beyond floor, BELOW Band 1'. A collapsed column reported as
    drift would make a DROP look adoptable-with-caveats."""
    verdict = {ON_SEED: "IMPROVED", THIN: "IMPROVED", ASPECT: "WORSENED",
               NEEDLE: "WITHIN FLOOR", LPIPS: "WITHIN FLOOR", PSNR: "WITHIN FLOOR"}
    rows = {k: {"delta": -1.0, "floor_spread_n3": 0.0017} for k in verdict}
    assert drift_columns(rows, verdict, {"per_arm_fired": [ASPECT],
                                         "cumulative_fired": []}) == []


def test_a_band2_column_that_CAUSED_the_fail_is_flagged_not_silently_listed():
    """Drift means "does not DROP". A Band 2 column that WORSENED is the reason the scene
    dropped, and it also satisfies the literal drift definition. It is reported -- hiding a
    measured move is worse -- but flagged, so a reader cannot take the whole list as
    'adoptable with caveats'.

    Would catch the flag being computed from the BAND-2 verdict alone, which would mark a
    merely-improved column on a failing scene.
    """
    verdict = {ON_SEED: "WORSENED", THIN: "WORSENED", ASPECT: "WORSENED",
               NEEDLE: "IMPROVED", LPIPS: "WITHIN FLOOR", PSNR: "WITHIN FLOOR"}
    rows = {k: {"delta": -0.003, "floor_spread_n3": 0.0007} for k in verdict}
    got = {x["metric"]: x["caused_band2_fail"] for x in
           drift_columns(rows, verdict, {"per_arm_fired": [], "cumulative_fired": []},
                         "FAIL")}
    assert got[ON_SEED] is True and got[THIN] is True
    assert got[ASPECT] is False, "aspect is not a Band 2 column"
    assert NEEDLE not in got, "an IMPROVED column is not drift at all"
    passing = {x["metric"]: x["caused_band2_fail"] for x in
               drift_columns(rows, verdict, {"per_arm_fired": [], "cumulative_fired": []},
                             "PASS")}
    assert not any(passing.values())


def test_a_psnr_LOSS_inside_band3_is_drift_and_a_psnr_GAIN_is_not():
    rows = {PSNR: {"delta": -0.103, "floor_spread_n3": 0.0116}}
    got = drift_columns(rows, {PSNR: "MOVED"}, {"per_arm_fired": [], "cumulative_fired": []})
    assert [x["metric"] for x in got] == [PSNR] and got[0]["sign"] == "worse"
    rows[PSNR]["delta"] = +0.103
    assert drift_columns(rows, {PSNR: "MOVED"},
                         {"per_arm_fired": [], "cumulative_fired": []}) == []


# ------------------------------------------------------------------ outcome classes

def _g(**over):
    d = {"scene": "s", "band1_fired": False, "band2": "PASS", "band3_fired": False,
         "drift": [], "dn": 0.0, "falsifier_triggered_on_this_scene": False,
         "scene_drop": False, "geometry_gate": {}, "psnr_verdict": "WITHIN FLOOR",
         "scene_pass": True}
    d.update(over)
    return d


def test_pass_on_both_with_NO_drift_is_KEEP_AS_DEFAULT():
    v = combined_verdict({"a": _g(), "b": _g()})
    assert v["decision"] == "KEEP AS DEFAULT"


def test_pass_on_both_WITH_drift_is_OPT_IN_DEFAULT_CANDIDATE_and_names_the_A_B():
    """A default-candidate is adoptable as default only by a blind visual A/B, not by this
    grader. Would catch an implementation that promotes drift-carrying arms to default."""
    v = combined_verdict({"a": _g(drift=[{"metric": ASPECT}]), "b": _g()})
    assert v["decision"] == "OPT-IN, DEFAULT-CANDIDATE"
    assert "A/B" in v["promotion_requires"]


def test_pass_on_ONE_and_within_floor_on_the_other_is_OPT_IN():
    v = combined_verdict({"a": _g(), "b": _g(band2="WITHIN FLOOR", scene_pass=False)})
    assert v["decision"] == "OPT-IN"


def test_band1_ANYWHERE_is_a_DROP_even_when_both_scenes_pass_band2():
    """The single most important ordering. Would catch an implementation that reaches an
    opt-in branch before the collapse branch."""
    v = combined_verdict({"a": _g(), "b": _g(band1_fired=True, scene_drop=True)})
    assert v["decision"] == "DROP" and v["regressed_on"] == ["b"]


def test_a_band2_FAIL_anywhere_is_a_DROP():
    v = combined_verdict({"a": _g(), "b": _g(band2="FAIL", scene_pass=False,
                                             scene_drop=True)})
    assert v["decision"] == "DROP"


def test_a_band3_firing_anywhere_is_a_DROP():
    v = combined_verdict({"a": _g(), "b": _g(band3_fired=True, scene_drop=True)})
    assert v["decision"] == "DROP"


def test_no_pass_and_no_drop_is_still_NOT_ADOPTED():
    v = combined_verdict({"a": _g(band2="WITHIN FLOOR", scene_pass=False),
                          "b": _g(band2="WITHIN FLOOR", scene_pass=False)})
    assert v["decision"].startswith("NOT ADOPTED")


# ------------------------------------------------------------------- the frozen anchor

def test_the_committed_anchor_matches_the_tier3_F_arm_means_it_claims_to_be(tmp_path):
    """The anchor is defined as the scene's Tier 3 F-arm means. If it is ever hand-edited
    away from them, the cumulative check silently measures against a fiction.

    Recomputes both scenes' anchors from the committed floors.json.
    """
    root = Path(__file__).resolve().parents[1] / "bench/results/plane_aux"
    anchor = json.loads((root / "tier3_anchor.json").read_text())
    for scene in ("pgeom", "arkit"):
        fl = json.loads((root / scene / "floors.json").read_text())["floors"]
        got = anchor["scenes"][scene]["values"]
        assert set(got) == set(COLLAPSE), f"{scene}: anchor columns != Band 1 columns"
        for col in COLLAPSE:
            assert got[col] == pytest.approx(fl[col]["mean"], rel=0, abs=1e-12), col


def test_the_anchor_REFUSES_a_run_at_a_different_budget_or_resolution():
    """'re-measured only when scene, budget or resolution changes' -- so a run that changed
    one must not be graded against the old anchor. Would catch a guard that checks only
    that the scene name matches."""
    cfg = {"budget": 500000, "steps": 30000, "max_resolution": 1920, "num_downscales": 0}
    check_anchor_applies("pgeom", {"config": cfg}, dict(cfg))
    for k, v in (("budget", 300000), ("max_resolution", 1600), ("steps", 7000),
                 ("num_downscales", 2)):
        bad = dict(cfg); bad[k] = v
        with pytest.raises(SystemExit, match=k):
            check_anchor_applies("pgeom", {"config": cfg}, bad)


def test_an_unknown_scene_has_no_anchor_and_must_ERROR():
    """Would catch a `.get(scene, {})` that turns a missing anchor into an empty one, which
    would then make every cumulative check vacuous."""
    with pytest.raises(SystemExit, match="lego"):
        load_anchor(Path(__file__).resolve().parents[1] /
                    "bench/results/plane_aux/tier3_anchor.json", "lego")


# ------------------------------------------------- smid / smax and the hard-needle column

# P-GEOM's REAL Tier 3 floors, from bench/results/plane_aux/pgeom/floors.json. A fixture
# with a made-up wide floor (0.01 on every column) hid this file's own drift test: aspect's
# real floor is 0.0017, so a -0.008 move is 4.6x floor and DRIFTS, while against 0.01 it
# reads WITHIN FLOOR and the test passed for the wrong reason. Use the measured widths.
PGEOM_FLOOR = {ON_SEED: 0.000694, THIN: 0.166511, ASPECT: 0.00173074,
               NEEDLE: 0.00128201, PSNR: 0.0400238, LPIPS: 0.00122465,
               "run.smid_p50_mm": 0.006, "run.smax_p50_mm": 0.185}


def _fl(**over):
    base = {ON_SEED: 0.0719, THIN: 30.35, ASPECT: 0.3126, NEEDLE: 0.1516, PSNR: 22.57,
            LPIPS: 0.4012, "run.smid_p50_mm": 7.15, "run.smax_p50_mm": 25.85}
    return {k: {"F0": v, "F1": v, "F2": v, "mean": v,
                "spread_n3": over.get(k + "_floor", PGEOM_FLOOR[k]),
                "repeat_pair_abs_diff": 0.0} for k, v in base.items()}


def test_the_drift_fixture_can_actually_SEPARATE_drift_from_within_floor():
    """The fixture's own discriminating power, asserted so a future widening of the floors
    cannot quietly re-pin the drift tests by making every move WITHIN FLOOR."""
    fl = _fl()
    assert abs(0.3046 - 0.3126) > 3 * fl[ASPECT]["spread_n3"]
    assert abs(0.1574 - 0.1516) > 3 * fl[NEEDLE]["spread_n3"]
    # ...and still far BELOW the Band 1 collapse threshold, or it would be a DROP not drift
    assert collapse_delta(ASPECT, 0.3046, 0.3126) < 0.2 * COLLAPSE[ASPECT]["threshold"]
    assert collapse_delta(NEEDLE, 0.1574, 0.1516) < 0.2 * COLLAPSE[NEEDLE]["threshold"]


def _t(**over):
    base = {ON_SEED: 0.0719, THIN: 30.35, ASPECT: 0.3126, NEEDLE: 0.1516, PSNR: 22.57,
            LPIPS: 0.4012, "run.smid_p50_mm": 7.15, "run.smax_p50_mm": 25.85}
    base.update(over)
    return {"seed": 42, "git": "x", "depth_source": "plane-aux", "seed_cloud": "tsdf.txt",
            "thin_axis_evaluated": 250000, "values": base}


def test_smid_and_smax_are_REPORTED_but_are_NOT_collapse_columns():
    """`Dlog aspect = Dlog smid - Dlog smax`, so aspect already IS that differential and a
    collapse test on the two halves would double-count it. They are reported because
    magnitude is what separates Task 19 from VOID and the raw sizes say it directly.

    Would catch someone adding them to COLLAPSE 'for symmetry'.
    """
    assert "run.smid_p50_mm" not in COLLAPSE and "run.smax_p50_mm" not in COLLAPSE
    d = grade("s", 0.0, _t(**{"run.smid_p50_mm": 6.96}), _fl(), _anchor(),
              {"config": {"budget": 500000, "steps": 30000, "max_resolution": 1920,
                          "num_downscales": 0}},
              {"budget": 500000, "steps": 30000, "max_resolution": 1920,
               "num_downscales": 0})
    assert "run.smid_p50_mm" in d["rows"] and "run.smax_p50_mm" in d["rows"]


def test_the_aspect_identity_that_justifies_not_gating_on_smid_and_smax():
    """Dlog(smid/smax) == Dlog smid - Dlog smax, on the real P-GEOM P0 numbers. If this
    were false the two columns would carry information aspect does not, and the decision
    above would be wrong."""
    smid_b, smax_b, smid_t, smax_t = 7.151, 25.845, 6.961, 25.661
    assert (math.log(smid_t / smax_t) - math.log(smid_b / smax_b)) == pytest.approx(
        math.log(smid_t / smid_b) - math.log(smax_t / smax_b), abs=1e-12)


def test_the_hard_needle_threshold_is_the_DELIVERY_FORMATS_orientation_error():
    """splat-transform's SOG writer stores a smallest-three quaternion as three uint8:
    `255 * (q * 0.5 + 0.5)` after scaling by +-sqrt(2) (verified in the installed
    package's `dist/index.mjs`, the `252 + maxComp` writer). One step is therefore
    sqrt(2)/255 in true component units, worst-case round-to-nearest error is step/2 per
    component over three components, and a quaternion perturbation of norm e is a rotation
    of 2e.

    Would catch the threshold drifting to a round number with no derivation.
    """
    step = math.sqrt(2) / 255
    max_rot_err = 2 * (step / 2) * math.sqrt(3)
    assert step == pytest.approx(0.005546, abs=5e-7)
    assert max_rot_err == pytest.approx(0.009606, abs=5e-7)
    assert HARD_NEEDLE_ASPECT >= max_rot_err
    assert HARD_NEEDLE_ASPECT == pytest.approx(0.01)


def test_hard_needle_is_reported_when_present_and_absent_without_error():
    """It comes from a ply, not from the trainer's report, so a re-grade of an archived arm
    may or may not have it. It must never be a gate column either way."""
    from plane_aux_arms import COLLAPSE as C, GEOMETRY_GATE as G
    assert "run.hard_needle_frac" not in C and "run.hard_needle_frac" not in G
    args = (_fl(), _anchor(),
            {"config": {"budget": 500000, "steps": 30000, "max_resolution": 1920,
                        "num_downscales": 0}},
            {"budget": 500000, "steps": 30000, "max_resolution": 1920,
             "num_downscales": 0})
    assert grade("s", 0.0, _t(), *args)["band2"] in ("PASS", "WITHIN FLOOR", "FAIL")


# --------------------------------------------------------- the whole rule, end to end

def _args(**over):
    cfg = {"budget": 500000, "steps": 30000, "max_resolution": 1920, "num_downscales": 0}
    return (_fl(), _anchor(), {"config": cfg}, dict(cfg))


TIER1_DOC = json.loads((Path(__file__).resolve().parents[1] /
                        "bench/results/plane_aux/tier1_void_row.json").read_text())


def _tier1_grade(arm: str) -> dict:
    """Grade a real Tier 1 arm with the real rule, against its own n=3 floors.

    The anchor IS the floors here (both are the B0a/B0b/B0d means), which is the same
    degenerate case Task 19 is in -- the ratchet only opens up once a treatment has been
    adopted and the floors move under it.
    """
    a = TIER1_DOC["arms"][arm]
    fl = TIER1_DOC["floors"]
    anchor_values = {c: fl[c]["mean"] for c in COLLAPSE}
    cfg = {k: a["resolved"][k] for k in
           ("budget", "steps", "max_resolution", "num_downscales")}
    t = {"seed": a["seed"], "git": a["git"], "depth_source": "n/a",
         "seed_cloud": "points3D.tsdf.txt", "thin_axis_evaluated": None,
         "values": a["values"]}
    return grade("pgeom-tier1", 0.05, t, fl, anchor_values, {"config": cfg}, cfg)


def test_the_TIER1_FLOORS_reproduce_section_8_2s_published_n3_spreads():
    """If the committed Tier 1 row does not reproduce the floor table the note publishes,
    it is not the same data and the VOID grade below means nothing. Would catch the row
    being rebuilt from the wrong three arms -- B0c instead of B0d, say, which is what that
    run's own floors.json uses."""
    published = {NEEDLE: 0.00221, ASPECT: 0.00270, ON_SEED: 0.00061, LPIPS: 0.00073,
                 THIN: 0.05481, PSNR: 0.10019}
    for k, p in published.items():
        assert TIER1_DOC["floors"][k]["spread_n3"] == pytest.approx(p, abs=5e-6), k


def test_THE_VOID_ROW_GRADED_END_TO_END_FROM_ITS_OWN_ARTIFACTS_IS_A_DROP():
    """THE DEMONSTRATION THE WHOLE RULE RESTS ON, run on the arm itself rather than on a
    table transcribed from it. `R1_openweights` is the pre-fix open-weight-path run;
    research/metal-gauss.md section 8.1 struck it through and it is the upper anchor of
    every Band 1 threshold.

    It must DROP, and it must do so on FOUR independent Band 1 columns -- so no single
    column carries the rule. Band 3 fires independently as well (-1.03 dB), which is a
    fourth path to the same verdict and is stated so nobody reads the Band 1 result as
    load-bearing on its own.
    """
    d = _tier1_grade("R1_openweights")
    assert d["scene_drop"] and d["band1_fired"]
    assert set(d["band1"]["per_arm_fired"]) == {NEEDLE, ASPECT, ON_SEED, LPIPS}
    assert set(d["band1"]["cumulative_fired"]) == {NEEDLE, ASPECT, ON_SEED, LPIPS}
    assert d["band3_fired"] and d["band3"]["loss_db"] > 1.0
    assert [x["metric"] for x in d["drift"]] == [PSNR], \
        "every COLLAPSED column must be excluded from drift; PSNR remains because it is " \
        "Band 3's business, not Band 1's"
    assert d["drift"][0]["caused_band3_fire"] is True, \
        "and it must be flagged as the reason for the drop, not read as a caveat"
    # and the discriminating check that motivates the whole battery: thin-axis, which the
    # recipe exists to improve, calls this destroyed reconstruction an IMPROVEMENT.
    assert d["geometry_gate"][THIN] == "IMPROVED"


@pytest.mark.parametrize("arm", ["R1", "R1p"])
def test_NO_ADOPTED_Tier_1_arm_trips_BAND_1_or_BAND_3(arm):
    """The other half of the fence. A rule that catches VOID by dropping everything is not
    a rule, so the arms this project ADOPTED must clear the COLLAPSE and PHOTOMETRIC bands,
    graded identically and against the same floors.

    Measured margins: every adopted arm sits at or below 0.28x of every Band 1 threshold,
    against VOID's 3.7x-7.4x. The fence separates them by more than an order of magnitude.
    """
    d = _tier1_grade(arm)
    assert not d["band1_fired"], d["band1"]["per_arm_fired"]
    assert max(abs(v["x_threshold"]) for v in d["band1"]["per_arm"].values()) < 0.3
    assert not d["band3_fired"]


def test_R1p_FAILS_BAND_2_against_the_Tier_1_baseline_and_that_is_a_REAL_finding():
    """RECORDED BECAUSE IT IS TRUE, NOT BECAUSE IT IS COMFORTABLE.

    R1 clears Band 2 (on-seed 0.0847 -> 0.0890). R1p does NOT: its on-seed FALLS to 0.0814,
    5.1x the 0.00061 floor, so Band 2 reads FAIL and the arm drops. Section 8.1's headline
    "on-seed UP on both" is about R1; R1p is the dn = 0 variant and it is not true of it.

    This is a property of the rule meeting the data, not an implementation fault, and it
    matters for reading Task 19: R1p is Task 19's BASE arm. Grading a base against a
    baseline is not what the rule is for -- Task 19 grades plane-aux against R1p's own
    re-measured floors, which is a different and legitimate comparison -- but a rule whose
    Band 2 would reject the base it measures from is worth stating out loud rather than
    discovering later.

    Would catch someone "fixing" this by loosening Band 2 to `not WORSENED`, which is
    exactly the condition the pre-registered rule deliberately does not use for on-seed.
    """
    r1, r1p = _tier1_grade("R1"), _tier1_grade("R1p")
    assert r1["band2"] == "PASS"
    assert r1["geometry_gate"][ON_SEED] == "IMPROVED"
    assert r1p["band2"] == "FAIL"
    assert r1p["geometry_gate"][ON_SEED] == "WORSENED"
    assert r1p["scene_drop"] and not r1p["band1_fired"] and not r1p["band3_fired"]


def test_even_R1_is_a_DEFAULT_CANDIDATE_not_a_clean_default_on_this_scene():
    """The adopted recipe itself carries drift -- needles 11.7x floor, aspect 8.3x, LPIPS
    2.8x -- so under the replacement rule R1 would have been OPT-IN, DEFAULT-CANDIDATE and
    not KEEP AS DEFAULT.

    Recorded so nobody reads Task 19's default-candidate verdict as unusually weak. On this
    scene, at these floors, a beyond-floor anti-collapse drift is what the recipe that IS
    the default already looks like.
    """
    d = _tier1_grade("R1")
    assert d["band2"] == "PASS" and not d["scene_drop"]
    assert {x["metric"] for x in d["drift"]} >= {NEEDLE, ASPECT}
    assert combined_verdict({"pgeom-tier1": d})["decision"] == "OPT-IN, DEFAULT-CANDIDATE"


def test_the_VOID_row_end_to_end_is_a_DROP_and_names_collapse_not_drift():
    """The rule as a whole, on the row it exists to catch -- not just band1() in isolation.
    Scaled onto the Tier 3 floors so it is graded exactly as a real arm would be."""
    d = grade("s", 0.0, _t(**{NEEDLE: 0.1516 + 0.402, ASPECT: 0.3126 * (0.0659 / 0.2957),
                              ON_SEED: 0.0719 * (0.0359 / 0.0847), LPIPS: 0.4012 + 0.1262,
                              THIN: 29.0}), *_args())
    assert d["scene_drop"] and d["band1_fired"]
    assert len(d["band1"]["per_arm_fired"]) == 4
    assert d["drift"] == [], "a collapsed column must not also be reported as drift"
    # and the discriminating check: thin-axis alone STILL calls this an improvement
    assert d["geometry_gate"][THIN] == "IMPROVED"


def test_an_arm_that_only_DRIFTS_is_not_a_drop():
    d = grade("s", 0.0, _t(**{ON_SEED: 0.0980, THIN: 28.0, ASPECT: 0.3046,
                              NEEDLE: 0.1574}), *_args())
    assert not d["scene_drop"] and d["band2"] == "PASS"
    assert {x["metric"] for x in d["drift"]} == {ASPECT, NEEDLE}
