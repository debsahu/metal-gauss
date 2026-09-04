"""The plane-aux keep/drop rule, unit-tested.

The rule is pre-registered in the first commit of `feat/plane-aux` and implemented in
`scripts/plane_aux_arms.py`. Without these tests it would be exercised only by an
eight-hour run, once, at the end -- which is precisely when a wrong rule is most expensive
and least likely to be noticed.

These need no GPU and no artifacts.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from plane_aux_arms import GEOMETRY_GATE, grade, verdict_for  # noqa: E402

ON_SEED = "stats.on_seed_frac_1cm"
THIN = "stats.thin_axis_angle_p50"
ASPECT = "run.aspect_p50"
NEEDLE = "run.needle_frac"
PSNR = "run.psnr_masked"


def test_the_floor_comparison_is_STRICT():
    """A delta exactly equal to the floor has NOT cleared it.

    Would catch `>=`. The difference only ever shows on an exact tie, which is exactly the
    case a hand-run comparison would wave through.
    """
    assert verdict_for(ON_SEED, 0.010, 0.010) == "WITHIN FLOOR"
    assert verdict_for(ON_SEED, 0.0100001, 0.010) == "IMPROVED"


@pytest.mark.parametrize("metric,delta,want", [
    (ON_SEED, +0.05, "IMPROVED"),   # higher on-seed is better
    (ON_SEED, -0.05, "WORSENED"),
    (THIN, -5.0, "IMPROVED"),       # LOWER thin-axis is better -- the sign that is easiest
    (THIN, +5.0, "WORSENED"),       # to get backwards, and a backwards one inverts the gate
    (ASPECT, +0.05, "IMPROVED"),    # higher in-plane aspect is better (less needly)
    (ASPECT, -0.05, "WORSENED"),
    (NEEDLE, -0.05, "IMPROVED"),    # lower needle fraction is better
    (NEEDLE, +0.05, "WORSENED"),
    (PSNR, +5.0, "MOVED"),          # two-sided: PSNR must stay put, either way
    (PSNR, -5.0, "MOVED"),
])
def test_each_metrics_direction_of_worse(metric, delta, want):
    assert verdict_for(metric, delta, 0.001) == want


def _fl(**over):
    """Floors for a base arm; every gate metric present, spread 0.01 unless overridden."""
    base = {ON_SEED: 0.08, THIN: 30.0, ASPECT: 0.27, NEEDLE: 0.19, PSNR: 22.5}
    return {k: {"F0": v, "F1": v, "F2": v, "mean": v, "spread_n3": over.get(k + "_floor", 0.01),
                "repeat_pair_abs_diff": 0.0} for k, v in base.items()}


def _t(**over):
    base = {ON_SEED: 0.08, THIN: 30.0, ASPECT: 0.27, NEEDLE: 0.19, PSNR: 22.5}
    base.update(over)
    return {"seed": 42, "git": "x", "depth_source": "plane-aux", "seed_cloud": "tsdf.txt",
            "thin_axis_evaluated": 250000, "values": base}


def test_a_clean_pass_passes():
    d = grade("s", 0.0, _t(**{ON_SEED: 0.10, THIN: 28.0}), _fl())
    assert d["scene_pass"] and not d["scene_drop"]
    assert not d["falsifier_triggered_on_this_scene"]


def test_on_seed_must_RISE_not_merely_not_fall():
    """The rule says on-seed@1cm must rise by more than the floor. An arm that improves
    thin-axis and leaves on-seed alone is NOT a pass -- would catch a gate written as
    `!= WORSENED` for this column, which is how the other three are written."""
    d = grade("s", 0.0, _t(**{THIN: 20.0}), _fl())
    assert not d["scene_pass"]
    assert d["geometry_gate"][ON_SEED] == "WITHIN FLOOR"


def test_a_needle_collapse_DROPS_even_when_thin_axis_improves():
    """THE VOID ROW, as a test. research/metal-gauss.md section 8.1: the pre-fix recipe hit
    a similar thin-axis by DESTROYING the splats -- in-plane aspect 0.2957 -> 0.0659,
    needle fraction 16.6% -> 56.8%, on-seed halved -- and thin-axis, opacity and dark
    fraction all called it healthier than baseline.

    Would catch a gate built from thin-axis and on-seed alone, which is the natural
    three-column gate someone would write.
    """
    d = grade("s", 0.0, _t(**{THIN: 20.0, ASPECT: 0.066, NEEDLE: 0.568, ON_SEED: 0.04}),
              _fl())
    assert d["scene_drop"] and not d["scene_pass"]
    assert d["geometry_gate"][ASPECT] == "WORSENED"
    assert d["geometry_gate"][NEEDLE] == "WORSENED"
    # and the discriminating check: thin-axis alone WOULD have called this an improvement
    assert d["geometry_gate"][THIN] == "IMPROVED"


def test_a_psnr_move_blocks_a_pass_in_EITHER_direction():
    for p in (25.0, 20.0):
        d = grade("s", 0.0, _t(**{ON_SEED: 0.10, PSNR: p}), _fl())
        assert d["psnr_verdict"] == "MOVED"
        assert not d["scene_pass"]


def test_the_falsifier_fires_only_when_BOTH_on_seed_and_thin_axis_stay_put():
    both_still = grade("s", 0.0, _t(**{ASPECT: 0.30}), _fl())
    assert both_still["falsifier_triggered_on_this_scene"]
    thin_moved = grade("s", 0.0, _t(**{THIN: 25.0}), _fl())
    assert not thin_moved["falsifier_triggered_on_this_scene"]
    seed_moved = grade("s", 0.0, _t(**{ON_SEED: 0.10}), _fl())
    assert not seed_moved["falsifier_triggered_on_this_scene"]


def test_a_MISSING_gate_column_is_an_error_and_never_a_pass():
    """THE FAILURE THIS PROJECT KEEPS REPEATING: a check that reads a condition something
    OTHER than the thing being checked could satisfy.

    If splatstats failed to emit `thin_axis_angle_p50`, `verdict.get(...)` returns None,
    `None != "WORSENED"` is True, and the gate would silently PASS on a metric that was
    never measured. It must raise instead.
    """
    fl = _fl(); fl.pop(THIN)
    t = _t(); t["values"].pop(THIN)
    with pytest.raises(SystemExit, match=THIN):
        grade("s", 0.0, t, fl)


def test_every_gate_column_has_a_declared_direction():
    """Would catch a column added to GEOMETRY_GATE without a DIRECTION entry, which would
    make it silently unverdicted -- and therefore, per the test above, an error rather
    than a silent pass; this asserts the error is a KeyError at authoring time instead."""
    from plane_aux_arms import DIRECTION
    for k in GEOMETRY_GATE:
        assert k in DIRECTION, k
        assert DIRECTION[k] in (+1, -1), f"{k} is a gate column, so it cannot be two-sided"


# ------------------------------------------------------------ reusing floors safely

def test_reused_floors_must_MATCH_the_configuration_and_a_missing_key_is_a_mismatch():
    """Would catch a `--skip-floors` that checks only that floors.json EXISTS.

    Reusing floors is legitimate (step 7 grades a second treatment against the same base)
    and is also exactly how a floor from the wrong configuration gets applied silently --
    section 8.2's failure. The last case is the one a permissive `.get(k, default)` would
    wave through: an older floors.json that predates the field cannot testify about it, so
    absence must read as MISMATCH, not as agreement.
    """
    from plane_aux_arms import check_floors_match
    check_floors_match({"dn": 0.0, "depth_loss_space": "disparity"}, 0.0, "disparity")
    with pytest.raises(SystemExit, match="dn=0.05"):
        check_floors_match({"dn": 0.05, "depth_loss_space": "disparity"}, 0.0, "disparity")
    with pytest.raises(SystemExit, match="space=metric"):
        check_floors_match({"dn": 0.0, "depth_loss_space": "metric"}, 0.0, "disparity")
    with pytest.raises(SystemExit, match="absent"):
        check_floors_match({"dn": 0.0}, 0.0, "disparity")
    with pytest.raises(SystemExit, match="absent"):
        check_floors_match({}, 0.0, "disparity")


# ---------------------------------------------------------- the cross-scene decision

def _g(**over):
    d = {"scene_pass": False, "scene_drop": False,
         "falsifier_triggered_on_this_scene": False, "dn": 0.0,
         "geometry_gate": {}, "psnr_verdict": "WITHIN FLOOR"}
    d.update(over)
    return d


def test_a_regression_on_ONE_scene_is_a_DROP_even_if_the_other_scene_passes():
    """The single most important ordering in the rule.

    Would catch an implementation that checks the opt-in condition ("passes on one, inside
    the floor on the other") before the drop condition. That reading turns
    pass-here/regress-there into a RECOMMENDATION, when the pre-registered rule says
    "DROP if it worsens any of the four geometry columns beyond the floor on EITHER scene".
    """
    from plane_aux_arms import combined_verdict
    v = combined_verdict({"pgeom": _g(scene_pass=True),
                          "arkit": _g(scene_drop=True)})
    assert v["decision"] == "DROP"
    assert v["regressed_on"] == ["arkit"] and v["passed_on"] == ["pgeom"]


def test_pass_on_both_is_the_recipe_default_and_pass_on_one_is_opt_in():
    from plane_aux_arms import combined_verdict
    both = combined_verdict({"a": _g(scene_pass=True), "b": _g(scene_pass=True)})
    assert both["decision"] == "KEEP AS RECIPE DEFAULT"
    one = combined_verdict({"a": _g(scene_pass=True), "b": _g()})
    assert one["decision"] == "KEEP AS OPT-IN"


def test_no_pass_and_no_regression_is_NOT_adoption():
    """An arm that changes nothing is not an opt-in. Would catch an `else: OPT-IN`."""
    from plane_aux_arms import combined_verdict
    v = combined_verdict({"a": _g(), "b": _g()})
    assert v["decision"].startswith("NOT ADOPTED")


def test_the_falsifier_is_reported_INCOMPLETE_while_only_one_dn_setting_is_measured():
    """The pre-registered falsifier requires both scenes AT BOTH dn SETTINGS. Step 6
    (dn = 0.05) is gated behind Task 20, so a dn = 0 sweep alone cannot falsify.

    Would catch reporting `falsifier_at_measured_dn` as the falsifier -- a partial
    falsifier presented as a falsification is exactly the over-claim this flag exists to
    prevent.
    """
    from plane_aux_arms import combined_verdict
    v = combined_verdict({"a": _g(falsifier_triggered_on_this_scene=True),
                          "b": _g(falsifier_triggered_on_this_scene=True)})
    assert v["falsifier_at_measured_dn"] is True
    assert v["falsifier_complete"] is False
    assert v["dn_settings_measured"] == [0.0]
    v2 = combined_verdict({"a": _g(falsifier_triggered_on_this_scene=True),
                           "b": _g(falsifier_triggered_on_this_scene=True, dn=0.05)})
    assert v2["falsifier_complete"] is True


def test_the_falsifier_needs_EVERY_scene_not_just_one():
    from plane_aux_arms import combined_verdict
    v = combined_verdict({"a": _g(falsifier_triggered_on_this_scene=True), "b": _g()})
    assert v["falsifier_at_measured_dn"] is False
