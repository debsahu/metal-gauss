"""The plane-aux keep/drop rule, unit-tested.

The rule is pre-registered in the first commit of `feat/plane-aux` and implemented in
`scripts/plane_aux_arms.py`. Without these tests it would be exercised only by an
eight-hour run, once, at the end -- which is precisely when a wrong rule is most expensive
and least likely to be noticed.

These need no GPU and no artifacts.
"""
import json
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

def _cfg(**over):
    c = {"depth_normal_weight": 0.0, "depth_loss_space": "disparity",
         "depth_source": "center", "flatten_loss_weight": 1.0,
         "depth_loss_weight": 1.0, "normal_loss_weight": 0.2,
         "budget": 500000, "steps": 30000, "max_resolution": 1920, "num_downscales": 0}
    c.update(over)
    return c


def test_reused_floors_must_match_the_base_configuration():
    """Would catch a --skip-floors that checks only that floors.json EXISTS.

    Reusing floors is legitimate (step 7 grades a second treatment against the same base)
    and is also exactly how a floor from the wrong configuration gets applied silently --
    section 8.2's failure.
    """
    from plane_aux_arms import check_floors_match
    ok = [_cfg(), _cfg(), _cfg()]
    check_floors_match(ok, "center", 0.0, "disparity")
    with pytest.raises(SystemExit, match="depth_normal_weight"):
        check_floors_match(ok, "center", 0.05, "disparity")
    with pytest.raises(SystemExit, match="depth_loss_space"):
        check_floors_match(ok, "center", 0.0, "metric")


def test_floors_measured_on_ONE_depth_source_are_not_the_floors_of_ANOTHER():
    """THE CASE THAT MATTERS FOR STEP 7. If the winning depth source is `plane-aux`, the
    `center` floors already on disk are not its floors, and the honest answer is to
    re-measure. Would catch a guard that checked dn and loss space but not depth source --
    which is what this guard originally did."""
    from plane_aux_arms import check_floors_match
    center_floors = [_cfg(), _cfg(), _cfg()]
    with pytest.raises(SystemExit, match="depth_source"):
        check_floors_match(center_floors, "plane-aux", 0.0, "disparity")


def test_floor_arms_that_DISAGREE_WITH_EACH_OTHER_are_not_a_noise_floor():
    """The check a summary field in floors.json cannot perform at all.

    Three runs that differ in a flag are not a repeat measurement of anything, and their
    max-min is a treatment effect wearing a noise floor's clothes -- which would then be
    used to decide whether the treatment arm 'moved' a metric. Reading each arm's OWN
    resolved settings is what makes this detectable.
    """
    from plane_aux_arms import check_floors_match
    mixed = [_cfg(), _cfg(flatten_loss_weight=0.0), _cfg()]
    with pytest.raises(SystemExit, match="differ in configuration"):
        check_floors_match(mixed, "center", 0.0, "disparity")
    drifted = [_cfg(), _cfg(), _cfg(max_resolution=1600)]
    with pytest.raises(SystemExit, match="differ in configuration"):
        check_floors_match(drifted, "center", 0.0, "disparity")


def test_a_missing_configuration_key_is_a_MISMATCH_not_agreement():
    """An older report that predates a field cannot testify about it."""
    from plane_aux_arms import check_floors_match
    c = _cfg(); c.pop("depth_loss_space")
    with pytest.raises(SystemExit, match="absent"):
        check_floors_match([c, c, c], "center", 0.0, "disparity")
    with pytest.raises(SystemExit, match="no floor arm reports"):
        check_floors_match([], "center", 0.0, "disparity")


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


# --------------------------------------------- --summary must NAME the scenes it counts

def _write_grade(d, scene, **over):
    import json
    d.mkdir(parents=True, exist_ok=True)
    g = {"scene": scene, "dn": 0.0, "scene_pass": False, "scene_drop": False,
         "falsifier_triggered_on_this_scene": False, "geometry_gate": {},
         "psnr_verdict": "WITHIN FLOOR"}
    g.update(over)
    (d / "grade.json").write_text(json.dumps(g))


def test_summary_refuses_an_UNNAMED_grade_json_rather_than_counting_it(tmp_path):
    """A NEAR-MISS MADE STRUCTURAL, not defensiveness.

    `--summary` globbed every subdirectory of --out holding a grade.json, and the grader's
    own SYNTHETIC smoke fixtures -- one a fabricated pass/regress pair -- were sitting in
    that same tree. Because DROP is checked first and is not overridable, a fabricated
    regression would have produced a DROP indistinguishable from a measured one, with
    nothing erroring. Moving those directories by hand is not a fix; naming the scenes is.

    The half that matters is the EXTRA case: a missing scene is loud anyway, while a stray
    one is exactly what a glob gets wrong.
    """
    from plane_aux_arms import collect_scenes
    _write_grade(tmp_path / "pgeom", "pgeom")
    _write_grade(tmp_path / "arkit", "arkit")
    assert set(collect_scenes(tmp_path, "pgeom,arkit")) == {"pgeom", "arkit"}

    _write_grade(tmp_path / "regrade_smoke", "regrade_smoke", scene_drop=True)
    with pytest.raises(SystemExit, match="regrade_smoke"):
        collect_scenes(tmp_path, "pgeom,arkit")


def test_summary_refuses_a_MISSING_named_scene(tmp_path):
    from plane_aux_arms import collect_scenes
    _write_grade(tmp_path / "pgeom", "pgeom")
    with pytest.raises(SystemExit, match="arkit"):
        collect_scenes(tmp_path, "pgeom,arkit")


def test_summary_refuses_an_EMPTY_scene_list(tmp_path):
    """Would catch a default that silently re-enables the glob."""
    from plane_aux_arms import collect_scenes
    _write_grade(tmp_path / "pgeom", "pgeom")
    with pytest.raises(SystemExit, match="requires --scenes"):
        collect_scenes(tmp_path, "")


def test_summary_refuses_when_the_directory_and_the_report_disagree(tmp_path):
    """The directory name is not evidence about what was measured; the report is. Would
    catch a grade.json copied into the wrong scene directory."""
    from plane_aux_arms import collect_scenes
    _write_grade(tmp_path / "arkit", "pgeom")          # wrong scene inside arkit/
    with pytest.raises(SystemExit, match="disagree"):
        collect_scenes(tmp_path, "arkit")


def test_the_floors_space_override_DEFAULTS_to_the_treatments_space():
    """`--floors-depth-loss-space` exists for step 7 only, where the base is the DISPARITY
    arm and the treatment is the METRIC one, so the floors to reuse are the base's.

    The override must not weaken the ordinary case. It defaults to empty, and the call site
    is `a.floors_depth_loss_space or a.depth_loss_space` -- so with the flag unset a
    disparity-floor / metric-arm reuse is still REFUSED, and the divergence has to be
    stated on the command line to happen at all. Would catch a default of "disparity",
    which would silently permit every cross-space reuse.
    """
    from plane_aux_arms import check_floors_match
    disparity_floors = [_cfg(), _cfg(), _cfg()]
    # flag unset -> falls back to the treatment's space -> refused
    with pytest.raises(SystemExit, match="depth_loss_space"):
        check_floors_match(disparity_floors, "center", 0.0, "" or "metric")
    # flag set to the BASE's space -> allowed, and only then
    check_floors_match(disparity_floors, "center", 0.0, "disparity" or "metric")


# ------------------------------------- a non-primary grade must not become THE verdict

def test_only_the_PRIMARY_treatment_writes_grade_json(tmp_path):
    """A DEFECT CAUGHT IN FLIGHT, now pinned.

    `--regrade` wrote `grade.json` unconditionally, so re-grading the step-7 arm
    (`--treatment-tag M0`) OVERWROTE the step-5 scene verdict with the metric-space one --
    and `--summary` reads `grade.json`. The cross-scene decision would then have been
    computed from the wrong arm and reported as the plane-aux result, with NOTHING
    ERRORING: both files are well-formed grades of real arms, differing only in which arm
    they grade. It actually happened, and was caught only because the two verdicts
    disagreed visibly.

    The scene verdict belongs to the pre-registered arm. P0 writes both files; every other
    tag writes only its own.
    """
    from plane_aux_arms import write_grade
    write_grade(tmp_path, "P0", {"scene": "s", "which": "plane-aux"})
    assert (tmp_path / "grade_P0.json").exists()
    assert json.loads((tmp_path / "grade.json").read_text())["which"] == "plane-aux"

    write_grade(tmp_path, "M0", {"scene": "s", "which": "metric-space"})
    assert json.loads((tmp_path / "grade_M0.json").read_text())["which"] == "metric-space"
    assert json.loads((tmp_path / "grade.json").read_text())["which"] == "plane-aux", \
        "a non-primary arm overwrote the scene verdict that --summary reads"
