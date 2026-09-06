"""Tests for the needle batch grader.

The grader is the thing that turns 30 GPU-hours into a verdict, so the defect that
matters is not "a number is formatted oddly" -- it is a bar that cannot fire, or an
ABSENT measurement grading as a pass. Both have happened in this project: a monitor whose
failure filter matched "oom" inside "playroom", and an assertion over an empty array.

Every test below names what it catches.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import needle_table as NT  # noqa: E402


GOOD_SHAPE = {"aspect_p50": 0.55, "needle_frac": 0.02, "hard_needle_frac": 0.0002,
              "smid_p50_mm": 9.0, "smax_p50_mm": 17.0}


def _write(out: Path, arm: str, *, shape=None, psnr=22.5, thin=49.0, n_thin=304000,
           on_seed=0.085, ungated=None, num_downscales=2, lpips=0.39, filter_3d=False):
    (out / f"{arm}.json").write_text(json.dumps({
        "env": {"git": "abc1234", "dirty": False},
        "resolved": {"num_downscales": num_downscales, "filter_3d": filter_3d},
        "metrics": {"psnr_masked": psnr, "lpips": lpips, "coverage": 1.0,
                    "ms_per_step": 32.0, "n_splats": 500000,
                    "shape": dict(GOOD_SHAPE if shape is None else shape)},
        "log": [{"step": 2500, "shape": dict(GOOD_SHAPE)},
                {"step": 30000, "shape": dict(GOOD_SHAPE if shape is None else shape)}]}))
    (out / f"{arm}.stats.json").write_text(json.dumps({"metrics": {
        "thin_axis_angle_median_deg": thin, "thin_axis_evaluated": n_thin,
        "on_seed_frac_1cm": on_seed}}))
    if ungated is not None:
        (out / f"{arm}.ungated.json").write_text(json.dumps({"metrics": {
            "thin_axis_angle_median_deg": ungated, "thin_axis_evaluated": 500000}}))


def _floors(out: Path):
    (out / "floors.json").write_text(json.dumps({"floors": {
        "stats.thin_axis_angle_median_deg": {"repeat_floor": 0.0548},
        "stats.on_seed_frac_1cm": {"repeat_floor": 0.0006}}}))


def _grade(out: Path, arms):
    return NT.grade([NT.read_arm(out, a) for a in arms], arms[0],
                    json.loads((out / "floors.json").read_text()))


def test_a_clean_arm_passes(tmp_path):
    """The control. Without it, every test below is satisfied by a grader that fails
    everything."""
    _floors(tmp_path)
    _write(tmp_path, "B0a", ungated=51.0)
    _write(tmp_path, "I1", ungated=50.0)
    rows = _grade(tmp_path, ["B0a", "I1"])
    assert [r["verdict"] for r in rows] == ["PASS", "PASS"], [r["fails"] for r in rows]


@pytest.mark.parametrize("key,bad,good", [
    ("needle_frac", 0.0501, 0.0499),
    ("aspect_p50", 0.3999, 0.4001),
    ("hard_needle_frac", 0.00101, 0.00099),
])
def test_each_shape_bar_fires_at_its_own_threshold(tmp_path, key, bad, good):
    """CATCHES a bar wired to the wrong key, wired with the wrong comparison direction,
    or not wired at all. Each case is checked just over AND just under the line, so a bar
    that always fails cannot pass this either."""
    _floors(tmp_path)
    _write(tmp_path, "B0a", ungated=51.0)
    for name, v in (("BAD", bad), ("GOOD", good)):
        _write(tmp_path, name, shape={**GOOD_SHAPE, key: v}, ungated=50.0)
    rows = {r["arm"]: r for r in _grade(tmp_path, ["B0a", "BAD", "GOOD"])}
    assert rows["BAD"]["verdict"] == "FAIL" and any(
        f.startswith(key) for f in rows["BAD"]["fails"]), rows["BAD"]["fails"]
    assert rows["GOOD"]["verdict"] == "PASS", rows["GOOD"]["fails"]


def test_a_MISSING_shape_block_fails_and_is_not_read_as_zero(tmp_path):
    """THE IMPOSSIBLE-VALUE RULE. An arm whose ply was never scored has no shape block.
    Defaulting it to 0.0 makes `needle_frac <= 0.05` the most spectacular pass in the
    batch, from an arm that produced no measurement at all."""
    _floors(tmp_path)
    _write(tmp_path, "B0a", ungated=51.0)
    _write(tmp_path, "X", shape={}, ungated=50.0)
    rows = {r["arm"]: r for r in _grade(tmp_path, ["B0a", "X"])}
    assert rows["X"]["verdict"] == "FAIL"
    assert sorted(f.split(":")[0] for f in rows["X"]["fails"]) == [
        "aspect_p50", "hard_needle_frac", "needle_frac"]
    assert all("MISSING" in f for f in rows["X"]["fails"])


def test_an_arm_with_no_report_at_all_is_NO_REPORT_not_a_pass(tmp_path):
    """CATCHES the aborted-arm case directly: six arms once reported success because
    `rc=$?` had read `date`'s status. A grader that skips absent arms silently would
    publish a five-row table for a seven-arm batch and nothing would say so."""
    _floors(tmp_path)
    _write(tmp_path, "B0a", ungated=51.0)
    rows = {r["arm"]: r for r in _grade(tmp_path, ["B0a", "NEVER_RAN"])}
    assert rows["NEVER_RAN"]["verdict"] == "NO REPORT"


def test_the_psnr_bar_is_relative_to_the_BASELINE_in_this_batch(tmp_path):
    """CATCHES an absolute PSNR bar, or one taken against a remembered number from
    another machine. -0.1001 dB fails, -0.0999 passes, and both are far from any
    absolute threshold."""
    _floors(tmp_path)
    _write(tmp_path, "B0a", psnr=22.5000, ungated=51.0)
    _write(tmp_path, "DROP", psnr=22.3999, ungated=50.0)
    _write(tmp_path, "OK", psnr=22.4001, ungated=50.0)
    rows = {r["arm"]: r for r in _grade(tmp_path, ["B0a", "DROP", "OK"])}
    assert rows["DROP"]["verdict"] == "FAIL" and any("psnr" in f for f in rows["DROP"]["fails"])
    assert rows["OK"]["verdict"] == "PASS", rows["OK"]["fails"]


def test_a_missing_UNGATED_thin_axis_is_reported_as_unreportable(tmp_path):
    """The pre-registered reporting rule. A gated-only thin-axis number must not pass
    silently -- it is the column that has produced two confident wrong results here."""
    _floors(tmp_path)
    _write(tmp_path, "B0a", ungated=51.0)
    _write(tmp_path, "NOUNGATED", ungated=None)
    rows = {r["arm"]: r for r in _grade(tmp_path, ["B0a", "NOUNGATED"])}
    assert any("UNGATED THIN-AXIS NOT MEASURED" in n for n in rows["NOUNGATED"]["notes"])
    assert not any("UNGATED" in n for n in rows["B0a"]["notes"])


def test_a_gated_population_shift_is_flagged_as_a_composition_statistic(tmp_path):
    """CATCHES the exact reasoning error the brief names: comparing gated thin-axis
    between two arms whose gates admitted different splat counts. 5% more splats admitted
    must raise the note; 1% must not, or the note fires on every arm and is ignored."""
    _floors(tmp_path)
    _write(tmp_path, "B0a", n_thin=300000, ungated=51.0)
    _write(tmp_path, "SHIFTED", n_thin=315000, ungated=50.0)
    _write(tmp_path, "STEADY", n_thin=303000, ungated=50.0)
    rows = {r["arm"]: r for r in _grade(tmp_path, ["B0a", "SHIFTED", "STEADY"])}
    assert any("COMPOSITION" in n for n in rows["SHIFTED"]["notes"])
    assert not any("COMPOSITION" in n for n in rows["STEADY"]["notes"])


def test_a_schedule_arm_is_marked_so_its_ms_per_step_is_not_compared(tmp_path):
    """--num-downscales changes how many steps run at which resolution, so ms/step means
    something different. The mark is the only thing stopping it being tabulated as a
    speed result."""
    _floors(tmp_path)
    _write(tmp_path, "B0a", num_downscales=2, ungated=51.0)
    _write(tmp_path, "N4", num_downscales=0, ungated=50.0)
    rows = {r["arm"]: r for r in _grade(tmp_path, ["B0a", "N4"])}
    assert rows["N4"]["schedule_arm"] and not rows["B0a"]["schedule_arm"]
    assert "!" in NT.render(list(rows.values()), "B0a").split("| N4 |")[1].split("\n")[0]


def test_the_time_course_is_carried_through_from_the_per_eval_log(tmp_path):
    """The per-eval `shape` block is the only thing that can say WHEN needles form --
    during the downscaled phases or after them. Dropping it on the floor would leave the
    batch unable to answer the cheapest question it was asked."""
    _floors(tmp_path)
    _write(tmp_path, "B0a", ungated=51.0)
    rows = _grade(tmp_path, ["B0a"])
    assert [s for s, _, _ in rows[0]["shape_course"]] == [2500, 30000]


def _write_ply_shape(out: Path, arm: str, **over):
    (out / f"{arm}.shape.json").write_text(json.dumps({**GOOD_SHAPE, "splats": 500000,
                                                      **over}))


def test_the_PLY_shape_wins_over_the_report_and_the_gap_is_reported(tmp_path):
    """CATCHES the --filter-3d measurement trap. That flag bakes Mip-Splatting's widened
    scales into the exported ply but never touches `p["log_scales"]`, which is what the
    per-eval `shape_metrics` reads. So on exactly the arm whose whole purpose is to widen
    thin splats, the report understates the effect and the ply is the only artifact a
    client actually receives.

    Here the report says the arm is a needle disaster and the ply says it is fine. A
    grader reading the report FAILS it; one reading the ply PASSES it and says why."""
    _floors(tmp_path)
    _write(tmp_path, "B0a", ungated=51.0); _write_ply_shape(tmp_path, "B0a")
    _write(tmp_path, "N1", shape={**GOOD_SHAPE, "needle_frac": 0.40, "aspect_p50": 0.11},
           ungated=50.0)
    _write_ply_shape(tmp_path, "N1", needle_frac=0.02, aspect_p50=0.55)
    rows = {r["arm"]: r for r in _grade(tmp_path, ["B0a", "N1"])}
    r = rows["N1"]
    assert r["shape_source"] == "ply"
    assert r["needle_frac"] == 0.02 and r["aspect_p50"] == 0.55
    assert r["verdict"] == "PASS", r["fails"]
    assert any("graded on the PLY" in n for n in r["notes"]), r["notes"]


def test_shape_read_only_from_the_report_is_flagged_as_incomplete(tmp_path):
    """An arm with no `.shape.json` may still be graded -- but silently grading it from
    the in-memory parameters is how the trap above goes unnoticed on the next batch."""
    _floors(tmp_path)
    _write(tmp_path, "B0a", ungated=51.0)
    rows = {r["arm"]: r for r in _grade(tmp_path, ["B0a"])}
    assert rows["B0a"]["shape_source"] == "report"
    assert any("shape read from the REPORT" in n for n in rows["B0a"]["notes"])


def test_identical_ply_and_report_shapes_raise_NO_disagreement_note(tmp_path):
    """The note must be silent when the two agree, or it fires on all six arms that have
    no filter and nobody reads it on the one that does."""
    _floors(tmp_path)
    _write(tmp_path, "B0a", ungated=51.0); _write_ply_shape(tmp_path, "B0a")
    rows = {r["arm"]: r for r in _grade(tmp_path, ["B0a"])}
    assert rows["B0a"]["shape_disagreement"] is None
    assert not any("graded on the PLY" in n for n in rows["B0a"]["notes"])


def test_collateral_columns_are_reported_in_units_of_the_FLOOR_and_not_folded_into_the_verdict(tmp_path):
    """The pre-registered thin-axis / on-seed bar reads 'within floor of the arm's own
    recipe value'. Taken literally that is a two-sided band of one REPEAT FLOOR --
    0.055 deg -- which no arm that changes anything can meet, a beneficial one included.
    Folding it into PASS/FAIL would therefore fail every arm in the batch and say nothing.

    CATCHES: (i) the literal bar silently deciding the verdict, (ii) a delta reported in
    raw units where 0.055 deg and 0.6 deg look equally small, (iii) an improvement being
    called damage. `thin_axis_p50` is better LOWER and `on_seed_1cm` better HIGHER, and a
    grader that got that backwards would recommend the wrong arm."""
    _floors(tmp_path)
    _write(tmp_path, "B0a", thin=49.0, on_seed=0.0850, ungated=51.0)
    _write_ply_shape(tmp_path, "B0a")
    # thin-axis 20 deg BETTER, on-seed 0.004 better: far outside the floor, all upside.
    _write(tmp_path, "GOOD", thin=29.0, on_seed=0.0890, ungated=31.0)
    _write_ply_shape(tmp_path, "GOOD")
    # thin-axis 3 deg worse, on-seed 0.003 worse.
    _write(tmp_path, "BAD", thin=52.0, on_seed=0.0820, ungated=54.0)
    _write_ply_shape(tmp_path, "BAD")
    rows = {r["arm"]: r for r in _grade(tmp_path, ["B0a", "GOOD", "BAD"])}
    assert rows["GOOD"]["verdict"] == "PASS" and rows["BAD"]["verdict"] == "PASS", (
        "collateral must not decide the verdict")
    g, b = rows["GOOD"]["collateral"], rows["BAD"]["collateral"]
    assert g["thin_axis_p50"]["direction"] == "better" and g["thin_axis_p50"]["no_worse"]
    assert g["on_seed_1cm"]["direction"] == "better" and g["on_seed_1cm"]["no_worse"]
    assert b["thin_axis_p50"]["direction"] == "worse" and not b["thin_axis_p50"]["no_worse"]
    assert b["on_seed_1cm"]["direction"] == "worse" and not b["on_seed_1cm"]["no_worse"]
    # x_floor: -20 deg against a 0.0548 deg floor is -365x, not "-20".
    assert g["thin_axis_p50"]["x_floor"] == pytest.approx(-20.0 / 0.0548, rel=1e-9)
    assert g["thin_axis_p50"]["strict"] == "outside"


def test_a_NONFINITE_arm_is_flagged_and_does_NOT_silently_pass(tmp_path):
    """CATCHES the 2026-09-06 --antialias case. That arm exported 31,158 non-finite
    scale_* values across 2.08% of its splats, and its needle fraction READ AS AN
    IMPROVEMENT -- because `(aspect < 0.1)` is False for NaN, so every contaminated splat
    was counted as a healthy one.

    Integrity is deliberately NOT the bar verdict: 'the flag did not fix needles' and
    'the flag emits NaN scales' are two findings, and one FAIL cannot carry both."""
    _floors(tmp_path)
    _write(tmp_path, "B0a", ungated=51.0)
    _write_ply_shape(tmp_path, "B0a", nonfinite_frac=0.0)
    _write(tmp_path, "N2", ungated=50.0)
    _write_ply_shape(tmp_path, "N2", nonfinite_frac=0.020772)
    rows = {r["arm"]: r for r in _grade(tmp_path, ["B0a", "N2"])}
    assert rows["B0a"]["integrity"] == "clean"
    assert rows["N2"]["integrity"].startswith("NON-FINITE"), rows["N2"]["integrity"]
    assert "2.077%" in rows["N2"]["integrity"], rows["N2"]["integrity"]
    assert any("filter-nan" in n for n in rows["N2"]["notes"]), rows["N2"]["notes"]
    # the bars still graded on their own merits -- the arm is clean on those
    assert rows["N2"]["verdict"] == "PASS", rows["N2"]["fails"]
    assert "NON-FINITE" in NT.render(list(rows.values()), "B0a")


def test_an_arm_scored_by_a_binary_with_NO_integrity_check_says_UNKNOWN(tmp_path):
    """Absence of the field must not read as 'clean'. Every report written before
    2026-09-06 lacks it, and those shape columns really may be over a contaminated
    population -- the MPS `median` would not have said so."""
    _floors(tmp_path)
    _write(tmp_path, "B0a", ungated=51.0)      # report-only shape, no nonfinite_frac
    rows = {r["arm"]: r for r in _grade(tmp_path, ["B0a"])}
    assert rows["B0a"]["integrity"] == "UNKNOWN"
    assert any("predating the check" in n for n in rows["B0a"]["notes"])


def test_the_disagreement_note_does_not_blame_a_flag_the_arm_never_used(tmp_path):
    """CATCHES a note that gives a CONFIDENT WRONG REASON, which is worse than no reason.

    The first version asserted "--filter-3d bakes the widened scales into the export" on
    every ply-vs-report disagreement, and it duly fired on the --antialias arm, whose gap
    has a completely different cause: 2.08% of its splats are non-finite and the two
    readings exclude them differently. A reader following that note would have gone
    looking for a filter that was never switched on."""
    _floors(tmp_path)
    _write(tmp_path, "B0a", ungated=51.0); _write_ply_shape(tmp_path, "B0a")
    _write(tmp_path, "N1", filter_3d=True, ungated=50.0)
    _write_ply_shape(tmp_path, "N1", needle_frac=0.03)
    _write(tmp_path, "N2", filter_3d=False, ungated=50.0)
    _write_ply_shape(tmp_path, "N2", needle_frac=0.03)
    rows = {r["arm"]: r for r in _grade(tmp_path, ["B0a", "N1", "N2"])}
    n1 = " ".join(rows["N1"]["notes"]); n2 = " ".join(rows["N2"]["notes"])
    assert "--filter-3d bakes" in n1, n1
    assert "does not use --filter-3d" in n2, n2
    assert "non-finite population" in n2, n2
