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
           on_seed=0.085, ungated=None, num_downscales=2, lpips=0.39):
    (out / f"{arm}.json").write_text(json.dumps({
        "env": {"git": "abc1234", "dirty": False},
        "resolved": {"num_downscales": num_downscales},
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
