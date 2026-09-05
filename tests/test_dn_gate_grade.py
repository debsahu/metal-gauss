"""Task 20 escalation GRADING -- the battery, the three bands, the anchors, the probe.

Each test says what it CATCHES; `tests/mutants_dn_gate_grade.py` asserts on these names.

Everything here is a pure function of the artifacts, which is the claim `--regrade` exists
to prove, so none of it needs a GPU or a scene. Two things are being defended:

  * the RULE (`3cfd8f3`'s three bands) is applied as written, including the two halves
    that are easy to lose -- the cumulative check, and DROP being checked first;
  * READING B (the early-divergence probe) can never return a verdict. It is diagnostic by
    pre-registration, and the temptation to promote an early difference is exactly what
    writing that down in advance was meant to forestall.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import dn_gate_arms as H  # noqa: E402


# --------------------------------------------------------------------------- fixtures

BASE = {"run.psnr_masked": 25.0, "run.psnr": 22.0, "run.coverage": 0.95,
        "run.lpips": 0.40, "run.ms_per_step": 113.0, "run.n_splats": 500000,
        "run.aspect_p50": 0.3126, "run.needle_frac": 0.1516,
        "run.hard_needle_frac": 0.00207, "run.smid_p50_mm": 7.15,
        "run.smax_p50_mm": 25.84,
        "stats.on_seed_frac_1cm": 0.0719, "stats.on_seed_frac_2cm": 0.24,
        "stats.thin_axis_angle_p50": 40.0, "stats.opacity_p50": 0.20}

FLOOR_COUNTS = {"fused_calls": 0, "torch_calls": 9, "dn_gated_calls": 0,
                "dn_ungated_calls": 9, "dn_skipped_calls": 0}
TREAT_COUNTS = {"fused_calls": 0, "torch_calls": 9, "dn_gated_calls": 9,
                "dn_ungated_calls": 0, "dn_skipped_calls": 0}


def write_arm(out: Path, tag: str, values: dict, seed: int, role: str,
              seed_cloud: str, colmap: str, resolved_extra: dict | None = None) -> None:
    """One arm's two artifacts, shaped exactly as the trainer and splatstats write them."""
    v = dict(BASE) | values
    resolved = {"colmap": colmap, "images": colmap + "/../images", "init_ply": None,
                "seed": seed, "budget": 500_000, "steps": 30_000,
                "max_resolution": 1920, "num_downscales": 0,
                "depth_normal_weight": 0.05, "depth_loss_space": "disparity",
                "flatten_loss_weight": 1.0, "depth_loss_weight": 1.0,
                "normal_loss_weight": 0.2, "export_every": 500,
                "masks": None, "mask_polarity": "drop"} | (resolved_extra or {})
    (out / f"{tag}.json").write_text(json.dumps({
        "schema": 1, "resolved": resolved, "env": {"git": "deadbee"},
        "observed": {"schema": 1,
                     "loss_path": FLOOR_COUNTS if role == "floor" else TREAT_COUNTS},
        "metrics": {
            "psnr": v["run.psnr"], "psnr_masked": v["run.psnr_masked"],
            "coverage": v["run.coverage"], "lpips": v["run.lpips"],
            "ms_per_step": v["run.ms_per_step"], "n_splats": v["run.n_splats"],
            "shape": {"aspect_p50": v["run.aspect_p50"],
                      "needle_frac": v["run.needle_frac"],
                      "hard_needle_frac": v["run.hard_needle_frac"],
                      "smid_p50_mm": v["run.smid_p50_mm"],
                      "smax_p50_mm": v["run.smax_p50_mm"]}}}, indent=2))
    (out / f"{tag}.stats.json").write_text(json.dumps({
        "seed_cloud": seed_cloud, "thin_axis_evaluated": 400000.0,
        "metrics": {k[len("stats."):]: v[k] for k in v if k.startswith("stats.")}},
        indent=2))


def make_scene(tmp: Path, name: str = "pgeom", treatment: dict | None = None,
               floors: tuple[dict, dict, dict] = ({}, {}, {}),
               seed_cloud_name: str = "points3D.tsdf.txt") -> Path:
    out = tmp / name
    out.mkdir(parents=True, exist_ok=True)
    colmap = tmp / "ds" / "sparse" / "0"
    colmap.mkdir(parents=True, exist_ok=True)
    (colmap / "points3D.bin").write_bytes(b"\0")
    ref = colmap / seed_cloud_name
    ref.write_text("")
    for tag, seed, extra in (("F0", 42, floors[0]), ("F1", 42, floors[1]),
                             ("F2", 43, floors[2])):
        write_arm(out, tag, extra, seed, "floor", str(ref), str(colmap))
    write_arm(out, "G0", treatment or {}, 42, "treatment", str(ref), str(colmap))
    return out


def graded(tmp: Path, **kw) -> dict:
    """Floors written, then the treatment scored and graded -- in that order, by the
    harness, which is the phase ordering the whole protocol rests on."""
    out = make_scene(tmp, **{k: v for k, v in kw.items() if k != "anchor"})
    H.write_floors(out, "pgeom", 0.05, "disparity")
    return H.grade_scene(out, "pgeom", 0.05, "G0", kw.get("anchor") or H.ANCHOR_PATH)


# ------------------------------------------------------- the battery and the seed cloud


def test_the_battery_refuses_a_reference_cloud_THE_ARM_ITSELF_seeded_from(tmp_path):
    """CATCHES the seed-cloud hole at the point it actually bites -- grading. The runner
    checks the COMMAND LINE; this checks the arm's own `resolved`, so an arm run by some
    other invocation, or re-graded later from committed artifacts alone, is checked
    against what IT seeded from rather than against what this invocation was told."""
    out = make_scene(tmp_path, seed_cloud_name="points3D.bin")
    with pytest.raises(SystemExit, match="the trainer initialised from"):
        H.battery(out, "F0")


def test_the_battery_reads_the_seed_from_the_ARMS_OWN_resolved(tmp_path):
    """Discriminating power for the test above, and the reason it is not just the runner's
    check again: with `init_ply` set the COLMAP cloud is no longer the seed, and the
    battery must learn that from the REPORT rather than from an argument nobody passed to
    a re-grade."""
    out = make_scene(tmp_path)
    b = H.battery(out, "F0")
    assert b["values"]["run.aspect_p50"] == BASE["run.aspect_p50"]
    assert b["values"]["stats.on_seed_frac_1cm"] == BASE["stats.on_seed_frac_1cm"]


def test_the_battery_re_asserts_every_arms_OBSERVED_LOSS_PATH(tmp_path):
    """CATCHES a re-grade of artifacts the runner never checked. `--regrade` is meant to
    reproduce a verdict from the repository alone, which means it cannot assume the arms
    came from this harness -- so the role assertion is made again, from the report."""
    out = make_scene(tmp_path)
    rep = json.loads((out / "F0.json").read_text())
    rep["observed"]["loss_path"] = dict(TREAT_COUNTS)      # a gated "floor"
    (out / "F0.json").write_text(json.dumps(rep))
    with pytest.raises(SystemExit, match="dn_gated_calls"):
        H.battery(out, "F0")


# ------------------------------------------------------------------------- floors


def test_the_floor_is_the_n3_SPREAD_and_the_pair_difference_is_reported_only(tmp_path):
    """CATCHES an n=2 floor. Section 8.2 is this project's record of |F0-F1| coming out
    25-45x too small and taking a day's conclusions with it, so the pair difference is
    recorded and never graded against."""
    out = make_scene(tmp_path, floors=({"run.needle_frac": 0.150},
                                       {"run.needle_frac": 0.151},
                                       {"run.needle_frac": 0.160}))
    fl = H.write_floors(out, "pgeom", 0.05, "disparity")["floors"]["run.needle_frac"]
    assert fl["spread_n3"] == pytest.approx(0.010)
    assert fl["repeat_pair_abs_diff"] == pytest.approx(0.001)
    assert fl["mean"] == pytest.approx((0.150 + 0.151 + 0.160) / 3)


def test_floor_arms_that_differ_in_a_CONFIGURATION_FLAG_are_not_a_floor(tmp_path):
    """CATCHES three runs that are not a repeat measurement of anything. Their spread is
    then a treatment effect wearing a noise floor's name."""
    out = make_scene(tmp_path)
    rep = json.loads((out / "F2.json").read_text())
    rep["resolved"]["depth_normal_weight"] = 0.0
    (out / "F2.json").write_text(json.dumps(rep))
    with pytest.raises(SystemExit, match="differ in configuration"):
        H.write_floors(out, "pgeom", 0.05, "disparity")


def test_the_floor_arms_must_share_ONE_reference_cloud(tmp_path):
    """CATCHES a floor spread computed across two different on-seed references, which is
    not a spread of anything -- on-seed is only comparable with the reference held fixed."""
    out = make_scene(tmp_path)
    st = json.loads((out / "F1.stats.json").read_text())
    st["seed_cloud"] = "/elsewhere/other_tsdf.txt"
    (out / "F1.stats.json").write_text(json.dumps(st))
    with pytest.raises(SystemExit, match="reference cloud"):
        H.write_floors(out, "pgeom", 0.05, "disparity")


def test_grading_REFUSES_until_the_floors_have_been_written(tmp_path):
    """CATCHES the phase order being left to operator discipline. Nobody may choose a
    floor after seeing a treatment number, and the only way to guarantee that is to make
    the treatment ungradeable until floors.json exists."""
    out = make_scene(tmp_path)
    with pytest.raises(SystemExit, match="floors"):
        H.grade_scene(out, "pgeom", 0.05, "G0", H.ANCHOR_PATH)


def test_a_floor_REBUILD_that_disagrees_with_the_written_floors_is_refused(tmp_path):
    """CATCHES the artifacts moving under a frozen record. floors.json was written under
    the phase order that makes the protocol trustworthy; recomputing it from the same
    reports must reproduce it exactly, and adding a column must never be a way to quietly
    re-measure a floor."""
    with pytest.raises(SystemExit, match="disagree"):
        H.merge_extended_floors({"m": {"F0": 1.0, "F1": 1.0, "F2": 1.0, "mean": 1.0,
                                       "spread_n3": 0.0}},
                                {"m": {"F0": 1.0, "F1": 1.0, "F2": 2.0, "mean": 1.33,
                                       "spread_n3": 1.0}})


# --------------------------------------- Band 1 thresholds and their scene dependence


def test_the_ABSOLUTE_Band1_thresholds_are_reported_relative_to_the_SCENES_OWN_baseline():
    """CATCHES quoting `+10.8 pp` as a constant across scenes. It is 71% of P-GEOM's own
    0.1516 needle baseline and ~46% of P-MASK's 0.2348 -- the SAME threshold meaning two
    very different things. Both figures are re-derived here so the grade cannot report one
    scene's relative size beside another scene's number."""
    spec = H.COLLAPSE["run.needle_frac"]
    assert H.threshold_relative(spec, 0.15156066914399466) == pytest.approx(0.7126, abs=1e-4)
    assert H.threshold_relative(spec, 0.2348) == pytest.approx(0.4600, abs=1e-4)


def test_the_LOG_thresholds_transfer_between_scenes_and_the_ABSOLUTE_ones_do_not():
    """CATCHES the two halves being described the same way. A log threshold is a ratio and
    is the same relative change on any baseline; an absolute one is not. `transfers` is
    DERIVED from the space rather than hand-set, so the two cannot drift apart."""
    for col, spec in H.COLLAPSE.items():
        assert H.transfers_between_scenes(spec) == (spec["space"] == "log"), col
    log = H.COLLAPSE["run.aspect_p50"]
    assert H.threshold_relative(log, 0.31) == pytest.approx(H.threshold_relative(log, 0.99))
    assert H.threshold_relative(log, 0.31) == pytest.approx(0.2924, abs=1e-4)
    assert H.threshold_relative(H.COLLAPSE["stats.on_seed_frac_1cm"], 0.07) \
        == pytest.approx(0.1689, abs=1e-4)
    absolute = H.COLLAPSE["run.needle_frac"]
    assert H.threshold_relative(absolute, 0.15) != pytest.approx(
        H.threshold_relative(absolute, 0.99))


def test_the_grade_carries_the_scene_baseline_and_the_relative_size_beside_every_verdict(
        tmp_path):
    """CATCHES a grade that reports a Band 1 verdict with no way to see what the threshold
    meant on THIS scene. Pre-registration section 5: every P-MASK grading must state the
    scene's own baseline beside the threshold rather than applying a constant derived on
    the other scene in silence."""
    g = graded(tmp_path)
    for col, row in g["band1"]["per_arm"].items():
        assert "scene_baseline" in row and "threshold_relative_to_baseline" in row
        assert "threshold_transfers_between_scenes" in row


def test_collapse_delta_is_POSITIVE_TOWARD_WORSE_in_both_spaces():
    """CATCHES a sign error, which would invert every Band 1 test: an arm that HALVED
    on-seed would read as a large improvement and no collapse could ever fire."""
    assert H.collapse_delta("run.needle_frac", 0.30, 0.10) > 0        # needles UP = worse
    assert H.collapse_delta("run.needle_frac", 0.05, 0.10) < 0
    assert H.collapse_delta("stats.on_seed_frac_1cm", 0.05, 0.10) > 0  # on-seed DOWN = worse
    assert H.collapse_delta("stats.on_seed_frac_1cm", 0.20, 0.10) < 0


# ------------------------------------------------------------------------ the bands


def test_band1_fires_on_a_COLLAPSE_and_a_2_point_5_percent_drift_is_NOT_one(tmp_path):
    """CATCHES the magnitude-blind rule the amendment replaced. Task 19's move (aspect
    -2.5%, needles +0.6 pp) and the VOID row (aspect -78%, needles +40 pp) differ by ~35x
    and the old rule returned the same word for both."""
    drift = graded(tmp_path, treatment={"run.aspect_p50": 0.3126 * 0.975,
                                        "run.needle_frac": 0.1516 + 0.006})
    assert not drift["band1_fired"]
    collapse = graded(tmp_path, treatment={"run.aspect_p50": 0.0659,
                                           "run.needle_frac": 0.5679})
    assert collapse["band1_fired"] and collapse["scene_drop"]


def test_band1_fires_CUMULATIVELY_even_when_the_per_arm_check_does_not(tmp_path):
    """CATCHES the rule ratcheting. Four accepted 8 pp needle drifts are a 32 pp collapse
    that no single arm ever fired on, which is the whole reason the cumulative half exists.
    Here the floors have already drifted most of the way and the arm adds the rest."""
    drifted = {"run.needle_frac": 0.24}
    g = graded(tmp_path, floors=(drifted, drifted, drifted),
               treatment={"run.needle_frac": 0.27})
    assert not g["band1"]["per_arm_fired"]
    assert "run.needle_frac" in g["band1"]["cumulative_fired"]
    assert g["band1_fired"] and g["scene_drop"]


def test_a_MISSING_Band1_column_is_refused_rather_than_reading_as_no_collapse(tmp_path):
    """CATCHES the failure shape CLAUDE.md names as the one this project keeps repeating:
    a check that reads a condition something other than the thing being checked could
    satisfy. A collapse column that was never measured must never read as `did not
    collapse`."""
    out = make_scene(tmp_path)
    H.write_floors(out, "pgeom", 0.05, "disparity")
    rep = json.loads((out / "G0.json").read_text())
    del rep["metrics"]["lpips"]
    (out / "G0.json").write_text(json.dumps(rep))
    with pytest.raises(SystemExit, match="run.lpips"):
        H.grade_scene(out, "pgeom", 0.05, "G0", H.ANCHOR_PATH)


def test_band2_FAILS_on_a_worsened_column_and_an_ABSENT_column_is_refused():
    """CATCHES an absent gate column reading as a pass, and catches Band 2 having silently
    kept aspect and needles -- moving those to Band 1, where magnitude decides, IS the
    amendment."""
    assert H.band2({"stats.on_seed_frac_1cm": "IMPROVED",
                    "stats.thin_axis_angle_p50": "WITHIN FLOOR"}) == "PASS"
    assert H.band2({"stats.on_seed_frac_1cm": "WITHIN FLOOR",
                    "stats.thin_axis_angle_p50": "WORSENED"}) == "FAIL"
    assert H.band2({"stats.on_seed_frac_1cm": "WITHIN FLOOR",
                    "stats.thin_axis_angle_p50": "WITHIN FLOOR"}) == "WITHIN FLOOR"
    with pytest.raises(SystemExit, match="thin_axis"):
        H.band2({"stats.on_seed_frac_1cm": "IMPROVED"})
    assert set(H.BAND2_GATE) == {"stats.on_seed_frac_1cm", "stats.thin_axis_angle_p50"}


def test_band3_is_ONE_SIDED_a_PSNR_GAIN_is_not_a_regression():
    """CATCHES restoring the old two-sided `must be WITHIN floor` condition, which is what
    made every Tier 3 arm unable to PASS whatever its geometry did."""
    assert not H.band3(30.0, 25.0)["fired"]
    assert not H.band3(24.80, 25.0)["fired"]        # 0.20 dB loss, inside the allowance
    assert H.band3(24.70, 25.0)["fired"]            # 0.30 dB loss
    assert not H.band3(25.0 - H.PSNR_DROP_DB, 25.0)["fired"], "the comparison is STRICT"


def test_band3_fires_on_CROSSING_the_24dB_Stage4_gate_from_above():
    """CATCHES losing the second clause. A 0.1 dB loss is inside the allowance and is
    still a product regression if it takes the scene under the delivery gate."""
    r = H.band3(23.95, 24.05)
    assert r["fired"] and r["crossed_stage4_gate"] and not r["exceeds_allowance"]
    assert not H.band3(23.4, 23.5)["fired"], \
        "a scene already under the gate cannot CROSS it; a 0.1 dB loss is inside "\
        "the allowance, so nothing fires"


def test_DRIFT_excludes_improvements_and_excludes_a_Band1_firing(tmp_path):
    """CATCHES two definition errors at once. Counting improvements makes KEEP AS DEFAULT
    unreachable, because Band 2 REQUIRES on-seed to improve beyond its floor; and
    reporting a Band 1 firing as drift describes a hard DROP as adoptable-with-caveats."""
    g = graded(tmp_path, treatment={"stats.on_seed_frac_1cm": 0.20,
                                    "run.needle_frac": 0.1516 + 0.02})
    metrics = {d["metric"] for d in g["drift"]}
    assert "stats.on_seed_frac_1cm" not in metrics, "an improvement is not drift"
    assert "run.needle_frac" in metrics
    collapse = graded(tmp_path, treatment={"run.needle_frac": 0.5679})
    assert "run.needle_frac" not in {d["metric"] for d in collapse["drift"]}


def test_a_drift_column_that_CAUSED_the_scene_to_fail_is_flagged_as_such(tmp_path):
    """CATCHES a list whose entries mean `adoptable with caveats` and `this is the DROP`
    at the same time -- aimed at a human reader, who is the one the rule's output is for."""
    g = graded(tmp_path, treatment={"run.psnr_masked": 25.0 - 0.30})
    d = [x for x in g["drift"] if x["metric"] == "run.psnr_masked"]
    assert d and d[0]["caused_band3_fire"] is True


# ------------------------------------------------------------------------ the anchor


def test_a_scene_with_NO_anchor_is_an_error_not_a_vacuous_cumulative_check(tmp_path):
    """CATCHES an empty anchor making every cumulative check pass while still writing a
    well-formed grade."""
    out = make_scene(tmp_path, name="nosuchscene")
    H.write_floors(out, "nosuchscene", 0.05, "disparity")
    with pytest.raises(SystemExit, match="no frozen"):
        H.grade_scene(out, "nosuchscene", 0.05, "G0", H.ANCHOR_PATH)


def test_the_self_anchor_is_written_WITH_the_floors_and_before_the_treatment_is_scored(
        tmp_path):
    """CATCHES a self-anchor chosen after a treatment number exists -- which would make it
    an anchor picked to produce a verdict."""
    out = make_scene(tmp_path, name="pmask")
    assert not (out / "anchor.json").exists()
    H.write_floors(out, "pmask", 0.05, "disparity", self_anchor=True)
    a = json.loads((out / "anchor.json").read_text())
    assert a["self_anchored"] is True
    assert set(a["values"]) == set(H.COLLAPSE)


def test_a_self_anchored_scenes_cumulative_check_is_VACUOUS_AND_THE_GRADE_SAYS_SO(tmp_path):
    """CATCHES a vacuous check reading as a passing one. Pre-registration section 5 states
    the consequence in advance rather than leaving it to be discovered: the anchor IS the
    floor mean, so the cumulative delta equals the per-arm delta exactly."""
    out = make_scene(tmp_path, name="pmask")
    H.write_floors(out, "pmask", 0.05, "disparity", self_anchor=True)
    g = H.grade_scene(out, "pmask", 0.05, "G0", H.ANCHOR_PATH)
    assert g["band1"]["cumulative_check_vacuous"] is True
    assert "VACUOUS" in g["band1"]["cumulative_note"]
    for col in H.COLLAPSE:
        assert g["band1"]["cumulative"][col]["delta"] == pytest.approx(
            g["band1"]["per_arm"][col]["delta"])


def test_vacuity_is_MEASURED_not_read_off_the_self_anchored_flag(tmp_path):
    """CATCHES trusting the flag. On the SECOND arm of a self-anchored scene the anchor is
    frozen while the floors have been re-measured, so the check stops being vacuous -- and
    a harness that reported vacuity from `self_anchored` would keep saying so forever,
    exactly when the check starts mattering."""
    out = make_scene(tmp_path, name="pmask")
    H.write_floors(out, "pmask", 0.05, "disparity", self_anchor=True)
    for tag, extra in (("F0", {}), ("F1", {}), ("F2", {})):
        write_arm(out, tag, {"run.needle_frac": 0.20} | extra, 42 if tag != "F2" else 43,
                  "floor", str(tmp_path / "ds" / "sparse" / "0" / "points3D.tsdf.txt"),
                  str(tmp_path / "ds" / "sparse" / "0"))
    (out / "floors.json").unlink()
    (out / "FLOORS_DONE").unlink()
    H.write_floors(out, "pmask", 0.05, "disparity", self_anchor=True)
    g = H.grade_scene(out, "pmask", 0.05, "G0", H.ANCHOR_PATH)
    assert g["anchor"]["self_anchored"] is True
    assert g["band1"]["cumulative_check_vacuous"] is False


def test_an_anchor_measured_at_ANOTHER_configuration_is_refused(tmp_path):
    """CATCHES grading a ratchet against a fiction. `steps` and `num_downscales` are
    checked as well as budget and resolution: both change what a 30k arm's geometry
    columns settle at, and neither is named in the anchor's own re-measure sentence, which
    is exactly why they are the two that would slip through."""
    out = make_scene(tmp_path)
    H.write_floors(out, "pgeom", 0.05, "disparity")
    for tag in ("F0", "F1", "F2", "G0"):
        rep = json.loads((out / f"{tag}.json").read_text())
        rep["resolved"]["num_downscales"] = 2
        (out / f"{tag}.json").write_text(json.dumps(rep))
    with pytest.raises(SystemExit, match="num_downscales"):
        H.grade_scene(out, "pgeom", 0.05, "G0", H.ANCHOR_PATH)


def test_the_cumulative_delta_is_reported_DECOMPOSED_into_gate_and_everything_else(
        tmp_path):
    """CATCHES a cumulative firing being read as a treatment effect. Pre-registration
    section 5: the frozen P-GEOM anchor differs from these arms in four ways that are NOT
    the gate (dn, loss chain, --export-every, and the MACHINE), so (F-mean - anchor) and
    (G0 - F-mean) must be reported separately."""
    g = graded(tmp_path)
    d = g["band1"]["decomposition"]
    for col in H.COLLAPSE:
        assert d[col]["everything_but_the_gate"] + d[col]["the_gate"] == pytest.approx(
            g["band1"]["cumulative"][col]["delta"])


# ------------------------------------------------------------ the cross-scene verdict


def _scene_verdict(**kw):
    return {"band1_fired": False, "band2": "PASS", "band3_fired": False,
            "drift": [], "dn": 0.05, "scene_pass": True, "scene_drop": False,
            "falsifier_triggered_on_this_scene": False} | kw


def test_DROP_is_checked_FIRST_and_is_not_overridable():
    """CATCHES an implementation that reaches an opt-in branch first. A pass on one scene
    and a collapse on another is not an opt-in; it is a DROP."""
    v = H.combined_verdict({"pgeom": _scene_verdict(),
                            "pmask": _scene_verdict(band1_fired=True, scene_pass=False)})
    assert v["decision"] == "DROP" and v["regressed_on"] == ["pmask"]


def test_the_grader_can_reach_EVERY_ONE_of_its_verdicts():
    """CATCHES a rule with an unreachable branch, which is a broken rule. Also catches
    drift silently promoting to KEEP: Band 2 REQUIRES on-seed to improve beyond floor, so
    if drift counted improvements KEEP AS DEFAULT could never be reached at all."""
    both = lambda **k: {"pgeom": _scene_verdict(**k), "pmask": _scene_verdict(**k)}
    assert H.combined_verdict(both())["decision"] == "KEEP AS DEFAULT"
    assert H.combined_verdict(both(drift=[{"metric": "run.needle_frac"}]))["decision"] \
        == "OPT-IN, DEFAULT-CANDIDATE"
    assert H.combined_verdict({"pgeom": _scene_verdict(),
                               "pmask": _scene_verdict(band2="WITHIN FLOOR",
                                                       scene_pass=False)})["decision"] \
        == "OPT-IN"
    assert H.combined_verdict(both(band2="WITHIN FLOOR", scene_pass=False))["decision"] \
        .startswith("NOT ADOPTED")
    assert H.combined_verdict(both(band1_fired=True))["decision"] == "DROP"


def test_a_NULL_RESULT_is_reported_as_a_null_and_not_as_an_adoption():
    """CATCHES reading 3 of `3cfd8f3`, pinned before any number existed: failing Band 2
    while triggering neither Band 1 nor Band 3 is NOT ADOPTED and the shipped default
    stands. For THIS arm that is the most likely outcome -- removing pixels from a loss
    has no prior reason to raise on-seed@1cm."""
    v = H.combined_verdict({"pgeom": _scene_verdict(band2="FAIL", scene_pass=False,
                                                    scene_drop=True)})
    assert v["decision"] == "DROP"
    v2 = H.combined_verdict({"pgeom": _scene_verdict(band2="WITHIN FLOOR",
                                                     scene_pass=False)})
    assert v2["decision"].startswith("NOT ADOPTED")


def test_summary_refuses_an_UNNAMED_grade_in_the_tree(tmp_path):
    """CATCHES the near-miss that made `collect_scenes` exist: a glob over --out would
    count the grader's OWN synthetic test fixtures as measured scenes, and because DROP is
    checked first and is not overridable, a fabricated regression would have produced a
    DROP indistinguishable from a measured one."""
    for n in ("pgeom", "smoke_fixture"):
        (tmp_path / n).mkdir()
        (tmp_path / n / "grade.json").write_text(json.dumps({"scene": n}))
    with pytest.raises(SystemExit, match="unnamed"):
        H.collect_scenes(tmp_path, "pgeom")


def test_summary_refuses_a_NAMED_scene_that_has_no_grade(tmp_path):
    """CATCHES a cross-scene decision computed from one scene while claiming two."""
    (tmp_path / "pgeom").mkdir()
    (tmp_path / "pgeom" / "grade.json").write_text(json.dumps({"scene": "pgeom"}))
    with pytest.raises(SystemExit, match="pmask"):
        H.collect_scenes(tmp_path, "pgeom,pmask")


def test_a_grade_whose_scene_field_disagrees_with_its_directory_is_refused(tmp_path):
    """CATCHES trusting a directory name. The directory is not evidence about what was
    measured; the report is."""
    (tmp_path / "pgeom").mkdir()
    (tmp_path / "pgeom" / "grade.json").write_text(json.dumps({"scene": "pmask"}))
    with pytest.raises(SystemExit, match="disagree"):
        H.collect_scenes(tmp_path, "pgeom")


def test_a_SECOND_arms_grade_never_overwrites_the_primary_scene_verdict(tmp_path):
    """CATCHES a defect the plane-aux branch shipped once: `--regrade --treatment-tag M0`
    wrote `grade.json` unconditionally, so a second arm's verdict silently replaced the
    pre-registered arm's -- and `--summary` reads `grade.json`. Both files are well-formed
    grades of real arms, so nothing errored. Here G0 is the pre-registered treatment and
    the conditional 9th arm G1 must not be able to take its filename."""
    assert H.grade_filename("G0") == "grade.json"
    assert H.grade_filename("G1") == "grade_G1.json"
    H.write_grade(tmp_path, "G1", {"scene": "pmask"})
    assert not (tmp_path / "grade.json").exists()
    H.write_grade(tmp_path, "G0", {"scene": "pmask"})
    assert (tmp_path / "grade.json").exists()


# --------------------------------------------------- Reading B: the divergence probe


def _checkpoints(out: Path, per_arm: dict[str, dict[int, dict]]) -> None:
    """Stand in for `bench/ply_shape` + splatstats over each checkpoint ply, so the probe's
    ARITHMETIC is tested without a ply reader in the loop."""
    for tag, steps in per_arm.items():
        for step, cols in steps.items():
            (out / f"{tag}.step{step:06d}.cols.json").write_text(json.dumps(cols))


@pytest.fixture
def probe_scene(tmp_path, monkeypatch):
    # NON-DEGENERATE floors on purpose: with all three floor arms identical every
    # spread_n3 is 0, and `effect / 30k floor` would be infinite for every column --
    # which would let the zero-noise test below pass without exercising anything.
    out = make_scene(tmp_path, floors=({"run.needle_frac": 0.150},
                                       {"run.needle_frac": 0.151},
                                       {"run.needle_frac": 0.160}))
    H.write_floors(out, "pgeom", 0.05, "disparity")
    monkeypatch.setattr(H, "checkpoint_columns",
                        lambda o, tag, step, seed_cloud: json.loads(
                            (o / f"{tag}.step{step:06d}.cols.json").read_text()))
    return out


def test_the_probe_reads_EXACTLY_the_pre_registered_steps_and_records_which(probe_scene):
    """CATCHES reselecting the checkpoints after seeing them. 500 and 2000 are named in
    the pre-registration -- 500 brackets the 21x transient, 2000 sits inside the flat
    regime -- and they are a CONSTANT here, not a flag. Recorded in the output so a reader
    never has to trust that."""
    cols = {"aspect_p50": 0.3, "needle_frac": 0.15, "hard_needle_frac": 0.002,
            "smid_p50_mm": 7.0, "smax_p50_mm": 25.0, "splats": 300000,
            "on_seed_frac_1cm": 0.07, "thin_axis_angle_p50": 41.0}
    _checkpoints(probe_scene, {t: {s: cols for s in (500, 2000)}
                               for t in ("F0", "F1", "G0")})
    r = H.early_divergence(probe_scene, "pgeom")
    assert H.GRADED_EARLY_STEPS == (500, 2000)
    assert r["steps_read"] == [500, 2000]


def test_the_probe_reports_EFFECT_NOISE_and_their_ratio_per_column(probe_scene):
    """CATCHES a probe that reports only the effect. `G0 - F0` alone says nothing without
    `F1 - F0` beside it: at 500 steps two same-seed runs may differ by almost nothing, so
    a `beyond floor` test there would fire on differences of no consequence."""
    def cols(**kw):
        return {"aspect_p50": 0.30, "needle_frac": 0.15, "hard_needle_frac": 0.002,
                "smid_p50_mm": 7.0, "smax_p50_mm": 25.0, "splats": 300000,
                "on_seed_frac_1cm": 0.07, "thin_axis_angle_p50": 41.0} | kw
    _checkpoints(probe_scene, {
        "F0": {s: cols() for s in (500, 2000)},
        "F1": {s: cols(needle_frac=0.16) for s in (500, 2000)},
        "G0": {s: cols(needle_frac=0.19) for s in (500, 2000)}})
    row = H.early_divergence(probe_scene, "pgeom")["steps"]["500"]["needle_frac"]
    assert row["effect_G0_minus_F0"] == pytest.approx(0.04)
    assert row["noise_F1_minus_F0"] == pytest.approx(0.01)
    assert row["effect_over_noise"] == pytest.approx(4.0)


def test_ZERO_NOISE_is_never_reported_as_an_infinite_ratio(probe_scene):
    """CATCHES the degenerate case the pre-registration handles in advance. F0 and F1
    share a seed, and over 500 steps the atomics divergence may not have reached the
    precision at which a ply prints. An undefined ratio must be reported as an absolute
    effect and against the 30k floor -- which is itself a finding about the probe's
    sensitivity, not a missing number."""
    def cols(**kw):
        return {"aspect_p50": 0.30, "needle_frac": 0.15, "hard_needle_frac": 0.002,
                "smid_p50_mm": 7.0, "smax_p50_mm": 25.0, "splats": 300000,
                "on_seed_frac_1cm": 0.07, "thin_axis_angle_p50": 41.0} | kw
    _checkpoints(probe_scene, {"F0": {s: cols() for s in (500, 2000)},
                               "F1": {s: cols() for s in (500, 2000)},
                               "G0": {s: cols(needle_frac=0.19) for s in (500, 2000)}})
    row = H.early_divergence(probe_scene, "pgeom")["steps"]["500"]["needle_frac"]
    assert row["noise_F1_minus_F0"] == 0.0
    assert row["effect_over_noise"] is None
    assert not math.isinf(row["effect_over_30k_floor"])
    assert row["effect_over_30k_floor"] is not None
    assert "below the ply" in row["note"]


def test_the_probe_EMITS_NO_VERDICT_ANYWHERE_IN_ITS_OUTPUT(probe_scene):
    """CATCHES the single thing the pre-registration wrote down in advance BECAUSE it
    would otherwise be tempting: `READING B MAY NEVER PRODUCE A KEEP, AN OPT-IN OR A
    DROP`. Searched over the whole serialised document, keys and values, so a verdict
    cannot arrive under a new key name."""
    cols = {"aspect_p50": 0.3, "needle_frac": 0.15, "hard_needle_frac": 0.002,
            "smid_p50_mm": 7.0, "smax_p50_mm": 25.0, "splats": 300000,
            "on_seed_frac_1cm": 0.07, "thin_axis_angle_p50": 41.0}
    _checkpoints(probe_scene, {t: {s: cols for s in (500, 2000)}
                               for t in ("F0", "F1", "G0")})
    doc = json.dumps(H.early_divergence(probe_scene, "pgeom"))
    for banned in ("KEEP", "OPT-IN", "DROP", "PASS", "FAIL", "verdict", "decision"):
        assert banned not in doc, f"Reading B emitted {banned!r}"
    assert "DIAGNOSTIC" in H.early_divergence(probe_scene, "pgeom")["reading"]


def test_the_probe_covers_the_five_shape_columns_the_splat_count_and_the_Band2_pair(
        probe_scene):
    """CATCHES a probe that could not see the decision column. Including the Band 2 pair
    was decided in the pre-registration BEFORE any arm ran, precisely so it could not be
    added or dropped after seeing a number."""
    cols = {"aspect_p50": 0.3, "needle_frac": 0.15, "hard_needle_frac": 0.002,
            "smid_p50_mm": 7.0, "smax_p50_mm": 25.0, "splats": 300000,
            "on_seed_frac_1cm": 0.07, "thin_axis_angle_p50": 41.0}
    _checkpoints(probe_scene, {t: {s: cols for s in (500, 2000)}
                               for t in ("F0", "F1", "G0")})
    got = set(H.early_divergence(probe_scene, "pgeom")["steps"]["500"])
    assert got == {"aspect_p50", "needle_frac", "hard_needle_frac", "smid_p50_mm",
                   "smax_p50_mm", "splats", "on_seed_frac_1cm", "thin_axis_angle_p50"}


def test_a_MISSING_checkpoint_is_refused_rather_than_dropping_the_column(probe_scene):
    """CATCHES a probe that silently reports on two arms instead of three. `G0 - F0` with
    F0 absent is not a smaller measurement, it is a different one."""
    cols = {"aspect_p50": 0.3, "needle_frac": 0.15, "hard_needle_frac": 0.002,
            "smid_p50_mm": 7.0, "smax_p50_mm": 25.0, "splats": 300000,
            "on_seed_frac_1cm": 0.07, "thin_axis_angle_p50": 41.0}
    _checkpoints(probe_scene, {t: {s: cols for s in (500, 2000)} for t in ("F0", "G0")})
    with pytest.raises((SystemExit, FileNotFoundError)):
        H.early_divergence(probe_scene, "pgeom")


# --------------------------------------------------------------------- regrade purity


def test_regrade_reproduces_the_grade_from_the_ARTIFACTS_ALONE(tmp_path):
    """CATCHES a grade that depends on anything but the committed artifacts -- which is
    the only form of reproducibility that survives the scratch directory being cleaned,
    and the claim that lets grading land AFTER the arms have already run."""
    out = make_scene(tmp_path, treatment={"stats.on_seed_frac_1cm": 0.09})
    H.write_floors(out, "pgeom", 0.05, "disparity")
    first = H.grade_scene(out, "pgeom", 0.05, "G0", H.ANCHOR_PATH)
    H.write_grade(out, "G0", first)
    again = H.grade_scene(out, "pgeom", 0.05, "G0", H.ANCHOR_PATH)
    assert json.dumps(again, sort_keys=True) == json.dumps(first, sort_keys=True)
