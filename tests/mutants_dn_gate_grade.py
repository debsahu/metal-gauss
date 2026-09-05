#!/usr/bin/env python3
"""Mutation battery for tests/test_dn_gate_grade.py.

Same contract as `tests/mutants_dn_gate_arms.py`: every mutant is PROVEN to change
behaviour by a probe that CALLS the code, then required to kill a NAMED test.

The probe drives the whole grading path -- battery, floors, the three bands, the anchors,
the cross-scene verdict and Reading B -- over synthetic scenes built by the test module's
own helpers, so the probe and the tests cannot drift into describing different fixtures.

    .venv/bin/python tests/mutants_dn_gate_grade.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "scripts" / "dn_gate_arms.py"
ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}

PROBE = r'''
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, %r); sys.path.insert(0, %r); sys.path.insert(0, %r)
import dn_gate_arms as H
from test_dn_gate_grade import make_scene, write_arm, BASE

TMP = Path(tempfile.mkdtemp())
def norm(x):
    return json.loads(json.dumps(x, default=str).replace(str(TMP), "<TMP>"))
def guarded(fn, *a, **kw):
    try:
        return {"ok": norm(fn(*a, **kw))}
    except BaseException as e:
        return {"raised": type(e).__name__, "msg": norm(str(e))[:400]}

sig = {}

# ---- the Band 1 table, its two spaces, and how big each threshold is on a baseline
sig["collapse_table"] = norm(H.COLLAPSE)
sig["band2_gate"] = list(H.BAND2_GATE)
sig["drift_scope"] = list(H.DRIFT_SCOPE)
sig["transfers"] = {c: H.transfers_between_scenes(s) for c, s in H.COLLAPSE.items()}
sig["threshold_relative"] = {
    c: [guarded(H.threshold_relative, s, b) for b in (0.15156066914399466, 0.2348, 0.99)]
    for c, s in H.COLLAPSE.items()}
sig["collapse_delta"] = {
    f"{m}:{v}:{r}": guarded(H.collapse_delta, m, v, r)
    for m, v, r in (("run.needle_frac", 0.30, 0.10), ("run.needle_frac", 0.05, 0.10),
                    ("stats.on_seed_frac_1cm", 0.05, 0.10),
                    ("stats.on_seed_frac_1cm", 0.20, 0.10),
                    ("run.aspect_p50", 0.20, 0.30), ("run.lpips", 0.42, 0.40))}

# ---- the bands as pure functions
V = ["IMPROVED", "WORSENED", "WITHIN FLOOR"]
sig["band2"] = {f"{a}|{b}": guarded(H.band2, {"stats.on_seed_frac_1cm": a,
                                              "stats.thin_axis_angle_p50": b})
                for a in V for b in V}
sig["band2_missing"] = guarded(H.band2, {"stats.on_seed_frac_1cm": "IMPROVED"})
sig["band3"] = {f"{t}v{b}": guarded(H.band3, t, b)
                for t, b in ((30.0, 25.0), (24.80, 25.0), (24.70, 25.0), (24.75, 25.0),
                             (23.95, 24.05), (23.4, 23.5), (23.0, 23.5))}

# ---- floors
out = make_scene(TMP, floors=({"run.needle_frac": 0.150}, {"run.needle_frac": 0.151},
                              {"run.needle_frac": 0.160}))
sig["floors"] = guarded(H.write_floors, out, "pgeom", 0.05, "disparity")
sig["grade_before_floors"] = guarded(
    H.grade_scene, make_scene(TMP / "nofloors"), "pgeom", 0.05, "G0", H.ANCHOR_PATH)
sig["merge_disagree"] = guarded(
    H.merge_extended_floors,
    {"m": {"F0": 1.0, "F1": 1.0, "F2": 1.0, "mean": 1.0, "spread_n3": 0.0}},
    {"m": {"F0": 1.0, "F1": 1.0, "F2": 2.0, "mean": 1.33, "spread_n3": 1.0}})
o2 = make_scene(TMP / "cfg")
rep = json.loads((o2 / "F2.json").read_text()); rep["resolved"]["depth_normal_weight"] = 0.0
(o2 / "F2.json").write_text(json.dumps(rep))
sig["floors_config_differ"] = guarded(H.write_floors, o2, "pgeom", 0.05, "disparity")
o3 = make_scene(TMP / "refs")
st = json.loads((o3 / "F1.stats.json").read_text()); st["seed_cloud"] = "/other.txt"
(o3 / "F1.stats.json").write_text(json.dumps(st))
sig["floors_two_refs"] = guarded(H.write_floors, o3, "pgeom", 0.05, "disparity")

# ---- the battery's own refusals
sig["battery_seeded_from"] = guarded(
    H.battery, make_scene(TMP / "seedcloud", seed_cloud_name="points3D.bin"), "F0")
o4 = make_scene(TMP / "role")
r4 = json.loads((o4 / "F0.json").read_text())
r4["observed"]["loss_path"] = {"fused_calls": 0, "torch_calls": 9, "dn_gated_calls": 9,
                               "dn_ungated_calls": 0, "dn_skipped_calls": 0}
(o4 / "F0.json").write_text(json.dumps(r4))
sig["battery_wrong_role"] = guarded(H.battery, o4, "F0")

# ---- graded scenes
def graded(name, **kw):
    o = make_scene(TMP / name, **kw)
    H.write_floors(o, "pgeom", 0.05, "disparity")
    return guarded(H.grade_scene, o, "pgeom", 0.05, "G0", H.ANCHOR_PATH)

sig["g_flat"] = graded("g_flat")
sig["g_drift"] = graded("g_drift", treatment={"run.aspect_p50": 0.3126 * 0.975,
                                              "run.needle_frac": 0.1576})
sig["g_collapse"] = graded("g_collapse", treatment={"run.aspect_p50": 0.0659,
                                                    "run.needle_frac": 0.5679})
sig["g_ratchet"] = graded("g_ratchet",
                          floors=({"run.needle_frac": 0.24},) * 3,
                          treatment={"run.needle_frac": 0.27})
sig["g_improved"] = graded("g_improved",
                           treatment={"stats.on_seed_frac_1cm": 0.20,
                                      "run.needle_frac": 0.1716})
sig["g_psnr"] = graded("g_psnr", treatment={"run.psnr_masked": 24.70})

o5 = make_scene(TMP / "nolpips")
H.write_floors(o5, "pgeom", 0.05, "disparity")
r5 = json.loads((o5 / "G0.json").read_text()); del r5["metrics"]["lpips"]
(o5 / "G0.json").write_text(json.dumps(r5))
sig["g_missing_band1_col"] = guarded(H.grade_scene, o5, "pgeom", 0.05, "G0", H.ANCHOR_PATH)

o6 = make_scene(TMP / "cfgmismatch")
H.write_floors(o6, "pgeom", 0.05, "disparity")
for t in ("F0", "F1", "F2", "G0"):
    r = json.loads((o6 / f"{t}.json").read_text()); r["resolved"]["num_downscales"] = 2
    (o6 / f"{t}.json").write_text(json.dumps(r))
sig["g_anchor_config"] = guarded(H.grade_scene, o6, "pgeom", 0.05, "G0", H.ANCHOR_PATH)

o7 = make_scene(TMP / "noanchor", name="nosuchscene")
H.write_floors(o7, "nosuchscene", 0.05, "disparity")
sig["g_no_anchor"] = guarded(H.grade_scene, o7, "nosuchscene", 0.05, "G0", H.ANCHOR_PATH)

# ---- the self-anchor, and vacuity measured rather than declared
o8 = make_scene(TMP / "self", name="pmask")
H.write_floors(o8, "pmask", 0.05, "disparity", self_anchor=True)
sig["self_anchor_doc"] = guarded(lambda: json.loads((o8 / "anchor.json").read_text()))
sig["self_first_arm"] = guarded(H.grade_scene, o8, "pmask", 0.05, "G0", H.ANCHOR_PATH)
ref8 = str(TMP / "self" / "ds" / "sparse" / "0" / "points3D.tsdf.txt")
for t, s in (("F0", 42), ("F1", 42), ("F2", 43)):
    write_arm(o8, t, {"run.needle_frac": 0.20}, s, "floor", ref8,
              str(TMP / "self" / "ds" / "sparse" / "0"))
(o8 / "floors.json").unlink(); (o8 / "FLOORS_DONE").unlink()
H.write_floors(o8, "pmask", 0.05, "disparity", self_anchor=True)
sig["self_second_arm"] = guarded(H.grade_scene, o8, "pmask", 0.05, "G0", H.ANCHOR_PATH)

# ---- the cross-scene verdict
def sv(**k):
    return {"band1_fired": False, "band2": "PASS", "band3_fired": False, "drift": [],
            "dn": 0.05, "scene_pass": True, "scene_drop": False,
            "falsifier_triggered_on_this_scene": False} | k
both = lambda **k: {"pgeom": sv(**k), "pmask": sv(**k)}
sig["combined"] = {
  "clean": guarded(H.combined_verdict, both()),
  "drift": guarded(H.combined_verdict, both(drift=[{"metric": "run.needle_frac"}])),
  "one_within": guarded(H.combined_verdict,
                        {"pgeom": sv(), "pmask": sv(band2="WITHIN FLOOR",
                                                    scene_pass=False)}),
  "none": guarded(H.combined_verdict, both(band2="WITHIN FLOOR", scene_pass=False)),
  "collapse": guarded(H.combined_verdict, both(band1_fired=True)),
  "mixed": guarded(H.combined_verdict,
                   {"pgeom": sv(), "pmask": sv(band1_fired=True, scene_pass=False)}),
  "b2fail": guarded(H.combined_verdict, {"pgeom": sv(band2="FAIL", scene_pass=False)})}

# ---- collect_scenes and grade filenames
root = TMP / "tree"; (root / "pgeom").mkdir(parents=True)
(root / "pgeom" / "grade.json").write_text(json.dumps({"scene": "pgeom"}))
sig["collect_ok"] = guarded(H.collect_scenes, root, "pgeom")
sig["collect_missing"] = guarded(H.collect_scenes, root, "pgeom,pmask")
(root / "fixture").mkdir()
(root / "fixture" / "grade.json").write_text(json.dumps({"scene": "fixture"}))
sig["collect_stray"] = guarded(H.collect_scenes, root, "pgeom")
bad = TMP / "tree2"; (bad / "pgeom").mkdir(parents=True)
(bad / "pgeom" / "grade.json").write_text(json.dumps({"scene": "pmask"}))
sig["collect_mismatch"] = guarded(H.collect_scenes, bad, "pgeom")
sig["grade_filename"] = [H.grade_filename("G0"), H.grade_filename("G1")]
wg = TMP / "wg"; wg.mkdir()
H.write_grade(wg, "G1", {"scene": "pmask"})
sig["write_grade_secondary_made_primary"] = (wg / "grade.json").exists()
H.write_grade(wg, "G0", {"scene": "pmask"})
sig["write_grade_primary"] = (wg / "grade.json").exists()

# ---- Reading B
o9 = make_scene(TMP / "probe", floors=({"run.needle_frac": 0.150},
                                       {"run.needle_frac": 0.151},
                                       {"run.needle_frac": 0.160}))
H.write_floors(o9, "pgeom", 0.05, "disparity")
def cols(**kw):
    return {"aspect_p50": 0.30, "needle_frac": 0.15, "hard_needle_frac": 0.002,
            "smid_p50_mm": 7.0, "smax_p50_mm": 25.0, "splats": 300000,
            "on_seed_frac_1cm": 0.07, "thin_axis_angle_p50": 41.0} | kw
TABLE = {}
H.checkpoint_columns = lambda o, tag, step, sc: TABLE[tag]
TABLE = {"F0": cols(), "F1": cols(needle_frac=0.16), "G0": cols(needle_frac=0.19)}
sig["probe_normal"] = guarded(H.early_divergence, o9, "pgeom")
TABLE = {"F0": cols(), "F1": cols(), "G0": cols(needle_frac=0.19)}
sig["probe_zero_noise"] = guarded(H.early_divergence, o9, "pgeom")
def missing(o, tag, step, sc):
    if tag == "F1":
        raise FileNotFoundError("no checkpoint")
    return TABLE[tag]
H.checkpoint_columns = missing
sig["probe_missing_arm"] = guarded(H.early_divergence, o9, "pgeom")

print(json.dumps(sig, sort_keys=True))
''' % (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests"))


def probe() -> str:
    r = subprocess.run([str(ROOT / ".venv/bin/python"), "-c", PROBE],
                       capture_output=True, text=True, env=ENV, cwd=str(ROOT))
    if r.returncode != 0:
        return "PROBE-CRASH:" + r.stderr.strip()[-600:]
    return r.stdout.strip()


def failing_test_names(node_ids) -> set:
    r = subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q", "--tb=no",
                        "-p", "no:cacheprovider", *node_ids],
                       capture_output=True, text=True, env=ENV, cwd=str(ROOT))
    return set(re.findall(r"^FAILED [^:]+::([\w\[\]\-.,]+)", r.stdout, re.M))


def sub(text, old, new, count=1):
    """Replace, and REFUSE AN AMBIGUOUS ANCHOR.

    `>= 1` was not enough. `"    if missing:\n"` occurs in four functions here, so a
    mutant aimed at `collect_scenes` silently rewrote `check_loss_path` instead: the probe
    saw a real behaviour change, the named test went on passing, and the battery reported
    a SURVIVOR that was actually a defective mutant. Requiring the anchor to be unique
    makes that a hard error at the mutation site rather than a puzzle at the result.
    """
    n = text.count(old)
    assert n >= 1, f"mutation anchor not found:\n{old}"
    assert n == count, (f"mutation anchor is AMBIGUOUS ({n} occurrences, expected "
                        f"{count}) -- it would mutate a site other than the intended "
                        f"one:\n{old}")
    return text.replace(old, new, count)


MUTANTS = [
    ("band1_ignores_the_CUMULATIVE_half",
     lambda s: sub(s, '    cf = [k for k, v in cum.items() if v["fired"]]', "    cf = []"),
     "test_band1_fires_CUMULATIVELY_even_when_the_per_arm_check_does_not"),

    ("collapse_delta_sign_inverted",
     lambda s: sub(s, '    return spec["worse"] * d', '    return -spec["worse"] * d'),
     "test_collapse_delta_is_POSITIVE_TOWARD_WORSE_in_both_spaces"),

    ("a_missing_Band1_column_reads_as_no_collapse",
     lambda s: sub(s, "        if col not in values:\n", "        if False:\n"),
     "test_a_MISSING_Band1_column_is_refused_rather_than_reading_as_no_collapse"),

    ("band2_treats_an_absent_gate_column_as_a_pass",
     lambda s: sub(s, "    missing = [k for k in BAND2_GATE if verdicts.get(k) is None]",
                   "    missing = []"),
     "test_band2_FAILS_on_a_worsened_column_and_an_ABSENT_column_is_refused"),

    ("band2_still_reads_aspect_and_needles",
     lambda s: sub(s, 'BAND2_GATE = ("stats.on_seed_frac_1cm", "stats.thin_axis_angle_p50")',
                   'BAND2_GATE = ("stats.on_seed_frac_1cm", "stats.thin_axis_angle_p50",\n'
                   '              "run.aspect_p50", "run.needle_frac")'),
     "test_band2_FAILS_on_a_worsened_column_and_an_ABSENT_column_is_refused"),

    ("band3_is_TWO_SIDED_again",
     lambda s: sub(s, '"exceeds_allowance": loss > PSNR_DROP_DB',
                   '"exceeds_allowance": abs(loss) > PSNR_DROP_DB')
                 .replace('"fired": bool(loss > PSNR_DROP_DB or crossed)',
                          '"fired": bool(abs(loss) > PSNR_DROP_DB or crossed)'),
     "test_band3_is_ONE_SIDED_a_PSNR_GAIN_is_not_a_regression"),

    ("band3_comparison_is_not_strict",
     lambda s: sub(s, '"fired": bool(loss > PSNR_DROP_DB or crossed)',
                   '"fired": bool(loss >= PSNR_DROP_DB or crossed)'),
     "test_band3_is_ONE_SIDED_a_PSNR_GAIN_is_not_a_regression"),

    ("band3_drops_the_Stage4_gate_clause",
     lambda s: sub(s, "    crossed = psnr_baseline >= STAGE4_PSNR_DB > psnr_treatment",
                   "    crossed = False"),
     "test_band3_fires_on_CROSSING_the_24dB_Stage4_gate_from_above"),

    ("drift_counts_IMPROVEMENTS",
     lambda s: sub(s, '            worse = verdicts[k] == "WORSENED"',
                   '            worse = verdicts[k] != "WITHIN FLOOR"'),
     "test_DRIFT_excludes_improvements_and_excludes_a_Band1_firing"),

    ("drift_includes_a_Band1_firing",
     lambda s: sub(s, '    fired = set(band1_detail["per_arm_fired"]) | '
                      'set(band1_detail["cumulative_fired"])',
                   "    fired = set()"),
     "test_DRIFT_excludes_improvements_and_excludes_a_Band1_firing"),

    ("drift_does_not_flag_the_column_that_CAUSED_the_fire",
     lambda s: sub(s, '        caused_b3 = bool(band3_fired and k == "run.psnr_masked")',
                   "        caused_b3 = False"),
     "test_a_drift_column_that_CAUSED_the_scene_to_fail_is_flagged_as_such"),

    ("threshold_relative_ignores_the_scene_baseline",
     lambda s: sub(s, '        return (spec["threshold"] / reference) if reference else None',
                   '        return spec["threshold"]'),
     "test_the_ABSOLUTE_Band1_thresholds_are_reported_relative_to_the_SCENES_OWN_baseline"),

    ("transfers_between_scenes_hardcoded_true",
     lambda s: sub(s, '    return spec["space"] == "log"\n\n\ndef threshold_relative',
                   "    return True\n\n\ndef threshold_relative"),
     "test_the_LOG_thresholds_transfer_between_scenes_and_the_ABSOLUTE_ones_do_not"),

    ("the_grade_omits_the_scenes_own_baseline",
     lambda s: sub(s, '                    "scene_baseline": reference[col],\n', ""),
     "test_the_grade_carries_the_scene_baseline_and_the_relative_size_beside_every_verdict"),

    ("vacuity_is_READ_OFF_the_self_anchored_flag",
     lambda s: sub(s, "    vacuous = all(abs(anchor_values[c] - base_values[c])\n"
                      "                  <= 1e-12 * max(1.0, abs(base_values[c])) "
                      "for c in COLLAPSE)",
                   "    vacuous = self_anchored"),
     "test_vacuity_is_MEASURED_not_read_off_the_self_anchored_flag"),

    ("the_self_anchor_is_REWRITTEN_every_time_the_floors_are",
     lambda s: sub(s, "    if p.exists():\n"
                      '        print(f"  self-anchor already frozen at {p}, keeping it", '
                      "flush=True)\n        return p\n", "    if False:\n        pass\n"),
     "test_vacuity_is_MEASURED_not_read_off_the_self_anchored_flag"),

    ("a_missing_anchor_becomes_an_EMPTY_one",
     lambda s: sub(s, "    raise SystemExit(\n"
                      '        f"no frozen Tier 3 anchor for scene {scene!r} in {path}, '
                      'and no self-anchor at "',
                   '    return {"values": {}, "config": {}, "self_anchored": False}\n'
                   "    raise SystemExit(\n"
                   '        f"no frozen Tier 3 anchor for scene {scene!r} in {path}, '
                   'and no self-anchor at "'),
     "test_a_scene_with_NO_anchor_is_an_error_not_a_vacuous_cumulative_check"),

    ("the_anchor_config_check_skips_steps_and_downscales",
     lambda s: sub(s, 'ANCHOR_CONFIG_KEYS = ("budget", "steps", "max_resolution", '
                      '"num_downscales")',
                   'ANCHOR_CONFIG_KEYS = ("budget", "max_resolution")'),
     "test_an_anchor_measured_at_ANOTHER_configuration_is_refused"),

    ("the_cumulative_delta_is_not_DECOMPOSED",
     lambda s: sub(s, '        c: {"everything_but_the_gate": collapse_delta(c, '
                      "base_values[c],\n                                                    "
                      "  anchor_values[c]),\n"
                      '            "the_gate": collapse_delta(c, t_values[c], '
                      "base_values[c])}",
                   '        c: {"everything_but_the_gate": 0.0, "the_gate": 0.0}'),
     "test_the_cumulative_delta_is_reported_DECOMPOSED_into_gate_and_everything_else"),

    ("DROP_is_not_checked_FIRST",
     lambda s: sub(s, "    if drops:\n        decision = \"DROP\"\n"
                      "    elif len(passes) == len(scenes):",
                   "    if len(passes) == len(scenes):"),
     "test_DROP_is_checked_FIRST_and_is_not_overridable"),

    ("KEEP_AS_DEFAULT_ignores_drift",
     lambda s: sub(s, '        decision = "KEEP AS DEFAULT" if not any_drift else '
                      '"OPT-IN, DEFAULT-CANDIDATE"',
                   '        decision = "KEEP AS DEFAULT"'),
     "test_the_grader_can_reach_EVERY_ONE_of_its_verdicts"),

    ("collect_scenes_GLOBS_the_tree",
     lambda s: sub(s, "    if extra:\n", "    if False:\n"),
     "test_summary_refuses_an_UNNAMED_grade_in_the_tree"),

    ("collect_scenes_does_not_notice_a_missing_named_scene",
     lambda s: sub(s, '    if missing:\n        raise SystemExit(f"--scenes named '
                      '{missing} but there is no {fname} for them "',
                   '    if False:\n        raise SystemExit(f"--scenes named '
                      '{missing} but there is no {fname} for them "'),
     "test_summary_refuses_a_NAMED_scene_that_has_no_grade"),

    ("collect_scenes_TRUSTS_the_directory_name",
     lambda s: sub(s, '        if g.get("scene") != n:\n', "        if False:\n"),
     "test_a_grade_whose_scene_field_disagrees_with_its_directory_is_refused"),

    ("a_SECOND_arms_grade_overwrites_the_primary_verdict",
     lambda s: sub(s, "    if tag == PRIMARY_TAG:\n"
                      '        (out / "grade.json").write_text',
                   "    if True:\n"
                   '        (out / "grade.json").write_text'),
     "test_a_SECOND_arms_grade_never_overwrites_the_primary_scene_verdict"),

    ("grading_does_not_require_the_floors_FIRST",
     lambda s: sub(s, '    if not (out / "FLOORS_DONE").exists() or not '
                      '(out / "floors.json").exists():\n', "    if False:\n"),
     "test_grading_REFUSES_until_the_floors_have_been_written"),

    ("the_floor_is_the_PAIR_difference_not_the_n3_spread",
     lambda s: sub(s, '"spread_n3": max(v) - min(v),', '"spread_n3": abs(v[0] - v[1]),'),
     "test_the_floor_is_the_n3_SPREAD_and_the_pair_difference_is_reported_only"),

    ("floor_arms_may_differ_in_configuration",
     lambda s: sub(s, "        if diff:\n", "        if False:\n"),
     "test_floor_arms_that_differ_in_a_CONFIGURATION_FLAG_are_not_a_floor"),

    ("floor_arms_may_use_different_reference_clouds",
     lambda s: sub(s, "    if len(refs) > 1:\n", "    if False:\n"),
     "test_the_floor_arms_must_share_ONE_reference_cloud"),

    ("the_floor_rebuild_cross_check_is_a_no_op",
     lambda s: sub(s, "            if a is None or b is None or abs(a - b) > 1e-12 * "
                      "max(1.0, abs(a)):\n", "            if False:\n"),
     "test_a_floor_REBUILD_that_disagrees_with_the_written_floors_is_refused"),

    ("the_battery_skips_the_role_assertion",
     lambda s: sub(s, "    check_loss_path(tag, role_of(tag), rep)\n", ""),
     "test_the_battery_re_asserts_every_arms_OBSERVED_LOSS_PATH"),

    ("the_battery_skips_the_seed_cloud_check",
     lambda s: sub(s, '    check_seed_cloud(ref, resolved.get("colmap"), '
                      'resolved.get("init_ply"))\n', ""),
     "test_the_battery_refuses_a_reference_cloud_THE_ARM_ITSELF_seeded_from"),

    ("the_probe_reports_an_INFINITE_ratio_on_zero_noise",
     lambda s: sub(s, '"effect_over_noise": (effect / noise) if noise != 0 else None,',
                   '"effect_over_noise": (effect / noise) if noise != 0 '
                   'else float("inf"),'),
     "test_ZERO_NOISE_is_never_reported_as_an_infinite_ratio"),

    ("the_probe_EMITS_A_VERDICT",
     lambda s: sub(s, '    return {"schema": 1, "scene": scene, "arms": list(PROBE_ARMS),',
                   '    return {"schema": 1, "scene": scene, "arms": list(PROBE_ARMS),\n'
                   '            "decision": "KEEP",'),
     "test_the_probe_EMITS_NO_VERDICT_ANYWHERE_IN_ITS_OUTPUT"),

    ("the_probe_reads_steps_other_than_the_pre_registered_ones",
     lambda s: sub(s, "GRADED_EARLY_STEPS = (500, 2000)",
                   "GRADED_EARLY_STEPS = (500, 1000)"),
     "test_the_probe_reads_EXACTLY_the_pre_registered_steps_and_records_which"),

    ("the_probe_drops_the_Band2_pair",
     lambda s: sub(s, '    "on_seed_frac_1cm": "stats.on_seed_frac_1cm",\n'
                      '    "thin_axis_angle_p50": "stats.thin_axis_angle_p50",\n', ""),
     "test_the_probe_covers_the_five_shape_columns_the_splat_count_and_the_Band2_pair"),

    ("the_probe_SKIPS_a_missing_checkpoint",
     lambda s: sub(s, "        arms = {t: checkpoint_columns(out, t, step, seed_cloud) "
                      "for t in PROBE_ARMS}\n",
                   "        arms = {}\n"
                   "        for _t in PROBE_ARMS:\n"
                   "            try:\n"
                   "                arms[_t] = checkpoint_columns(out, _t, step, "
                   "seed_cloud)\n"
                   "            except BaseException:\n"
                   "                continue\n"),
     "test_a_MISSING_checkpoint_is_refused_rather_than_dropping_the_column"),

    ("the_probe_swaps_EFFECT_and_NOISE",
     lambda s: sub(s, "            effect, noise = g0 - f0, f1 - f0",
                   "            effect, noise = f1 - f0, g0 - f0"),
     "test_the_probe_reports_EFFECT_NOISE_and_their_ratio_per_column"),
]

NODES = ["tests/test_dn_gate_grade.py"]


def main() -> int:
    src = TARGET.read_text()
    base_sig = probe()
    assert not base_sig.startswith("PROBE-CRASH"), base_sig
    base_fail = failing_test_names(NODES)
    assert not base_fail, f"the suite must be green before mutating; failing: {base_fail}"
    print(f"baseline green, probe signature {len(base_sig)} chars\n")

    results = []
    backup = Path(tempfile.mkdtemp()) / "dn_gate_arms.py"
    shutil.copy2(TARGET, backup)
    try:
        for name, mutate, must_fail in MUTANTS:
            TARGET.write_text(mutate(src))
            sig = probe()
            changed = sig != base_sig
            fails = failing_test_names(NODES) if changed else set()
            killed = must_fail in fails
            results.append((name, changed, killed))
            mark = "KILLED" if (changed and killed) else (
                "SURVIVED" if changed else "NO-BEHAVIOUR-CHANGE (not a mutant)")
            print(f"{mark:34s} {name}")
            if changed and not killed:
                print(f"    expected {must_fail} to fail; got {sorted(fails)}")
            if sig.startswith("PROBE-CRASH"):
                print("    probe crashed (still a behaviour change): " + sig[:240])
    finally:
        shutil.copy2(backup, TARGET)

    assert TARGET.read_text() == src, "restore failed -- the target is NOT the original"
    after = failing_test_names(NODES)
    assert not after, f"suite not green after restore: {after}"
    n_ok = sum(1 for _, c, k in results if c and k)
    print(f"\n{n_ok}/{len(results)} killed; restore verified by re-running the suite green")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
