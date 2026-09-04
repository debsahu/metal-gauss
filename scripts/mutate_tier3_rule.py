#!/usr/bin/env python3
"""Mutation battery for the Tier 3 three-band rule.

    PYTHONDONTWRITEBYTECODE=1 scripts/mutate_tier3_rule.py

Each entry substitutes ONE wrong implementation, runs the three rule test files, and
records WHICH TESTS FAILED BY NAME. A count is not the criterion: any failure satisfies a
count, including one caused by an import error or a dropped argument, so each mutant
declares the test it must kill and the run FAILS if that specific test survived.

`PYTHONDONTWRITEBYTECODE=1` is set on the child regardless of the caller: a stale
`__pycache__` makes this battery report FALSE SURVIVED, which is the direction that reads
as "the tests are fine" while proving nothing.

Every mutant is also checked for BEHAVIOURAL EFFECT before its kill counts. A mutant that
does not change what the module computes -- a comment, a renamed local, an equivalent
rewrite -- proves nothing when it survives and proves less than nothing when a test
crashes on it.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULE = ROOT / "scripts/plane_aux_arms.py"
PLY = ROOT / "scripts/ply_shape.py"
TESTS = ["tests/test_plane_aux_tier3_rule.py", "tests/test_plane_aux_grading.py",
         "tests/test_ply_shape.py"]

# (name, file, old, new, test that MUST fail, one line on what wrong behaviour it models)
MUTANTS = [
    ("band1-sign-flip", RULE,
     'return spec["worse"] * d',
     'return -spec["worse"] * d',
     "test_THE_VOID_ROW_fires_band1_on_FOUR_INDEPENDENT_COLUMNS",
     "worse/better inverted: a halved on-seed would read as a large improvement"),
    ("band1-not-strict", RULE,
     '"space": COLLAPSE[col]["space"], "fired": d > thr}',
     '"space": COLLAPSE[col]["space"], "fired": d >= thr}',
     "test_the_band1_comparison_is_STRICT_at_the_threshold",
     "a delta exactly at the threshold counted as a collapse"),
    ("band1-cumulative-is-per-arm", RULE,
     'cum = _collapse_side(t_values, anchor_values, "anchor")',
     'cum = _collapse_side(t_values, base_values, "anchor")',
     "test_CUMULATIVE_band1_catches_a_RATCHET_that_no_single_arm_fires_on",
     "the cumulative check grades against the re-measured floors, so the rule ratchets"),
    ("band1-needs-all-columns", RULE,
     'return {"per_arm": per, "cumulative": cum, "per_arm_fired": pf,\n'
     '            "cumulative_fired": cf, "fired": bool(pf or cf)}',
     'return {"per_arm": per, "cumulative": cum, "per_arm_fired": pf,\n'
     '            "cumulative_fired": cf,\n'
     '            "fired": len(pf) == len(COLLAPSE) or len(cf) == len(COLLAPSE)}',
     "test_band1_fires_on_ANY_ONE_column_alone",
     "`all` where the rule says `any one column`"),
    ("band1-missing-column-passes", RULE,
     '        if col not in values:\n'
     '            raise SystemExit(f"Band 1 column {col} is missing from the treatment battery. "\n'
     '                             f"A collapse column that was never measured must never read "\n'
     '                             f"as \'did not collapse\'.")',
     '        if col not in values:\n'
     '            continue',
     "test_a_band1_column_MISSING_from_the_treatment_is_an_ERROR_not_a_pass",
     "an unmeasured collapse column silently reads as 'did not collapse'"),
    ("band1-threshold-retuned", RULE,
     '"run.aspect_p50":       {"space": "log", "worse": -1, "threshold": 0.346},',
     '"run.aspect_p50":       {"space": "log", "worse": -1, "threshold": 0.050},',
     "test_every_collapse_threshold_is_REDERIVED_from_the_section_8_1_table",
     "a threshold nudged after seeing an arm, with no derivation behind it"),
    ("band2-keeps-the-old-four-column-gate", RULE,
     'BAND2_GATE = ("stats.on_seed_frac_1cm", "stats.thin_axis_angle_p50")',
     'BAND2_GATE = ("stats.on_seed_frac_1cm", "stats.thin_axis_angle_p50",\n'
     '              "run.aspect_p50", "run.needle_frac")',
     "test_band2_does_NOT_read_aspect_or_needles",
     "the magnitude-blind gate survives inside Band 2 and drops Task 19 again"),
    ("band2-missing-column-passes", RULE,
     '    missing = [k for k in BAND2_GATE if verdicts.get(k) is None]\n'
     '    if missing:\n'
     '        raise SystemExit(f"Band 2 columns missing from the battery: {missing}. An absent "\n'
     '                         f"gate column must never read as a pass.")',
     '    missing = []',
     "test_a_missing_band2_verdict_is_an_ERROR",
     "an absent gate column reads as a pass"),
    ("band3-two-sided", RULE,
     '"fired": bool(loss > PSNR_DROP_DB or crossed)}',
     '"fired": bool(abs(loss) > PSNR_DROP_DB or crossed)}',
     "test_band3_does_NOT_fire_on_a_psnr_GAIN",
     "the retired two-sided PSNR condition, which no Tier 3 arm can pass"),
    ("band3-stage4-reads-the-baseline", RULE,
     'crossed = psnr_baseline >= STAGE4_PSNR_DB > psnr_treatment',
     'crossed = psnr_treatment >= STAGE4_PSNR_DB > psnr_baseline',
     "test_band3_fires_when_a_scene_that_was_ABOVE_24_dB_drops_below_it",
     "the Stage 4 crossing tested in the wrong direction"),
    ("drift-counts-improvements", RULE,
     '            worse = verdicts[k] == "WORSENED"',
     '            worse = rows[k]["floor_spread_n3"] < abs(d)',
     "test_drift_is_WORSENING_ONLY_or_KEEP_AS_DEFAULT_is_unreachable",
     "`moves` used as the drift predicate, making KEEP AS DEFAULT unreachable"),
    ("drift-includes-collapsed-columns", RULE,
     '    fired = set(band1_detail["per_arm_fired"]) | set(band1_detail["cumulative_fired"])',
     '    fired = set()',
     "test_a_column_that_FIRED_band1_is_a_COLLAPSE_not_a_drift",
     "a collapsed column reported as drift, so a hard DROP reads as adoptable"),
    ("decision-opt-in-before-drop", RULE,
     '    if drops:\n        decision = "DROP"\n    elif len(passes) == len(scenes):',
     '    if False:\n        decision = "DROP"\n    elif len(passes) == len(scenes):',
     "test_band1_ANYWHERE_is_a_DROP_even_when_both_scenes_pass_band2",
     "the opt-in branch reached before the drop branch"),
    ("decision-drift-ignored", RULE,
     '        decision = "KEEP AS DEFAULT" if not any_drift else "OPT-IN, DEFAULT-CANDIDATE"',
     '        decision = "KEEP AS DEFAULT"',
     "test_pass_on_both_WITH_drift_is_OPT_IN_DEFAULT_CANDIDATE_and_names_the_A_B",
     "a drift-carrying arm promoted straight to the recipe default"),
    ("anchor-config-unchecked", RULE,
     'ANCHOR_CONFIG_KEYS = ("budget", "steps", "max_resolution", "num_downscales")',
     'ANCHOR_CONFIG_KEYS = ()',
     "test_the_anchor_REFUSES_a_run_at_a_different_budget_or_resolution",
     "an anchor from another budget or resolution applied without complaint"),
    ("anchor-missing-scene-is-empty", RULE,
     '    entry = doc.get("scenes", {}).get(scene)\n    if not entry:',
     '    entry = doc.get("scenes", {}).get(scene, {"values": {}, "config": {}})\n'
     '    if False:',
     "test_an_unknown_scene_has_no_anchor_and_must_ERROR",
     "a missing anchor becoming a vacuous cumulative check"),
    ("summary-tag-ignored", RULE,
     '    return "grade.json" if tag == PRIMARY_TAG else f"grade_{tag}.json"',
     '    return "grade.json"',
     "test_a_NON_PRIMARY_arms_summary_reads_its_OWN_file_and_never_grade_json",
     "a second arm's summary computed from the pre-registered arm's verdicts"),
    ("ply-average-median", PLY,
     '    return float(np.partition(x, (x.size - 1) // 2)[(x.size - 1) // 2])',
     '    return float(np.median(x))',
     "test_the_median_is_torchs_LOWER_median_not_numpys_average",
     "np.median where the trainer used torch's lower median"),
    ("ply-hard-needle-is-the-soft-one", PLY,
     'HARD_NEEDLE_ASPECT = 0.01',
     'HARD_NEEDLE_ASPECT = 0.1',
     "test_hard_needles_are_counted_at_the_DELIVERY_threshold_not_the_needle_one",
     "the hard-needle column silently duplicating needle_frac"),
    ("ply-cross-check-is-a-warning", PLY,
     '    if bad:\n        raise SystemExit(',
     '    if False:\n        raise SystemExit(',
     "test_the_cross_check_REFUSES_when_the_recomputation_disagrees",
     "a shape sidecar written despite disagreeing with its own arm's report"),
    ("ply-scale-fields-by-offset", PLY,
     '    return np.stack([rows[:, names.index(w)] for w in want], axis=1).astype(np.float64)',
     '    return rows[:, -7:-4].astype(np.float64)',
     "test_the_parser_reads_scales_BY_NAME_not_by_a_fixed_offset",
     "the INRIA layout assumed instead of parsed, reading the wrong three columns"),
    ("ply-median-convention-ignored", PLY,
     '    med = MEDIANS[median]',
     '    med = MEDIANS["lower"]',
     "test_the_convention_argument_REACHES_the_computed_columns",
     "the convention argument ignored, so a Tier 1 reference silently mismatches"),
    ("ply-empty-crosscheck-stamps-verified", PLY,
     '    unchecked = [k for k in REQUIRED_CHECKS if k not in checked]',
     '    unchecked = []',
     "test_a_reference_with_NO_shape_columns_cannot_stamp_a_file_as_verified",
     "a report with no shape block passing zero comparisons and being called verified"),
    ("drift-band3-flag-always-false", RULE,
     "        caused_b3 = bool(band3_fired and k == \"run.psnr_masked\")",
     "        caused_b3 = False",
     "test_THE_VOID_ROW_GRADED_END_TO_END_FROM_ITS_OWN_ARTIFACTS_IS_A_DROP",
     "the Band 3 firing reported as an unqualified caveat-level drift"),
    ("tier1-anchor-uses-one-arm", RULE,
     'DRIFT_SCOPE = tuple(dict.fromkeys(tuple(COLLAPSE) + BAND2_GATE + ("run.psnr_masked",)))',
     'DRIFT_SCOPE = tuple(COLLAPSE)',
     "test_a_psnr_LOSS_inside_band3_is_drift_and_a_psnr_GAIN_is_not",
     "PSNR and thin-axis dropped from the drift scope, so a photometric drift is invisible"),
    ("sidecar-verification-skipped", RULE,
     '    if d.get("verified_against_report") is not True:',
     '    if False:',
     "test_an_UNVERIFIED_sidecar_is_refused_by_the_battery",
     "a shape sidecar from another ply admitted into the battery"),
]


def failed_tests(env: dict) -> tuple[set[str], str]:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", "-p",
                        "no:cacheprovider", *TESTS],
                       cwd=str(ROOT), capture_output=True, text=True, env=env)
    names = set(re.findall(r"^(?:FAILED|ERROR) \S+::(\w+)", r.stdout, re.M))
    # a collection error names no test; treat it as a distinct, disqualifying outcome
    if "error" in r.stdout.lower() and not names and r.returncode != 0:
        names.add("<COLLECTION-ERROR>")
    return names, r.stdout[-1500:]


PROBE = r"""
import sys, json, math, tempfile, pathlib
sys.path.insert(0, 'scripts')
import numpy as np
import plane_aux_arms as m, ply_shape as p

def q(f, *a, **k):
    "Value, or the exception -- so a guard that stops raising CHANGES the fingerprint."
    try:
        return f(*a, **k)
    except BaseException as e:
        return f"{type(e).__name__}:{str(e)[:120]}"

B = {'run.needle_frac': 0.166, 'run.aspect_p50': 0.2957,
     'stats.on_seed_frac_1cm': 0.0847, 'run.lpips': 0.3952}
V = {'run.needle_frac': 0.568, 'run.aspect_p50': 0.0659,
     'stats.on_seed_frac_1cm': 0.0359, 'run.lpips': 0.5214}
T = {'run.needle_frac': 0.174, 'run.aspect_p50': 0.2900,
     'stats.on_seed_frac_1cm': 0.0860, 'run.lpips': 0.3960}
# a base that has itself drifted 0.24 from the anchor: per-arm inside, cumulative out
DRIFTED = dict(B, **{'run.needle_frac': B['run.needle_frac'] + 0.24})
RATCHET = dict(T, **{'run.needle_frac': DRIFTED['run.needle_frac'] + 0.08})
ONE = {c: (B[c] * math.exp(-0.4) if m.COLLAPSE[c]['space'] == 'log' else B[c] + 0.2)
       for c in m.COLLAPSE}
SHORT = {k: v for k, v in V.items() if k != 'run.lpips'}
EXACT = dict(B, **{'run.needle_frac': m.COLLAPSE['run.needle_frac']['threshold']})
ZERO = dict(B, **{'run.needle_frac': 0.0})

out = {}
out['void'] = q(lambda: m.band1(V, B, B)['fired'])
out['void_cols'] = q(lambda: sorted(m.band1(V, B, B)['per_arm_fired']))
out['healthy'] = q(lambda: m.band1(T, B, B)['fired'])
out['ratchet'] = q(lambda: (m.band1(RATCHET, DRIFTED, B)['per_arm_fired'],
                            m.band1(RATCHET, DRIFTED, B)['cumulative_fired'],
                            m.band1(RATCHET, DRIFTED, B)['fired']))
out['one_col'] = q(lambda: sorted(m.band1(dict(B, **{'run.needle_frac': ONE['run.needle_frac']}),
                                          B, B)['per_arm_fired']))
out['all_cols'] = q(lambda: m.band1(ONE, B, B)['fired'])
out['strict'] = q(lambda: (m.band1(EXACT, ZERO, ZERO)['fired'],
                           m.band1(dict(EXACT, **{'run.needle_frac':
                               math.nextafter(EXACT['run.needle_frac'], 1.0)}),
                               ZERO, ZERO)['fired']))
out['missing_t'] = q(lambda: m.band1(SHORT, B, B)['fired'])
out['missing_a'] = q(lambda: m.band1(V, B, SHORT)['fired'])
out['deltas'] = q(lambda: sorted((c, round(m.collapse_delta(c, V[c], B[c]), 9))
                                 for c in m.COLLAPSE))
out['thresholds'] = q(lambda: sorted((k, v['threshold'], v['space'], v['worse'])
                                     for k, v in m.COLLAPSE.items()))
G = {'stats.on_seed_frac_1cm': 'IMPROVED', 'stats.thin_axis_angle_p50': 'IMPROVED',
     'run.aspect_p50': 'WORSENED', 'run.needle_frac': 'WORSENED'}
out['b2_pass'] = q(lambda: m.band2(G))
out['b2_thin'] = q(lambda: m.band2(dict(G, **{'stats.thin_axis_angle_p50': 'WORSENED'})))
out['b2_flat'] = q(lambda: m.band2(dict(G, **{'stats.on_seed_frac_1cm': 'WITHIN FLOOR'})))
out['b2_short'] = q(lambda: m.band2({'stats.on_seed_frac_1cm': 'IMPROVED'}))
out['b3'] = q(lambda: [m.band3(22.0, 22.2)['fired'], m.band3(27.0, 22.2)['fired'],
                       m.band3(23.99, 24.05)['fired'], m.band3(23.99, 23.9)['fired'],
                       m.band3(24.00, 24.05)['fired'], m.band3(22.2 - 0.25, 22.2)['fired']])
DV = {k: 'WORSENED' for k in m.DRIFT_SCOPE}
DV['run.psnr_masked'] = 'MOVED'
UP = {k: 'IMPROVED' for k in m.DRIFT_SCOPE}
UP['run.psnr_masked'] = 'MOVED'
rows_dn = {k: {'delta': -0.01, 'floor_spread_n3': 0.001} for k in m.DRIFT_SCOPE}
rows_up = {k: {'delta': +0.01, 'floor_spread_n3': 0.001} for k in m.DRIFT_SCOPE}
NOFIRE = {'per_arm_fired': [], 'cumulative_fired': []}
out['drift_dn'] = q(lambda: sorted(x['metric'] for x in
                                   m.drift_columns(rows_dn, DV, NOFIRE, 'PASS')))
out['drift_up'] = q(lambda: sorted(x['metric'] for x in
                                   m.drift_columns(rows_up, UP, NOFIRE, 'PASS')))
out['drift_excl'] = q(lambda: sorted(x['metric'] for x in m.drift_columns(
    rows_dn, DV, {'per_arm_fired': ['run.aspect_p50'], 'cumulative_fired': []}, 'PASS')))
out['drift_flags'] = q(lambda: sorted((x['metric'], x['caused_band2_fail'],
                                       x['caused_band3_fire'])
                                      for x in m.drift_columns(rows_dn, DV, NOFIRE,
                                                               'FAIL', True)))
def g(**o):
    d = {'band1_fired': False, 'band2': 'PASS', 'band3_fired': False, 'drift': [],
         'dn': 0.0, 'falsifier_triggered_on_this_scene': False}
    d.update(o)
    return d
out['dec'] = q(lambda: [
    m.combined_verdict({'a': g(), 'b': g()})['decision'],
    m.combined_verdict({'a': g(drift=[{'metric': 'x'}]), 'b': g()})['decision'],
    m.combined_verdict({'a': g(), 'b': g(band2='WITHIN FLOOR')})['decision'],
    m.combined_verdict({'a': g(), 'b': g(band1_fired=True)})['decision'],
    m.combined_verdict({'a': g(), 'b': g(band2='FAIL')})['decision'],
    m.combined_verdict({'a': g(), 'b': g(band3_fired=True)})['decision'],
    m.combined_verdict({'a': g(band2='WITHIN FLOOR'),
                        'b': g(band2='WITHIN FLOOR')})['decision'],
    m.combined_verdict({'a': g(), 'b': g(band1_fired=True)})['regressed_on'],
])
out['anchor_keys'] = q(lambda: list(m.ANCHOR_CONFIG_KEYS))
CFG = {'budget': 500000, 'steps': 30000, 'max_resolution': 1920, 'num_downscales': 0}
out['anchor_ok'] = q(lambda: m.check_anchor_applies('s', {'config': CFG}, dict(CFG)))
out['anchor_bad'] = q(lambda: m.check_anchor_applies(
    's', {'config': CFG}, dict(CFG, budget=1)))
AP = pathlib.Path('bench/results/plane_aux/tier3_anchor.json')
out['anchor_load'] = q(lambda: sorted(m.load_anchor(AP, 'pgeom')['values']))
out['anchor_miss'] = q(lambda: m.load_anchor(AP, 'lego'))
out['fname'] = q(lambda: [m.grade_filename('P0'), m.grade_filename('M0')])
with tempfile.TemporaryDirectory() as td:
    d = pathlib.Path(td)
    (d / 'P0.shape.json').write_text(json.dumps({'hard_needle_frac': 0.9}))
    out['sidecar_unver'] = q(lambda: m.hard_needle_from_sidecar(d, 'P0'))
    (d / 'P0.shape.json').write_text(json.dumps(
        {'hard_needle_frac': 0.9, 'verified_against_report': True}))
    out['sidecar_ok'] = q(lambda: m.hard_needle_from_sidecar(d, 'P0'))
    # a ply whose scale fields are NOT at the INRIA offset
    props = ['x', 'y', 'z', 'scale_0', 'scale_1', 'scale_2', 'opacity',
             'rot_0', 'rot_1', 'rot_2', 'rot_3', 'e0', 'e1']
    n = 4
    rows = np.zeros((n, len(props)), dtype='<f4')
    for j, nm in enumerate(props):
        rows[:, j] = 7.0
    ls = np.log(np.array([[1e-4, 0.005, 1.0], [1e-4, 0.05, 1.0],
                          [1e-4, 0.09, 1.0], [1e-4, 0.5, 1.0]], dtype=np.float32))
    for j, nm in enumerate(('scale_0', 'scale_1', 'scale_2')):
        rows[:, props.index(nm)] = ls[:, j]
    hdr = ['ply', 'format binary_little_endian 1.0', f'element vertex {n}']
    hdr += [f'property float {x}' for x in props] + ['end_header', '']
    f = d / 'a.ply'
    f.write_bytes(chr(10).join(hdr).encode() + rows.tobytes())
    out['ply_shape'] = q(lambda: {k: round(v, 9) if isinstance(v, float) else v
                                  for k, v in p.shape_from_ply(f).items()})
    out['ply_avg'] = q(lambda: round(p.shape_from_ply(f, 'average')['aspect_p50'], 9))
    out['ply_scales'] = q(lambda: np.round(p.read_ply_scales(f), 6).tolist())
    out['ply_thr'] = q(lambda: [p.HARD_NEEDLE_ASPECT, p.NEEDLE_ASPECT,
                                sorted(p.REQUIRED_CHECKS)])
    out['lower_med'] = q(lambda: p.lower_median(np.array([1., 2., 3., 4.])))
    r = d / 'r.json'
    r.write_text(json.dumps({'metrics': {'shape': {'aspect_p50': 0.5,
                                                   'needle_frac': 0.25}}}))
    good = {'aspect_p50': 0.5, 'needle_frac': 0.25, 'smid_p50_mm': 1.0,
            'smax_p50_mm': 2.0, 'median_convention': 'lower'}
    out['xc_ok'] = q(lambda: sorted(p.cross_check(good, r)))
    out['xc_bad'] = q(lambda: p.cross_check(dict(good, needle_frac=0.30), r))
    r2 = d / 'r2.json'
    r2.write_text(json.dumps({'metrics': {'shape': {}}}))
    out['xc_empty'] = q(lambda: sorted(p.cross_check(good, r2)))
print(json.dumps(out, default=str, sort_keys=True))
"""


def module_fingerprint(env: dict) -> str:
    """A behavioural probe: what the modules COMPUTE, not what they contain.

    A mutant whose fingerprint is unchanged is behaviourally identical, and neither its
    survival nor its death is evidence about the tests.

    THE FIRST VERSION OF THIS PROBE WAS TOO NARROW AND SAID SO WRONGLY: it reported 12 of
    22 mutants "behaviourally identical" while the test suite killed 10 of them, i.e. the
    guard was producing false negatives about the very thing it exists to certify. It
    covered only happy paths, so every mutant that removed a GUARD -- the shape most of
    these are -- moved nothing it looked at. `q()` captures raised exceptions as values for
    exactly that reason: a check that stops raising must change the fingerprint.
    """
    r = subprocess.run([sys.executable, "-c", PROBE], cwd=str(ROOT),
                       capture_output=True, text=True, env=env)
    return r.stdout.strip() or ("ERR:" + r.stderr.strip()[-400:])


def main() -> None:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    base_failed, base_out = failed_tests(env)
    if base_failed:
        raise SystemExit(f"the suite is NOT GREEN before mutating: {sorted(base_failed)}\n"
                         f"{base_out}")
    base_fp = module_fingerprint(env)
    if base_fp.startswith("ERR:"):
        raise SystemExit(f"the behavioural probe does not run on clean code:\n{base_fp}")

    results, bad = [], []
    for name, path, old, new, must_kill, why in MUTANTS:
        src = path.read_text()
        if src.count(old) != 1:
            bad.append(f"{name}: the mutation target appears {src.count(old)} times in "
                       f"{path.name}, not once -- the mutant is not well-defined")
            continue
        path.write_text(src.replace(old, new, 1))
        try:
            fp = module_fingerprint(env)
            changed = fp != base_fp
            failed, out = failed_tests(env)
            killed = must_kill in failed
        finally:
            path.write_text(src)
        results.append({"mutant": name, "models": why, "must_kill": must_kill,
                        "behaviour_changed": changed, "killed_by_named_test": killed,
                        "also_failed": sorted(failed - {must_kill})})
        if not changed:
            bad.append(f"{name}: BEHAVIOURALLY IDENTICAL -- the fingerprint did not move, "
                       f"so this mutant proves nothing either way")
        if not killed:
            bad.append(f"{name}: SURVIVED its named test {must_kill!r}. "
                       f"Other failures: {sorted(failed)}\n{out}")

    restored, _ = failed_tests(env)
    if restored:
        bad.append(f"the suite is not green after restoring: {sorted(restored)}")

    print(json.dumps({"mutants": len(results),
                      "killed": sum(r["killed_by_named_test"] for r in results),
                      "behaviour_changed": sum(r["behaviour_changed"] for r in results),
                      "results": results}, indent=2))
    if bad:
        print("\n".join(["", "=== PROBLEMS ==="] + bad), file=sys.stderr)
        raise SystemExit(1)
    print(f"\nALL {len(results)} MUTANTS CHANGED BEHAVIOUR AND WERE KILLED BY THEIR "
          f"NAMED TEST; suite green before and after.")


if __name__ == "__main__":
    main()
