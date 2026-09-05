#!/usr/bin/env python3
"""Mutation battery for tests/test_dn_gate_arms.py.

Every mutant is (a) PROVEN to change behaviour by a probe that CALLS the code -- never by
reading source text, which is the error research/metal-gauss.md 13.5 item 5 records -- and
then (b) required to kill a NAMED test. A failure COUNT would be satisfied by any failure,
including one the mutant caused by accident.

The probe's signature is normalised against the temporary directory it runs in, and
exceptions are recorded as their type plus which of a FIXED keyword list appears in the
message. That keeps the signature stable across runs while staying sensitive to a message
that stops naming the thing it caught -- which is the only way a reader learns what went
wrong at 3am, six hours into a ten-hour queue.

`PYTHONDONTWRITEBYTECODE=1` is set here rather than left to the caller: with stale .pyc
files a battery fails toward FALSE SURVIVED.

    .venv/bin/python tests/mutants_dn_gate_arms.py
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
import json, os, sys, tempfile
from pathlib import Path
sys.path.insert(0, %r); sys.path.insert(0, %r); sys.path.insert(0, %r)
import dn_gate_arms as H
from test_dn_gate_arms import FAKE_TRAINER

KEYWORDS = ["fused_calls", "torch_calls", "dn_gated_calls", "dn_ungated_calls",
            "dn_skipped_calls", "no geometry term ran", "invariant", "observed", "named",
            "the trainer initialised from", "disk", "watchdog", "0-byte log", "NO REPORT",
            "rc=6", "armstatus", "no export ply", "HELD", "settings other than",
            "seed", "export_every", "mask_polarity", "unknown arm role"]
TMP = Path(tempfile.mkdtemp())

def norm(x):
    return json.loads(json.dumps(x, default=str).replace(str(TMP), "<TMP>"))

def guarded(fn, *a, **kw):
    try:
        return {"ok": norm(fn(*a, **kw))}
    except BaseException as e:
        return {"raised": type(e).__name__,
                "keywords": [k for k in KEYWORDS if k in str(e)]}

sig = {}

# ---- pure: overlays, queue, argv, expected_resolved
sig["overlay"] = {r: guarded(H.env_overlay, r) for r in ("floor", "treatment", "bogus")}

def args(**kw):
    base = ["--scene", "s", "--colmap", "/c/sparse/0", "--images", "/c/images",
            "--seed-cloud", "/c/sparse/0/points3D.tsdf.txt", "--out", "/o"]
    for k, v in kw.items():
        base += ["--" + k.replace("_", "-"), str(v)]
    return H.build_parser().parse_args(base)

A = args(seed=42, masks="/c/masks", init_ply="/c/init.ply", depth_dir="/c/d",
         normal_dir="/c/n")
sig["queue"] = guarded(lambda: [list(x) for x in H.arm_queue(A)])
sig["argv"] = guarded(lambda: [H.build_arm_argv(A, x) for x in H.arm_queue(A)])
sig["expected"] = guarded(lambda: [H.expected_resolved(A, x) for x in H.arm_queue(A)])
sig["argv_nomask"] = guarded(lambda: H.build_arm_argv(args(), H.arm_queue(args())[0]))
sig["polarity_auto_refused"] = guarded(args, masks="/c/masks", mask_polarity="auto")

# ---- check_resolved
_arm = H.arm_queue(A)[2]
sig["resolved_ok"] = guarded(H.check_resolved, _arm, A, H.expected_resolved(A, _arm))
for k, v in (("seed", 42), ("export_every", 0), ("mask_polarity", "auto")):
    sig["resolved_" + k] = guarded(H.check_resolved, _arm, A,
                                   H.expected_resolved(A, _arm) | {k: v})

# ---- extension lock path, both env states
os.environ["TORCH_EXTENSIONS_DIR"] = str(TMP / "scratch")
sig["lock_redirected"] = guarded(lambda: str(H.extension_lock_path()))
os.environ.pop("TORCH_EXTENSIONS_DIR")
sig["lock_default_tail"] = guarded(
    lambda: "/".join(str(H.extension_lock_path()).rsplit("/", 3)[1:]))

# ---- clear_stale_lock: the redirected lock goes, an unrelated one stays
os.environ["TORCH_EXTENSIONS_DIR"] = str(TMP / "scratch")
real = TMP / "scratch" / "metal_gauss_metal" / "lock"
decoy = TMP / "home" / "py312_cpu" / "metal_gauss_metal" / "lock"
for p in (real, decoy):
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text("")
_orig_root = H._default_build_root
H._default_build_root = lambda: str(TMP / "home")
sig["clear_lock"] = guarded(H.clear_stale_lock)
sig["clear_lock_files"] = {"redirected_gone": not real.exists(), "decoy_kept": decoy.exists()}
real.parent.mkdir(parents=True, exist_ok=True); real.write_text("")
_orig_lsof = H._lsof
H._lsof = lambda p: "python 123 me 5w REG"
sig["held_lock"] = guarded(H.clear_stale_lock)
sig["held_lock_survived"] = real.exists()
H._lsof = _orig_lsof
H._default_build_root = _orig_root
real.unlink(missing_ok=True)

# ---- check_loss_path
def counts(**kw):
    d = dict(fused_calls=0, torch_calls=3, dn_gated_calls=0, dn_ungated_calls=3,
             dn_skipped_calls=0)
    d.update(kw); return {"observed": {"schema": 1, "loss_path": d}}
FLOORC, TREATC = counts(), counts(dn_gated_calls=3, dn_ungated_calls=0)
cases = {"floor_as_floor": ("floor", FLOORC), "treat_as_treat": ("treatment", TREATC),
         "floor_as_treat": ("treatment", FLOORC), "treat_as_floor": ("floor", TREATC),
         "fused_floor": ("floor", counts(fused_calls=3, torch_calls=0)),
         "all_zero": ("floor", counts(torch_calls=0, dn_ungated_calls=0)),
         "invariant": ("floor", counts(torch_calls=4)),
         "no_observed": ("floor", {"schema": 1, "resolved": {}}),
         "bogus_role": ("nonsense", FLOORC)}
sig["loss_path"] = {k: guarded(H.check_loss_path, "X", r, rep) for k, (r, rep) in cases.items()}

# ---- check_seed_cloud
sp = TMP / "sparse" / "0"; sp.mkdir(parents=True, exist_ok=True)
(sp / "points3D.bin").write_bytes(b"\0"); (sp / "points3D.tsdf.txt").write_text("")
alias = TMP / "reference_cloud.txt"
if not alias.exists(): alias.symlink_to(sp / "points3D.bin")
initp = TMP / "dense_init.ply"; initp.write_bytes(b"ply\n")
other = TMP / "elsewhere"; other.mkdir(exist_ok=True); (other / "points3D.txt").write_text("")
sig["seed_cloud"] = {
  "colmap_bin":  guarded(H.check_seed_cloud, str(sp / "points3D.bin"), str(sp), None),
  "init_ply":    guarded(H.check_seed_cloud, str(initp), str(sp), str(initp)),
  "symlink":     guarded(H.check_seed_cloud, str(alias), str(sp), None),
  "other_txt":   guarded(H.check_seed_cloud, str(other / "points3D.txt"), str(sp), None),
  "tsdf_ok":     guarded(H.check_seed_cloud, str(sp / "points3D.tsdf.txt"), str(sp), None),
  "ply_not_seed": guarded(H.check_seed_cloud, str(initp), str(sp), None)}

# ---- check_disk
_free = H._free_bytes
H._free_bytes = lambda p: 1 << 30
sig["disk_tight"] = guarded(H.check_disk, TMP, args(steps=30000, budget=500000))
H._free_bytes = lambda p: 400 << 30
sig["disk_roomy"] = guarded(H.check_disk, TMP, args(steps=30000, budget=500000))
H._free_bytes = _free

# ---- run_arm, against a real child process
H.CAFFEINATE = []
H.TRAINER_CMD = [sys.executable, "-c", FAKE_TRAINER]
H.POLL_INTERVAL_S = 0.02
H.fix_openmp = lambda: None
H.clear_stale_lock = lambda: None
os.environ["MG_DN_GATE_NEIGHBOURS"] = "1"      # the operator's shell, exporting the gate

def leg(name, tag, extra, overlay=None, **kw):
    d = TMP / "run" / name; d.mkdir(parents=True, exist_ok=True)
    r = guarded(H.run_arm, tag, d, list(extra),
                overlay if overlay is not None else H.env_overlay("floor"), **kw)
    rep = d / (tag + ".json")
    if rep.exists():
        try:
            r["child_env"] = json.loads(rep.read_text()).get("child_env")
        except Exception:
            pass
    return r

sig["run"] = {}
sig["run"]["died_after_report"] = leg("a", "F0", ["--fake-rc", "6"], watchdog_s=30.0)
sig["run"]["no_report"] = leg("b", "F0", ["--fake-no-report", "1"], watchdog_s=30.0)
sig["run"]["overlay_wins"] = leg("c", "F0", [], watchdog_s=30.0)
d = TMP / "run" / "d"; d.mkdir(parents=True, exist_ok=True)
sig["run"]["no_export_ply"] = leg("d", "F0", ["--export", str(d / "F0.ply")], watchdog_s=30.0)
e = TMP / "run" / "e"; e.mkdir(parents=True, exist_ok=True); (e / "F0.ply").write_bytes(b"ply\n")
sig["run"]["with_export_ply"] = leg("e", "F0", ["--export", str(e / "F0.ply")], watchdog_s=30.0)
f = TMP / "run" / "f"; f.mkdir(parents=True, exist_ok=True); (f / "F0.json").write_text("{}")
sig["run"]["resume_no_status"] = leg("f", "F0", [], watchdog_s=30.0)
sig["run"]["clean"] = leg("g", "F0", [], watchdog_s=30.0)
sig["run"]["clean_resume"] = leg("g", "F0", [], watchdog_s=30.0)
st = TMP / "run" / "g" / "F0.armstatus.json"
doc = json.loads(st.read_text()); doc["rc"] = 6; st.write_text(json.dumps(doc))
sig["run"]["resume_rc6"] = leg("g", "F0", [], watchdog_s=30.0)
sig["run"]["watchdog"] = leg("h", "F0", ["--fake-sleep", "30"], watchdog_s=0.3)
sig["run"]["empty_log"] = leg("i", "F0", ["--fake-sleep", "30", "--fake-quiet", "1"],
                              watchdog_s=30.0, empty_log_s=0.2)

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
    assert text.count(old) >= 1, f"mutation anchor not found:\n{old}"
    return text.replace(old, new, count)


# (name, mutate(src) -> src, the test whose NAME must fail)
MUTANTS = [
    ("floor_overlay_omits_the_gate_variable",
     lambda s: sub(s, 'return {"MG_TORCH_LOSS": "1", "MG_DN_GATE_NEIGHBOURS": "0"}',
                   'return {"MG_TORCH_LOSS": "1"}'),
     "test_the_floor_arms_pin_the_gate_OFF_rather_than_leaving_it_inherited"),

    ("treatment_left_on_the_fused_chain",
     lambda s: sub(s, 'return {"MG_TORCH_LOSS": "1", "MG_DN_GATE_NEIGHBOURS": "1"}',
                   'return {"MG_DN_GATE_NEIGHBOURS": "1"}'),
     "test_the_treatment_arm_sets_both_environment_variables"),

    ("floor_triple_shares_one_seed",
     lambda s: sub(s, 'Arm("F2", FLOOR, a.seed + 1)', 'Arm("F2", FLOOR, a.seed)'),
     "test_the_queue_is_three_floors_at_two_seeds_and_one_treatment"),

    ("port_the_depth_source_dimension_across",
     lambda s: sub(s, '    argv = ["--colmap", a.colmap,',
                   '    argv = ["--depth-source", "center", "--colmap", a.colmap,'),
     "test_no_arm_passes_depth_source_and_the_trainer_would_reject_it_if_it_did"),

    ("export_every_lost",
     lambda s: sub(s, "EXPORT_EVERY = 500  ", "EXPORT_EVERY = 0  "),
     "test_every_arm_exports_a_checkpoint_every_500_steps"),

    ("mask_polarity_left_to_auto",
     lambda s: sub(s, 'argv += ["--masks", a.masks, "--mask-polarity", a.mask_polarity]',
                   'argv += ["--masks", a.masks, "--mask-polarity", "auto"]'),
     "test_masks_carry_an_EXPLICIT_polarity_and_both_flags_vanish_when_there_are_no_masks"),

    # THE WRONG FIX MOST LIKELY TO BE WRITTEN LATER: swap the prefix, keep the
    # `py312_cpu` segment. torch adds that segment ONLY on the default branch, so this
    # rebuilds the very defect the derivation removes -- a guard on a path that cannot
    # exist. Named by the operator before the code was written.
    ("lock_path_substitutes_the_prefix_and_keeps_the_py_segment",
     lambda s: sub(s, '    root = os.environ.get("TORCH_EXTENSIONS_DIR")\n'
                      "    if root is None:\n",
                   '    root = os.environ.get("TORCH_EXTENSIONS_DIR")\n'
                   "    if True:\n"),
     "test_the_lock_path_matches_torchs_OWN_build_directory_under_a_redirect"),

    ("lock_path_hardwired_to_the_home_cache",
     lambda s: sub(s, '    root = os.environ.get("TORCH_EXTENSIONS_DIR")\n',
                   "    root = None\n"),
     "test_clear_stale_lock_removes_the_REDIRECTED_lock_and_leaves_the_home_one_alone"),

    ("a_held_lock_is_deleted_anyway",
     lambda s: sub(s, "    if held.strip():\n", "    if False:\n"),
     "test_a_HELD_lock_is_refused_rather_than_removed"),

    ("the_report_existing_is_treated_as_success",
     lambda s: sub(s, "    if rc != 0:\n        raise SystemExit(\n"
                      '            f"{tag}: the report exists but the process exited '
                      'rc={rc}.',
                   "    if False:\n        raise SystemExit(\n"
                   '            f"{tag}: the report exists but the process exited '
                   'rc={rc}.'),
     "test_run_arm_refuses_an_arm_that_wrote_a_report_and_then_DIED"),

    ("resume_on_the_report_alone",
     lambda s: sub(s, "        if not status.exists():\n", "        if False:\n"),
     "test_a_report_with_no_recorded_exit_status_is_NOT_resumed"),

    ("resume_ignores_the_recorded_status",
     lambda s: sub(s, '        if st.get("rc") != 0:\n', "        if False:\n"),
     "test_an_arm_resumes_only_when_the_RECORDED_status_is_zero"),

    ("the_export_ply_is_never_checked",
     lambda s: sub(s, '    if "--export" in argv:\n', "    if False:\n"),
     "test_run_arm_refuses_an_arm_that_wrote_a_report_but_NO_EXPORT_PLY"),

    ("an_all_zero_report_reads_as_an_ungated_floor",
     lambda s: sub(s, "    if all(lp[k] == 0 for k in LOSS_PATH_KEYS):\n",
                   "    if False:\n"),
     "test_an_all_zero_report_is_refused_rather_than_read_as_an_ungated_floor"),

    ("the_two_role_rules_are_swapped",
     lambda s: sub(s, "ROLE_LOSS_PATH = {\n    FLOOR:", "ROLE_LOSS_PATH = {\n    TREATMENT:")
                   .replace("    TREATMENT: {\"torch_calls\": \"gt0\", \"fused_calls\": "
                            "\"eq0\",\n                \"dn_gated_calls\": \"gt0\", "
                            "\"dn_ungated_calls\": \"eq0\"},",
                            "    FLOOR: {\"torch_calls\": \"gt0\", \"fused_calls\": "
                            "\"eq0\",\n                \"dn_gated_calls\": \"gt0\", "
                            "\"dn_ungated_calls\": \"eq0\"},"),
     "test_the_role_rules_can_TELL_THE_TWO_ROLES_APART"),

    ("the_call_total_invariant_is_not_checked",
     lambda s: sub(s, '    if lp["fused_calls"] + lp["torch_calls"] != ',
                   '    if False and lp["fused_calls"] + lp["torch_calls"] != '),
     "test_a_report_violating_the_call_total_invariant_is_refused"),

    ("a_report_with_no_observed_block_is_tolerated",
     lambda s: sub(s, "    if not isinstance(obs, dict) or not isinstance("
                      'obs.get("loss_path"), dict):\n', "    if False:\n"),
     "test_a_report_with_no_observed_block_is_refused"),

    ("the_resolved_check_is_an_argument_sink",
     lambda s: sub(s, "    if bad:\n        raise SystemExit(f\"{arm.tag}: the trainer "
                      "ran with settings",
                   "    if False:\n        raise SystemExit(f\"{arm.tag}: the trainer "
                   "ran with settings"),
     "test_a_report_whose_export_every_is_not_500_is_refused"),

    ("a_flag_is_added_to_the_argv_without_an_assertion",
     lambda s: sub(s, '            "--seed", str(arm.seed),',
                   '            "--sh-degree", "3",\n            "--seed", str(arm.seed),'),
     "test_EVERY_flag_the_harness_puts_on_the_command_line_is_verified_in_the_report"),

    ("the_seed_cloud_guard_is_a_NAME_blacklist_again",
     lambda s: sub(s, "    if ref in seeds:\n", "    if False:\n"),
     "test_the_guard_RESOLVES_paths_rather_than_comparing_strings"),

    ("the_seed_cloud_guard_compares_strings",
     lambda s: sub(s, "def _resolve(p: str | os.PathLike) -> Path:\n"
                      "    return Path(p).expanduser().resolve()",
                   "def _resolve(p: str | os.PathLike) -> Path:\n    return Path(p)"),
     "test_the_guard_RESOLVES_paths_rather_than_comparing_strings"),

    ("the_name_blacklist_is_dropped_once_resolution_exists",
     lambda s: sub(s, 'SEED_CLOUD_NAMES = ("points3D.txt", "points3D.bin", "seed.ply")',
                   "SEED_CLOUD_NAMES = ()"),
     "test_the_NAME_blacklist_is_kept_alongside_the_resolution_check"),

    ("the_disk_preflight_never_refuses",
     lambda s: sub(s, "    if free < need:\n", "    if False:\n"),
     "test_the_disk_preflight_refuses_a_run_that_cannot_hold_its_checkpoints"),

    ("the_watchdog_never_fires",
     lambda s: sub(s, "            if el > watchdog_s:\n", "            if False:\n"),
     "test_the_watchdog_kills_a_run_that_never_finishes"),

    ("the_empty_log_liveness_test_never_fires",
     lambda s: sub(s, "            if el > empty_log_s and log.stat().st_size == 0:\n",
                   "            if False:\n"),
     "test_an_empty_log_past_the_liveness_window_is_refused_as_a_stale_lock"),

    ("the_operators_shell_beats_the_overlay",
     lambda s: sub(s, "    env.update(overlay)\n",
                   "    env.update({k: v for k, v in overlay.items() "
                   "if k not in os.environ})\n"),
     "test_the_environment_overlay_beats_an_exported_variable_in_the_operators_shell"),
]

NODES = ["tests/test_dn_gate_arms.py"]


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
