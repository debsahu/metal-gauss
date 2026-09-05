"""Task 20 escalation RUNNER -- the arm queue and every guard, tested without a GPU.

Each test says what it CATCHES. The mutation battery
(`tests/mutants_dn_gate_arms.py`) asserts on these names, never on a failure count.

The runner's job is to spend ~10 GPU-hours correctly. Nothing here grades an arm --
grading is a pure function of the artifacts and lands separately -- so every test below
is about PROVENANCE: that the arm which ran is the arm the pre-registration names, and
that an arm which did not is refused rather than queued behind three more.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import dn_gate_arms as H  # noqa: E402


# A stand-in trainer. It parses the two flags the guards care about, writes a report
# shaped like `_run_report`'s, and exits with whatever `--fake-rc` says. Using a real
# child process means `run_arm`'s watchdog, liveness poll, log capture, environment
# overlay and exit-status handling are all exercised for real rather than mocked.
FAKE_TRAINER = r'''
import json, os, sys, time
a = sys.argv[1:]
def opt(name, default=None):
    return a[a.index(name) + 1] if name in a else default
time.sleep(float(opt("--fake-sleep", "0")))
if opt("--fake-quiet") is None:
    print("fake trainer running", flush=True)
rep = opt("--fake-report-path") or opt("--report")
if opt("--fake-no-report") is None:
    counts = json.loads(opt("--fake-counts", '{"fused_calls": 0, "torch_calls": 3,'
                            ' "dn_gated_calls": 0, "dn_ungated_calls": 3,'
                            ' "dn_skipped_calls": 0}'))
    seen = {k: os.environ.get(k) for k in ("MG_TORCH_LOSS", "MG_DN_GATE_NEIGHBOURS",
                                           "PYTHONDONTWRITEBYTECODE")}
    resolved = {}
    i = 0
    while i < len(a):
        if a[i].startswith("--") and not a[i].startswith("--fake-"):
            dest = a[i][2:].replace("-", "_")
            if i + 1 < len(a) and not a[i + 1].startswith("--"):
                v = a[i + 1]
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
                resolved[dest] = v
                i += 2
                continue
            resolved[dest] = True
        i += 1
    json.dump({"schema": 1, "resolved": resolved, "env": {"git": "fake"},
               "observed": {"schema": 1, "loss_path": counts},
               "child_env": seen, "metrics": {}}, open(rep, "w"))
sys.exit(int(opt("--fake-rc", "0")))
'''


@pytest.fixture
def fake_trainer(monkeypatch, tmp_path):
    """Point `run_arm` at the stand-in, and make its poll fast enough to test."""
    monkeypatch.setattr(H, "CAFFEINATE", [])
    monkeypatch.setattr(H, "TRAINER_CMD", [sys.executable, "-c", FAKE_TRAINER])
    monkeypatch.setattr(H, "POLL_INTERVAL_S", 0.02)
    monkeypatch.setattr(H, "fix_openmp", lambda: None)
    monkeypatch.setattr(H, "clear_stale_lock", lambda: None)
    return tmp_path


def _args(**kw):
    """A parsed namespace, built through the harness's OWN parser so a test cannot
    inherit a default the command line does not have."""
    base = ["--scene", "pgeom", "--colmap", "/c/sparse/0", "--images", "/c/images",
            "--seed-cloud", "/c/sparse/0/points3D.tsdf.txt", "--out", "/o"]
    for k, v in kw.items():
        flag = "--" + k.replace("_", "-")
        base += [flag] if v is True else [flag, str(v)]
    return H.build_parser().parse_args(base)


# ============================================================ the treatment variable


def test_the_treatment_arm_sets_both_environment_variables():
    """CATCHES: a treatment arm that forgot MG_DN_GATE_NEIGHBOURS, or that set it on the
    fused chain. The pre-registration forces every arm onto the torch chain so that
    gate-vs-no-gate is not confounded with torch-vs-fused."""
    assert H.env_overlay("treatment") == {"MG_TORCH_LOSS": "1",
                                          "MG_DN_GATE_NEIGHBOURS": "1"}


def test_the_floor_arms_pin_the_gate_OFF_rather_than_leaving_it_inherited():
    """CATCHES: a floor arm silently gated because the operator's shell exported
    MG_DN_GATE_NEIGHBOURS=1. Leaving the variable out of the overlay would inherit it,
    and the floor would then measure the treatment."""
    assert H.env_overlay("floor") == {"MG_TORCH_LOSS": "1",
                                      "MG_DN_GATE_NEIGHBOURS": "0"}


def test_the_queue_is_three_floors_at_two_seeds_and_one_treatment():
    """CATCHES: an n=2 floor (section 8.2: 25-45x too small), a floor triple that shares
    one seed, or a treatment whose seed does not pair with F0."""
    q = H.arm_queue(_args(seed=42))
    assert [(x.tag, x.role, x.seed) for x in q] == [
        ("F0", "floor", 42), ("F1", "floor", 42), ("F2", "floor", 43),
        ("G0", "treatment", 42)]


def test_no_arm_passes_depth_source_and_the_trainer_would_reject_it_if_it_did():
    """CATCHES: porting Task 19's `--depth-source` dimension across. That flag is Task
    19's and does not exist on this branch, so an arm carrying it exits 2 -- which is why
    the second half of this test matters: it proves the flag is genuinely absent from the
    trainer rather than merely absent from our argv."""
    a = _args()
    for arm in H.arm_queue(a):
        assert "--depth-source" not in H.build_arm_argv(a, arm)
    from metal_gauss.train import build_parser as trainer_parser
    with pytest.raises(SystemExit):
        trainer_parser().parse_args(["--colmap", "x", "--images", "y",
                                     "--depth-source", "center"])


def test_every_arm_exports_a_checkpoint_every_500_steps():
    """CATCHES: losing Reading B. The early-divergence probe reads steps 500 and 2000, and
    those checkpoints exist only if every arm ran with --export-every 500."""
    a = _args()
    for arm in H.arm_queue(a):
        argv = H.build_arm_argv(a, arm)
        assert argv[argv.index("--export-every") + 1] == "500"


def test_masks_carry_an_EXPLICIT_polarity_and_both_flags_vanish_when_there_are_no_masks():
    """CATCHES: relying on `--mask-polarity auto`. Section 13.2 records auto resolving to
    `drop` on P-MASK, but `resolved` records the string "auto", not what it resolved to,
    so only an explicit value is provenance. Also catches passing --masks to P-GEOM,
    which has none."""
    a = _args(masks="/c/masks")
    argv = H.build_arm_argv(a, H.arm_queue(a)[0])
    assert argv[argv.index("--masks") + 1] == "/c/masks"
    assert argv[argv.index("--mask-polarity") + 1] == "drop"
    bare = H.build_arm_argv(_args(), H.arm_queue(_args())[0])
    assert "--masks" not in bare and "--mask-polarity" not in bare


def test_the_harness_refuses_to_be_asked_for_an_auto_polarity():
    """CATCHES: an operator reaching for `auto` on the command line. It is not offered,
    for the reason in the test above."""
    with pytest.raises(SystemExit):
        _args(masks="/c/masks", mask_polarity="auto")


# ============================================================ the extension lock


def _torch_build_dir(name="metal_gauss_metal"):
    from torch.utils.cpp_extension import _get_build_directory
    return Path(_get_build_directory(name, False))


def test_the_lock_path_matches_torchs_OWN_build_directory_under_a_redirect(monkeypatch,
                                                                          tmp_path):
    """CATCHES: the hardwired ~/Library/Caches path. The M4 Max redirects
    TORCH_EXTENSIONS_DIR into an isolated scratch tree, and torch does NOT append its
    py312_cpu segment in that case -- so a transcribed path is not merely stale, it names
    a directory that cannot exist. Checked against torch's own function so a torch upgrade
    that moves the directory fails here rather than in a silent guard."""
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "scratch"))
    assert H.extension_lock_path() == _torch_build_dir() / "lock"


def test_the_lock_path_matches_torchs_own_build_directory_with_no_redirect(monkeypatch,
                                                                          tmp_path):
    """CATCHES: getting the DEFAULT branch wrong while fixing the redirect branch. The
    py-version/accelerator segment only exists when TORCH_EXTENSIONS_DIR is unset."""
    monkeypatch.delenv("TORCH_EXTENSIONS_DIR", raising=False)
    import torch.utils.cpp_extension as ce
    monkeypatch.setattr(ce, "get_default_build_root", lambda: str(tmp_path / "home"))
    monkeypatch.setattr(H, "_default_build_root", lambda: str(tmp_path / "home"))
    assert H.extension_lock_path() == _torch_build_dir() / "lock"


def test_clear_stale_lock_removes_the_REDIRECTED_lock_and_leaves_the_home_one_alone(
        monkeypatch, tmp_path):
    """CATCHES: a guard that checks a path unrelated to the thing being checked -- the
    failure shape this project keeps repeating. A hardwired implementation deletes the
    decoy and leaves the lock that is actually hanging every run."""
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "scratch"))
    real = tmp_path / "scratch" / "metal_gauss_metal" / "lock"
    decoy = tmp_path / "home" / "py312_cpu" / "metal_gauss_metal" / "lock"
    for p in (real, decoy):
        p.parent.mkdir(parents=True)
        p.write_text("")
    monkeypatch.setattr(H, "_default_build_root", lambda: str(tmp_path / "home"))
    H.clear_stale_lock()
    assert not real.exists(), "the lock that is actually held was not cleared"
    assert decoy.exists(), "a lock outside the redirect was deleted"


def test_a_HELD_lock_is_refused_rather_than_removed(monkeypatch, tmp_path):
    """CATCHES: deleting a lock a concurrent build owns, which corrupts that build.
    `lsof` returning nothing is the only proof a lock is stale."""
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "scratch"))
    lock = tmp_path / "scratch" / "metal_gauss_metal" / "lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("")
    monkeypatch.setattr(H, "_lsof", lambda p: "python  123 me  5w  REG ...")
    with pytest.raises(SystemExit, match="HELD"):
        H.clear_stale_lock()
    assert lock.exists()


# ============================================================ run_arm


def _run(tmp, tag="F0", extra=(), **kw):
    return H.run_arm(tag, tmp, ["--fake-marker", "1", *extra],
                     kw.pop("env_overlay", {"MG_TORCH_LOSS": "1"}), **kw)


def test_run_arm_refuses_an_arm_that_wrote_a_report_and_then_DIED(fake_trainer):
    """CATCHES the exact defect the operator names: Task 21's harness recorded rc=0 on an
    `Abort trap: 6` and only noticed downstream when a ply was missing. The artifact
    existing is not the same claim as the process succeeding, and a harness must make
    both."""
    with pytest.raises(SystemExit, match="rc=6"):
        _run(fake_trainer, extra=["--fake-rc", "6"], watchdog_s=30.0)


def test_run_arm_refuses_an_arm_that_exited_cleanly_with_NO_report(fake_trainer):
    """CATCHES: the converse -- a run that returns 0 having produced nothing. A harness
    once printed six `done` lines in three seconds for six crashed runs."""
    with pytest.raises(SystemExit, match="NO REPORT"):
        _run(fake_trainer, extra=["--fake-no-report", "1"], watchdog_s=30.0)


def test_run_arm_refuses_an_arm_that_wrote_a_report_but_NO_EXPORT_PLY(fake_trainer):
    """CATCHES: an arm that died between the report and the export. `train.py` writes the
    report first (`:871`) and exports second (`:876`), so a report is NOT evidence of a
    ply -- and a missing ply is exactly how Task 21's rc=0-on-abort was eventually
    noticed, downstream and late. Verified by reading the write order, not assumed."""
    with pytest.raises(SystemExit, match="no export ply"):
        _run(fake_trainer, extra=["--export", str(fake_trainer / "F0.ply")],
             watchdog_s=30.0)


def test_an_arm_that_wrote_both_a_report_and_its_ply_is_accepted(fake_trainer):
    """Discriminating power for the test above: a guard that refused whenever --export was
    present would satisfy it while stopping every real arm."""
    ply = fake_trainer / "F0.ply"
    ply.write_bytes(b"ply\n")
    assert _run(fake_trainer, extra=["--export", str(ply)], watchdog_s=30.0).exists()


def test_the_environment_overlay_beats_an_exported_variable_in_the_operators_shell(
        fake_trainer, monkeypatch):
    """CATCHES: a floor arm inheriting MG_DN_GATE_NEIGHBOURS=1 from the shell. Asserted on
    what the CHILD saw, not on the dict we built, because the dict is the claim and the
    child is the measurement."""
    monkeypatch.setenv("MG_DN_GATE_NEIGHBOURS", "1")
    rep = _run(fake_trainer, watchdog_s=30.0, env_overlay=H.env_overlay("floor"))
    assert json.loads(rep.read_text())["child_env"] == {
        "MG_TORCH_LOSS": "1", "MG_DN_GATE_NEIGHBOURS": "0",
        "PYTHONDONTWRITEBYTECODE": "1"}


def test_a_report_with_no_recorded_exit_status_is_NOT_resumed(fake_trainer):
    """CATCHES: `if report.exists(): skip`. On resume, the exit status is gone -- so a
    report left by the crashed run above would be adopted as a finished arm. The status
    sidecar is what makes the skip a real check rather than a file-existence test."""
    (fake_trainer / "F0.json").write_text("{}")
    with pytest.raises(SystemExit, match="armstatus"):
        _run(fake_trainer, watchdog_s=30.0)


def test_an_arm_resumes_only_when_the_RECORDED_status_is_zero(fake_trainer):
    """CATCHES: a sidecar that is written but never read back. rc=6 must block the resume
    exactly as it blocked the original run."""
    rep = _run(fake_trainer, watchdog_s=30.0)
    st = fake_trainer / "F0.armstatus.json"
    doc = json.loads(st.read_text())
    assert doc["rc"] == 0
    again = _run(fake_trainer, watchdog_s=30.0)      # clean resume, no re-run
    assert again == rep
    doc["rc"] = 6
    st.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="rc=6"):
        _run(fake_trainer, watchdog_s=30.0)


def test_the_watchdog_kills_a_run_that_never_finishes(fake_trainer):
    """CATCHES: macOS has no `timeout`(1), so a hung arm would hold the queue forever."""
    with pytest.raises(SystemExit, match="watchdog"):
        _run(fake_trainer, extra=["--fake-sleep", "30"], watchdog_s=0.3)


def test_an_empty_log_past_the_liveness_window_is_refused_as_a_stale_lock(fake_trainer):
    """CATCHES: the FileBaton hang. A 0-byte log at 90 s is an impossible healthy state,
    and asserting only that the report exists at the END is structurally blind to a run
    that never reaches an end."""
    with pytest.raises(SystemExit, match="0-byte log"):
        _run(fake_trainer, extra=["--fake-sleep", "30", "--fake-quiet", "1"],
             watchdog_s=30.0, empty_log_s=0.2)


# ============================================================ observed counters


def _counts(**kw):
    d = dict(fused_calls=0, torch_calls=3, dn_gated_calls=0, dn_ungated_calls=3,
             dn_skipped_calls=0)
    d.update(kw)
    return {"observed": {"schema": 1, "loss_path": d}}


def test_a_floor_arm_that_ran_the_FUSED_kernel_is_refused():
    """CATCHES: a lost MG_TORCH_LOSS. The gated arm can only run on the torch chain, so a
    fused floor would confound gate-vs-no-gate with fused-vs-torch -- and `resolved`
    records no environment variable, so nothing else in the report could tell."""
    with pytest.raises(SystemExit, match="fused_calls"):
        H.check_loss_path("F0", "floor", _counts(fused_calls=3, torch_calls=0))


def test_a_floor_arm_whose_dn_term_ran_GATED_is_refused():
    """CATCHES: an inherited MG_DN_GATE_NEIGHBOURS=1 that survived the overlay -- the
    floor would then be a second treatment arm wearing a floor's tag."""
    with pytest.raises(SystemExit, match="dn_gated_calls"):
        H.check_loss_path("F1", "floor", _counts(dn_gated_calls=3, dn_ungated_calls=0))


def test_a_treatment_arm_whose_dn_term_ran_UNGATED_is_refused():
    """CATCHES: the treatment variable never reaching the child -- a typo, a nohup
    boundary, a non-inheriting shell. This is the arm the whole protocol turns on."""
    with pytest.raises(SystemExit, match="dn_gated_calls"):
        H.check_loss_path("G0", "treatment", _counts())


def test_an_all_zero_report_is_refused_rather_than_read_as_an_ungated_floor():
    """CATCHES: absence reading as agreement. An arm whose whole purpose is the geometry
    recipe and whose geometry never ran must not be gradeable at all -- which is why the
    trainer emits counts rather than booleans."""
    with pytest.raises(SystemExit, match="no geometry term ran"):
        H.check_loss_path("F0", "floor", _counts(torch_calls=0, dn_ungated_calls=0))


def test_a_report_violating_the_call_total_invariant_is_refused():
    """CATCHES: a report assembled by hand, or a trainer whose branches drifted apart.
    `fused + torch == gated + ungated + skipped` holds without knowing the schedule."""
    with pytest.raises(SystemExit, match="invariant"):
        H.check_loss_path("F0", "floor", _counts(torch_calls=4))


def test_a_report_with_no_observed_block_is_refused():
    """CATCHES: an arm run by a binary older than 39d5806. Its loss chain is simply
    unknown, and unknown must not grade as torch."""
    with pytest.raises(SystemExit, match="observed"):
        H.check_loss_path("F0", "floor", {"schema": 1, "resolved": {}})


def test_the_role_rules_can_TELL_THE_TWO_ROLES_APART():
    """CATCHES a fixture with no discriminating power. Each role's own counts must pass
    under its own role AND be refused under the other; a rule that accepted both would
    pass every test above that expects an acceptance while catching nothing."""
    floor = _counts()
    treatment = _counts(dn_gated_calls=3, dn_ungated_calls=0)
    H.check_loss_path("F0", "floor", floor)                     # accepted
    H.check_loss_path("G0", "treatment", treatment)             # accepted
    with pytest.raises(SystemExit):
        H.check_loss_path("F0", "floor", treatment)
    with pytest.raises(SystemExit):
        H.check_loss_path("G0", "treatment", floor)


# ============================================================ resolved vs requested


def test_a_report_whose_seed_is_not_the_one_asked_for_is_refused():
    """CATCHES: the F2 seed silently equalling F0's, which would turn the n=3 floor into
    an n=3 repeat of one seed."""
    a = _args(seed=42)
    arm = H.arm_queue(a)[2]                                     # F2, seed 43
    bad = H.expected_resolved(a, arm) | {"seed": 42}
    with pytest.raises(SystemExit, match="seed"):
        H.check_resolved(arm, a, bad)


def test_a_report_whose_export_every_is_not_500_is_refused():
    """CATCHES: the flag not surviving. Reading B is unrecoverable afterwards -- the
    checkpoints simply do not exist, and the arms cost ~10 GPU-hours to re-run."""
    a = _args()
    arm = H.arm_queue(a)[0]
    bad = H.expected_resolved(a, arm) | {"export_every": 0}
    with pytest.raises(SystemExit, match="export_every"):
        H.check_resolved(arm, a, bad)


def test_a_report_that_resolved_the_mask_polarity_to_auto_is_refused():
    """CATCHES: `auto` reaching the trainer anyway. `resolved` would then record the
    string "auto" and no artifact would say which polarity actually applied."""
    a = _args(masks="/c/masks")
    arm = H.arm_queue(a)[0]
    with pytest.raises(SystemExit, match="mask_polarity"):
        H.check_resolved(arm, a, H.expected_resolved(a, arm) | {"mask_polarity": "auto"})


def test_a_faithful_report_passes_the_resolved_check():
    """The discriminating-power half of the three tests above: if `check_resolved` refused
    everything they would all pass while catching nothing."""
    a = _args(masks="/c/masks", seed=42)
    for arm in H.arm_queue(a):
        H.check_resolved(arm, a, H.expected_resolved(a, arm))


def test_EVERY_flag_the_harness_puts_on_the_command_line_is_verified_in_the_report():
    """CATCHES an argument sink: a flag added to the argv whose survival nothing checks.
    Both historical wrong numbers in this repo (--steps-scaler, --budget) were exactly
    that. The check is derived from the argv rather than from a hand-kept list, so adding
    a flag without a matching assertion fails here."""
    a = _args(masks="/c/masks", init_ply="/c/seed.ply", depth_dir="/c/depth",
              normal_dir="/c/normal")
    for arm in H.arm_queue(a):
        argv = H.build_arm_argv(a, arm)
        flags = {x[2:].replace("-", "_") for x in argv if x.startswith("--")}
        unverified = flags - set(H.expected_resolved(a, arm)) - H.NOT_IN_RESOLVED
        assert not unverified, f"{arm.tag}: on the command line but never verified: {unverified}"


# ============================================================ the seed-cloud guard


def test_the_reference_cloud_may_not_be_the_COLMAP_points3D_the_trainer_SEEDED_FROM(tmp_path):
    """CATCHES the exact hole the operator names: the original refuses `points3D.txt` and
    `seed.ply` by NAME, and this scene's seed is `points3D.bin`, which sails through.
    Scoring thin-axis against the cloud the trainer initialised from was an 11.6 deg
    error once -- larger than every recipe gain in CLAUDE.md's table."""
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "points3D.bin").write_bytes(b"\0")
    with pytest.raises(SystemExit, match="the trainer initialised from"):
        H.check_seed_cloud(str(sparse / "points3D.bin"), str(sparse), None)


def test_the_reference_cloud_may_not_be_the_INIT_PLY(tmp_path):
    """CATCHES: the ARKitScenes-shaped case, where the seed lives outside the COLMAP model
    entirely and a name check has nothing to match on."""
    p = tmp_path / "dense_init.ply"
    p.write_bytes(b"ply\n")
    with pytest.raises(SystemExit, match="the trainer initialised from"):
        H.check_seed_cloud(str(p), str(tmp_path), str(p))


def test_the_guard_RESOLVES_paths_rather_than_comparing_strings(tmp_path):
    """CATCHES: a string comparison. A symlink, a relative path or a `..` segment all
    name the same file under a different spelling, and only the resolved path is the
    file."""
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "points3D.bin").write_bytes(b"\0")
    alias = tmp_path / "reference_cloud.txt"
    alias.symlink_to(sparse / "points3D.bin")
    with pytest.raises(SystemExit, match="the trainer initialised from"):
        H.check_seed_cloud(str(alias), str(sparse), None)


def test_the_NAME_blacklist_is_kept_alongside_the_resolution_check(tmp_path):
    """CATCHES: dropping the cheap check when adding the precise one. A `points3D.txt`
    from some OTHER model is still overwhelmingly likely to be a seed cloud, and the
    resolution check cannot see that."""
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "points3D.txt").write_text("")
    with pytest.raises(SystemExit, match="named"):
        H.check_seed_cloud(str(other / "points3D.txt"), str(tmp_path / "sparse"), None)


def test_a_TSDF_reference_cloud_beside_the_model_is_ACCEPTED(tmp_path):
    """The discriminating-power half: `points3D.tsdf.txt` sits in the very same directory
    as the seed and is the cloud the protocol REQUIRES. A guard that refused it would make
    every test above pass while making the harness unusable."""
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "points3D.bin").write_bytes(b"\0")
    (sparse / "points3D.tsdf.txt").write_text("")
    H.check_seed_cloud(str(sparse / "points3D.tsdf.txt"), str(sparse), None)


def test_the_refusal_keys_on_WHAT_THE_TRAINER_SEEDED_FROM_not_on_the_file_being_a_ply(
        tmp_path):
    """CATCHES a guard that refuses every `.ply`, or that reads only the COLMAP location.
    One variable moves: the SAME reference cloud is accepted when `--init-ply` is absent
    (the trainer seeded from the COLMAP model) and refused when `--init-ply` names it.

    Note what this does NOT claim. A `points3D.bin` displaced by `--init-ply` is still
    refused, by the unconditional name blacklist -- so this A/B has to use a file whose
    name is not on it, or the two halves would both be decided by the blacklist and the
    test would measure nothing about the resolution check."""
    p = tmp_path / "dense_init.ply"
    p.write_bytes(b"ply\n")
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "points3D.bin").write_bytes(b"\0")
    H.check_seed_cloud(str(p), str(sparse), None)                    # accepted
    with pytest.raises(SystemExit, match="the trainer initialised from"):
        H.check_seed_cloud(str(p), str(sparse), str(p))              # refused


# ============================================================ preflight


def test_the_disk_preflight_refuses_a_run_that_cannot_hold_its_checkpoints(monkeypatch,
                                                                          tmp_path):
    """CATCHES: filling the disk 6 hours in. `--export-every 500` over 30k steps is 60
    checkpoints per arm; at a 500k budget and 59 float32 ply fields that is ~7 GB per arm
    and ~28 GB per scene, which nothing else in the protocol mentions."""
    monkeypatch.setattr(H, "_free_bytes", lambda p: 1 << 30)     # 1 GiB
    with pytest.raises(SystemExit, match="disk"):
        H.check_disk(tmp_path, _args(steps=30000, budget=500000))


def test_the_disk_preflight_passes_when_there_is_room(monkeypatch, tmp_path):
    """Discriminating power: a preflight that always refused would satisfy the test above
    and stop every run."""
    monkeypatch.setattr(H, "_free_bytes", lambda p: 400 << 30)
    H.check_disk(tmp_path, _args(steps=30000, budget=500000))
