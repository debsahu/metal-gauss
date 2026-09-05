#!/usr/bin/env python3
"""Task 20 escalation arm -- THE RUNNER. Four arms per scene, every guard, no grading.

    scripts/dn_gate_arms.py --scene pgeom --colmap DIR --images DIR --seed-cloud PATH
                            [--masks DIR] [--depth-dir DIR] [--normal-dir DIR]
                            [--init-ply PATH] [--max-resolution N] --out DIR

Ported from `feat/plane-aux`'s `scripts/plane_aux_arms.py` (phase ordering, the arm queue,
the floor triple, the config-vs-report checks) and `scripts/plane_aux_throughput.py`
(`run_arm`, `clear_stale_lock`, `fix_openmp`, the hand-rolled watchdog, the empty-log
liveness test, the `uv run --frozen` OpenMP revert). That branch is unmerged and carries
`--depth-source` and kernel changes that DO NOT EXIST HERE, so this is a port and not a
merge: an arm that passed `--depth-source` would exit 2 before doing any work.

WHAT THIS FILE IS FOR, AND WHAT IT DELIBERATELY IS NOT. Grading is a pure function of the
artifacts -- that is the whole claim `--regrade` exists to prove -- so it does not have to
exist before ~10 GPU-hours of arms can start. This file runs the arms and refuses the ones
that are not what they claim to be. The battery, the three bands, the early-divergence
probe, the anchors and `--summary` land beside it.

## The arms (pre-registration `e94ee45` section 1)

    F0  R1 ungated  seed 42   }
    F1  R1 ungated  seed 42   }  the n=3 floor for THIS arm's own configuration
    F2  R1 ungated  seed 43   }
    G0  R1 GATED    seed 42      the treatment

R1 = `--depth-loss-weight 1.0 --normal-loss-weight 0.2 --depth-normal-weight 0.05
--flatten-loss-weight 1.0 --depth-loss-space disparity`, at `--num-downscales 0
--budget 500000 --steps 30000 --eval-split-every 8 --eval-every 2500 --export-every 500`.

## The treatment variable is an ENVIRONMENT VARIABLE, and that is the whole risk

`_run_report` records `vars(args)` and an env snapshot, and NEITHER carries an environment
variable. So this harness does two things a flag-based one would not need:

  * it sets the variable EXPLICITLY on every arm, including `MG_DN_GATE_NEIGHBOURS=0` on
    the floors. Leaving it out would inherit whatever the operator's shell exported, and a
    floor measured under an inherited gate is a second treatment arm wearing a floor's tag;
  * it asserts `report["observed"]["loss_path"]` (`39d5806`) per arm ROLE, so an arm whose
    variable was lost across a nohup, a typo or a non-inheriting shell is REFUSED rather
    than graded on the strength of nothing. Absence never reads as agreement: an all-zero
    report means no geometry term ran at all and is refused too.

## Every arm asserts BOTH that the artifact exists AND that the process succeeded

Task 21's harness recorded `rc=0` on an `Abort trap: 6` (research/metal-gauss.md 14.8a) and
only caught it downstream when a ply was missing. `plane_aux_throughput.run_arm` checks
only that the report exists, which is the other half of the same blind spot. This checks
both, and writes `<tag>.armstatus.json` so a RESUME is a real check too -- on resume the
exit status is gone, and `if report.exists(): skip` would adopt the crashed run's report as
a finished arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path

# MG_ROOT, never `__file__`. research/metal-gauss.md 11.3c: freeze everything a long job
# READS, and launch from an immutable snapshot -- both bash and python re-read source, so
# editing this file while arms are running splices a hybrid into the queue. Copy it
# somewhere and set MG_ROOT to the checkout.
ROOT = Path(os.environ.get("MG_ROOT") or Path(__file__).resolve().parents[1])
if not (ROOT / "metal_gauss" / "train.py").exists():
    raise SystemExit(f"MG_ROOT={ROOT} is not a metal-gauss checkout (no "
                     f"metal_gauss/train.py). Set MG_ROOT when running from a snapshot.")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Monkeypatchable so the guards can be exercised against a stand-in child in the tests.
# A mocked subprocess would test the mock; a real child tests the watchdog, the liveness
# poll, the log capture, the environment overlay and the exit status.
CAFFEINATE = ["caffeinate", "-i"]
TRAINER_CMD = [str(ROOT / ".venv/bin/python"), "-m", "metal_gauss.train"]
POLL_INTERVAL_S = 2.0

EXT_NAME = "metal_gauss_metal"

# Fixed by the pre-registration, so they are constants rather than flags. A knob is a way
# for an arm to differ from the design without the command line saying so.
EXPORT_EVERY = 500          # Reading B grades steps 500 and 2000; no flag may lose these
EVAL_EVERY = 2500
EVAL_SPLIT_EVERY = 8
NUM_DOWNSCALES = 0
DEPTH_LOSS_WEIGHT = 1.0
NORMAL_LOSS_WEIGHT = 0.2
FLATTEN_LOSS_WEIGHT = 1.0

FLOOR = "floor"
TREATMENT = "treatment"

Arm = namedtuple("Arm", "tag role seed")

# `export_ply` writes 59 float32 fields per splat (x,y,z + f_dc*3 + f_rest*45 + opacity +
# scale*3 + rot*4). Used only by the disk preflight, which is an UPPER bound: with --grow
# the early checkpoints carry fewer than `budget` splats.
PLY_BYTES_PER_SPLAT = 59 * 4

#: Flags this harness puts on the trainer's command line whose survival is deliberately
#: NOT asserted against the report. Empty, and meant to stay that way: an unasserted flag
#: is an argument sink, and both wrong numbers this project has published (--steps-scaler,
#: then --budget) were exactly that. `--report` is added by `run_arm`, not by
#: `build_arm_argv`, and is asserted by the report existing at that path at all.
NOT_IN_RESOLVED: frozenset[str] = frozenset()


# ============================================================ the extension build lock


def _default_build_root() -> str:
    """torch's own default extensions root. Indirected so a test can redirect it."""
    from torch.utils.cpp_extension import get_default_build_root
    return get_default_build_root()


def extension_build_dir(name: str = EXT_NAME) -> Path:
    """Where `torch.utils.cpp_extension.load` builds our Metal extension.

    DERIVED, NOT TRANSCRIBED, and that is adaptation 4 of this port.
    `plane_aux_throughput.py` hardwires
    `~/Library/Caches/torch_extensions/py312_cpu/metal_gauss_metal/lock`. The M4 Max --
    the machine every arm of this protocol runs on -- redirects `TORCH_EXTENSIONS_DIR`
    into an isolated scratch tree, and torch does **not** append its `py312_cpu` segment
    in that case (`cpp_extension._get_build_directory`: the segment is added only on the
    `TORCH_EXTENSIONS_DIR is None` branch). So the hardwired path there is not merely
    stale, it names a directory that CANNOT EXIST -- and `clear_stale_lock` then reports
    success for a reason unrelated to the thing it is checking, which is the failure shape
    this project keeps repeating.

    `tests/test_dn_gate_arms.py` compares this against torch's own `_get_build_directory`
    under both env states, so a torch upgrade that moves the directory fails a test rather
    than silently disarming the guard.
    """
    root = os.environ.get("TORCH_EXTENSIONS_DIR")
    if root is None:
        import torch
        if torch.version.hip is not None:
            acc = f'rocm{torch.version.hip.replace(".", "")}'
        elif torch.version.cuda is not None:
            acc = f'cu{torch.version.cuda.replace(".", "")}'
        else:
            acc = "cpu"
        pyv = f'py{sys.version_info.major}{sys.version_info.minor}' \
              f'{getattr(sys, "abiflags", "")}'
        root = os.path.join(_default_build_root(), f"{pyv}_{acc}")
    return Path(root) / name


def extension_lock_path() -> Path:
    return extension_build_dir() / "lock"


def _lsof(path: Path) -> str:
    return subprocess.run(["lsof", str(path)], capture_output=True, text=True).stdout


def clear_stale_lock() -> None:
    """Remove the `FileBaton` lock, but ONLY if nothing holds it.

    A process killed between `try_acquire` and `release` never releases it, and
    `FileBaton.wait()` has no timeout -- so one dead run makes every later run spin
    forever with a 0-byte log (CONTRIBUTING.md, research/metal-gauss.md 11.6b). A HELD
    lock means a real concurrent build and removing it would corrupt that build; `lsof`
    returning nothing is the proof that it is stale.

    The path is printed on every call, because a guard whose target is derived from the
    environment should say out loud which file it looked at.
    """
    lock = extension_lock_path()
    if not lock.exists():
        print(f"  extension lock clear ({lock})", flush=True)
        return
    held = _lsof(lock)
    if held.strip():
        raise SystemExit(f"the extension lock {lock} is HELD by another build:\n{held}")
    lock.unlink()
    print(f"  cleared stale lock {lock}", flush=True)


def fix_openmp() -> None:
    """Re-point the four vendored `libomp.dylib` copies at one real library.

    `uv run --frozen` reconciles the venv against the lockfile and restores the vendored
    copy over the symlink, so this is re-applied after every arm rather than once.
    """
    subprocess.run([sys.executable, str(ROOT / "scripts/fix_openmp.py")],
                   capture_output=True, text=True)


# ============================================================ the arm queue


def env_overlay(role: str) -> dict:
    """The environment every arm of `role` runs under.

    `MG_TORCH_LOSS=1` ON EVERY ARM, floors included: the fused kernel refuses the gate
    flag by design, so an ungated floor measured on the fused path would confound
    gate-vs-no-gate with fused-vs-torch. Cost is section 11's 96.50 vs 79.50 ms/step.

    `MG_DN_GATE_NEIGHBOURS` is set EXPLICITLY on both roles, including to "0" on the
    floors. Omitting it would inherit the operator's shell, and there is no flag and no
    `resolved` key that would ever say so.
    """
    if role == TREATMENT:
        return {"MG_TORCH_LOSS": "1", "MG_DN_GATE_NEIGHBOURS": "1"}
    if role == FLOOR:
        return {"MG_TORCH_LOSS": "1", "MG_DN_GATE_NEIGHBOURS": "0"}
    raise SystemExit(f"unknown arm role {role!r}")


def arm_queue(a) -> list[Arm]:
    """F0/F1 at `--seed`, F2 at `--seed + 1`, then the treatment at `--seed`.

    n=3, not n=2: section 8.2 is this project's record of an n=2 floor coming out 25-45x
    too small and taking a day's conclusions with it. Two floors share a seed (the repeat
    pair) and the third does not (the seed floor); the treatment shares F0's seed so the
    paired comparison exists even though grading is against the n=3 spread.
    """
    t_seed = a.seed if a.treatment_seed is None else a.treatment_seed
    if a.treatment_tag == "G0" and t_seed != a.seed:
        raise SystemExit("--treatment-seed differs from --seed, so this is the "
                         "pre-registered CONDITIONAL 9th arm, not G0. Give it its own "
                         "--treatment-tag (e.g. G1) so it cannot overwrite G0.")
    return [Arm("F0", FLOOR, a.seed), Arm("F1", FLOOR, a.seed),
            Arm("F2", FLOOR, a.seed + 1), Arm(a.treatment_tag, TREATMENT, t_seed)]


def build_arm_argv(a, arm: Arm) -> list[str]:
    """The trainer's argv for one arm. `--report` is appended by `run_arm`.

    THERE IS NO `--depth-source` HERE AND THERE MUST NEVER BE. This branch is cut from
    main `86a9e03`, which predates Task 19; centre depth is simply what the trainer does.
    The flag belongs to `feat/plane-aux`, and an arm carrying it exits 2 before doing any
    work -- which is how the whole queue would die instantly if this were ported verbatim.
    """
    out = Path(a.out)
    argv = ["--colmap", a.colmap, "--images", a.images,
            "--max-resolution", str(a.max_resolution),
            "--steps", str(a.steps), "--budget", str(a.budget),
            "--num-downscales", str(NUM_DOWNSCALES),
            "--eval-split-every", str(EVAL_SPLIT_EVERY),
            "--eval-every", str(EVAL_EVERY),
            "--depth-loss-space", a.depth_loss_space,
            "--flatten-loss-weight", str(FLATTEN_LOSS_WEIGHT),
            "--depth-loss-weight", str(DEPTH_LOSS_WEIGHT),
            "--normal-loss-weight", str(NORMAL_LOSS_WEIGHT),
            "--depth-normal-weight", str(a.dn),
            "--export-every", str(EXPORT_EVERY),
            "--seed", str(arm.seed),
            "--export", str(out / f"{arm.tag}.ply"),
            "--eval-dump", str(out / f"{arm.tag}.dump")]
    for flag, val in (("--depth-dir", a.depth_dir), ("--normal-dir", a.normal_dir),
                      ("--init-ply", a.init_ply)):
        if val:
            argv += [flag, val]
    if a.masks:
        # EXPLICIT polarity, never `auto`. Section 13.2 records auto resolving to `drop`
        # on P-MASK -- but `resolved` records the string "auto", not what it resolved to,
        # so an explicit value is the only one that is provenance.
        argv += ["--masks", a.masks, "--mask-polarity", a.mask_polarity]
    return argv


def expected_resolved(a, arm: Arm) -> dict:
    """What the trainer's `resolved` block must say for this arm, keyed by argparse dest.

    Every flag `build_arm_argv` emits appears here; `NOT_IN_RESOLVED` is empty and a test
    derives the flag set from the argv rather than from a hand-kept list, so adding a flag
    without a matching assertion fails rather than becoming an argument sink.
    """
    out = Path(a.out)
    exp = {"colmap": a.colmap, "images": a.images,
           "max_resolution": a.max_resolution, "steps": a.steps, "budget": a.budget,
           "num_downscales": NUM_DOWNSCALES, "eval_split_every": EVAL_SPLIT_EVERY,
           "eval_every": EVAL_EVERY, "depth_loss_space": a.depth_loss_space,
           "flatten_loss_weight": FLATTEN_LOSS_WEIGHT,
           "depth_loss_weight": DEPTH_LOSS_WEIGHT,
           "normal_loss_weight": NORMAL_LOSS_WEIGHT,
           "depth_normal_weight": a.dn, "export_every": EXPORT_EVERY,
           "seed": arm.seed, "export": str(out / f"{arm.tag}.ply"),
           "eval_dump": str(out / f"{arm.tag}.dump")}
    for dest, val in (("depth_dir", a.depth_dir), ("normal_dir", a.normal_dir),
                      ("init_ply", a.init_ply)):
        if val:
            exp[dest] = val
    if a.masks:
        exp["masks"] = a.masks
        exp["mask_polarity"] = a.mask_polarity
    return exp


def check_resolved(arm: Arm, a, resolved: dict) -> None:
    """Refuse an arm that ran with settings other than the ones asked for.

    `bench/runner.py`'s reason, applied to a harness that does not go through it: both
    wrong numbers this project has published came from a harness recording its OWN
    namespace as the protocol while the trainer quietly ran something else.
    """
    exp = expected_resolved(a, arm)
    bad = {k: {"asked": v, "ran": resolved.get(k, "<absent>")}
           for k, v in exp.items() if resolved.get(k, "<absent>") != v}
    if bad:
        raise SystemExit(f"{arm.tag}: the trainer ran with settings other than the ones "
                         f"asked for: {json.dumps(bad, default=str)}")


# ============================================================ observed loss path


LOSS_PATH_KEYS = ("fused_calls", "torch_calls",
                  "dn_gated_calls", "dn_ungated_calls", "dn_skipped_calls")

#: Per-role expectations, from the pre-registration section 4 and the counter contract in
#: `39d5806`. `dn_skipped_calls` is deliberately UNCONSTRAINED: the pre-registration names
#: four conditions per role and does not name that one, and a harness must not invent a
#: fifth gate after the design was fixed. It is reported on every arm instead.
ROLE_LOSS_PATH = {
    FLOOR:     {"torch_calls": "gt0", "fused_calls": "eq0",
                "dn_gated_calls": "eq0", "dn_ungated_calls": "gt0"},
    TREATMENT: {"torch_calls": "gt0", "fused_calls": "eq0",
                "dn_gated_calls": "gt0", "dn_ungated_calls": "eq0"},
}


def check_loss_path(tag: str, role: str, report: dict) -> dict:
    """Refuse an arm whose loss chain was not the one its role requires.

    Reads what the branches COUNTED, never what the environment requested -- the two are
    different questions and only the first can catch a variable lost across a nohup.

    Every violation is reported in one message rather than the first: a fused floor
    violates two conditions at once, and naming only `torch_calls` would leave the reader
    hunting for the fused kernel that actually ran.
    """
    obs = (report or {}).get("observed")
    if not isinstance(obs, dict) or not isinstance(obs.get("loss_path"), dict):
        raise SystemExit(
            f"{tag}: the report has no `observed.loss_path` block. That arm was produced "
            f"by a binary older than 39d5806, so which loss chain ran is simply unknown "
            f"-- and unknown must not grade as torch.")
    lp = obs["loss_path"]
    missing = [k for k in LOSS_PATH_KEYS if not isinstance(lp.get(k), int)]
    if missing:
        raise SystemExit(f"{tag}: observed.loss_path is missing integer counters "
                         f"{missing}; all five keys are always present by contract.")
    if lp["fused_calls"] + lp["torch_calls"] != (lp["dn_gated_calls"]
                                                 + lp["dn_ungated_calls"]
                                                 + lp["dn_skipped_calls"]):
        raise SystemExit(f"{tag}: observed.loss_path breaks the call-total invariant "
                         f"fused+torch == gated+ungated+skipped: {lp}")
    if all(lp[k] == 0 for k in LOSS_PATH_KEYS):
        raise SystemExit(
            f"{tag}: no geometry term ran at all -- every loss-path counter is zero. An "
            f"arm whose whole purpose is the geometry recipe and whose geometry never ran "
            f"must not be gradeable as an ungated floor.")
    rules = ROLE_LOSS_PATH.get(role)
    if rules is None:
        raise SystemExit(f"{tag}: unknown arm role {role!r}")
    bad = {k: {"want": ("> 0" if how == "gt0" else "== 0"), "got": lp[k]}
           for k, how in rules.items()
           if (lp[k] <= 0 if how == "gt0" else lp[k] != 0)}
    if bad:
        raise SystemExit(
            f"{tag}: the loss chain that RAN is not the one a {role} arm requires: "
            f"{json.dumps(bad)}. Full counters {json.dumps(lp)}. The treatment variable "
            f"is an environment variable and `resolved` records none, so this is the only "
            f"artifact that could have said so.")
    return dict(lp)


# ============================================================ the seed-cloud guard


#: Names that are, or may be, the cloud a trainer seeded from. Kept ALONGSIDE the
#: resolution check below rather than replaced by it: a `points3D.txt` belonging to some
#: OTHER model is still overwhelmingly likely to be a seed, and no amount of path
#: resolution can see that.
SEED_CLOUD_NAMES = ("points3D.txt", "points3D.bin", "seed.ply")


def _resolve(p: str | os.PathLike) -> Path:
    return Path(p).expanduser().resolve()


def check_seed_cloud(seed_cloud: str, colmap: str | None, init_ply: str | None) -> None:
    """Refuse a reference cloud that is the file the trainer ACTUALLY SEEDED FROM.

    Scoring thin-axis against the cloud the trainer initialised from was an 11.6 deg error
    once -- larger than every recipe gain in CLAUDE.md's table.

    The plane-aux original refuses `points3D.txt` and `seed.ply` BY NAME. That is a name
    blacklist, and P-MASK's seed is `points3D.bin`, which sails straight through it. So
    the seed is derived from the arm's own inputs instead:

      * `--init-ply` given  -> the seed is that ply, and `dataset.py:168` IGNORES the
        COLMAP model's points3D entirely, so the model's cloud is no longer the seed and
        there is no reason left to refuse it;
      * otherwise           -> the seed is the COLMAP model's own `points3D.{bin,txt}`.

    Compared on RESOLVED paths, because a symlink, a relative path or a `..` segment all
    name the same file under a different spelling.

    THE NAME BLACKLIST IS UNCONDITIONAL AND OVERLAPS THE RESOLUTION CHECK ON PURPOSE. A
    `points3D.bin` displaced by `--init-ply` is no longer THIS run's seed and is still
    refused, because nothing in this project's protocol ever wants a COLMAP model's own
    cloud as the on-seed reference. Two checks that can fire for different reasons are
    worth more than one that fires for the union of them.
    """
    ref = _resolve(seed_cloud)
    if init_ply:
        seeds = [_resolve(init_ply)]
    elif colmap:
        seeds = [_resolve(Path(colmap) / n) for n in ("points3D.bin", "points3D.txt")]
    else:
        seeds = []
    if ref in seeds:
        raise SystemExit(
            f"--seed-cloud {seed_cloud} resolves to {ref}, which is the cloud the trainer "
            f"initialised from. Scoring on-seed and thin-axis against the training seed "
            f"was an 11.6 deg error once. Use the TSDF-only cloud.")
    if ref.name in SEED_CLOUD_NAMES:
        raise SystemExit(
            f"--seed-cloud {seed_cloud} is named {ref.name!r}, which is (or may be) a "
            f"cloud some trainer initialised from. Use the TSDF-only cloud.")


# ============================================================ preflight


def _free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def check_disk(out: Path, a, headroom: float = 1.2) -> dict:
    """Refuse a queue that cannot hold its own checkpoints.

    `--export-every 500` is mandated by the pre-registration because Reading B needs the
    checkpoints, and nothing in the design says what they cost: 30k/500 = 60 plys per arm,
    4 arms, 59 float32 fields per splat. At a 500k budget that is ~7 GB per arm and ~28 GB
    per scene. An UPPER bound -- with `--grow` the early checkpoints are smaller -- so a
    pass here is a real pass and a refusal may be conservative.
    """
    per_ckpt = a.budget * PLY_BYTES_PER_SPLAT
    n_ckpt = a.steps // EXPORT_EVERY
    n_arms = len(arm_queue(a))
    need = int(n_arms * (n_ckpt + 1) * per_ckpt * headroom)
    free = _free_bytes(out)
    info = {"checkpoints_per_arm": n_ckpt, "arms": n_arms,
            "bytes_per_checkpoint": per_ckpt, "need_bytes": need, "free_bytes": free}
    if free < need:
        raise SystemExit(
            f"not enough disk at {out}: {free / 2**30:.1f} GiB free, "
            f"{need / 2**30:.1f} GiB needed for {n_arms} arms x {n_ckpt} checkpoints "
            f"(--export-every {EXPORT_EVERY} at a {a.budget:,} budget). Pass "
            f"--allow-low-disk if you have arranged to move them as they land.")
    return info


# ============================================================ running one arm


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def run_arm(tag: str, out_dir: Path, argv: list[str], overlay: dict,
            watchdog_s: float = 14_400.0, empty_log_s: float = 90.0) -> Path:
    """Run one arm to completion, and refuse it if anything about that is not true.

    FOUR THINGS ARE ASSERTED, NOT ONE. The plane-aux original checks only that the report
    exists; Task 21's harness checked only the exit status, and read `date`'s. No one of
    these implies another:

      * the report exists at the path we asked for;
      * the export ply exists -- which is precisely how Task 21's `rc=0`-on-`Abort trap: 6`
        was eventually noticed, downstream and late;
      * the process exited 0 -- an `Abort trap: 6` after a report is written looks exactly
        like a finished arm to a file-existence check;
      * on a RESUME, `<tag>.armstatus.json` records what that status WAS. Without it the
        skip is a file-existence test again, and the crashed arm's report is adopted.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / f"{tag}.json"
    log = out_dir / f"{tag}.log"
    status = out_dir / f"{tag}.armstatus.json"

    if report.exists():
        if not status.exists():
            raise SystemExit(
                f"{tag}: {report.name} exists but {status.name} does not, so nothing "
                f"records how that run ENDED. A report is written before the process "
                f"exits, so its presence is not evidence the arm finished. Delete both "
                f"and re-run, or write the armstatus by hand if you know the status.")
        st = json.loads(status.read_text())
        if st.get("rc") != 0:
            raise SystemExit(f"{tag}: a previous attempt exited rc={st.get('rc')} "
                             f"({status.name}). Refusing to resume it as a finished arm.")
        print(f"  {tag}: report exists and armstatus records rc=0, skipping", flush=True)
        return report

    clear_stale_lock()
    lock = extension_lock_path()
    cmd = [*CAFFEINATE, *TRAINER_CMD, *argv, "--report", str(report)]
    env = dict(os.environ)
    # PYTHONUNBUFFERED so a killed process does not lose its buffered stdout;
    # PYTHONDONTWRITEBYTECODE set HERE rather than left to the caller, because a stale
    # .pyc makes a battery fail toward FALSE SURVIVED.
    env.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    env.update(overlay)

    started, t0 = _now(), time.perf_counter()
    killed = None
    with log.open("wb") as fh:
        p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
                             env=env, start_new_session=True)
        while p.poll() is None:
            time.sleep(POLL_INTERVAL_S)
            el = time.perf_counter() - t0
            # LIVENESS, not just correctness: asserting the report exists at the END is
            # structurally blind to a run that never reaches an end.
            if el > empty_log_s and log.stat().st_size == 0:
                p.kill()
                killed = (f"{tag}: 0-byte log after {el:.0f}s -- almost certainly the "
                          f"FileBaton lock. Check `lsof {lock}`.")
                break
            if el > watchdog_s:
                p.kill()
                killed = f"{tag}: watchdog at {el:.0f}s"
                break
        rc = p.wait()
    fix_openmp()

    status.write_text(json.dumps(
        {"schema": 1, "tag": tag, "rc": rc, "killed": killed,
         "started_at": started, "finished_at": _now(),
         "elapsed_s": round(time.perf_counter() - t0, 1),
         "cmd": cmd, "env_overlay": overlay, "extension_lock": str(lock),
         "report_exists": report.exists(),
         "harness_sha256": _sha256(Path(__file__).resolve()),
         "harness_file": str(Path(__file__).resolve())}, indent=2))
    if killed:
        raise SystemExit(killed)
    tail = "\n".join(log.read_text(errors="replace").strip().splitlines()[-8:])
    if not report.exists():
        raise SystemExit(f"{tag}: NO REPORT written (rc={rc})\n{tail}")
    if "--export" in argv:
        ply = Path(argv[argv.index("--export") + 1])
        if not ply.exists():
            raise SystemExit(f"{tag}: no export ply at {ply} (rc={rc}). The report is "
                             f"written before the export, so a report without a ply is "
                             f"an arm that died on the way out.\n{tail}")
    if rc != 0:
        raise SystemExit(
            f"{tag}: the report exists but the process exited rc={rc}. A report written "
            f"before an abort looks exactly like a finished arm to a file-existence "
            f"check, which is how six aborted arms once reported success.\n{tail}")
    return report


# ============================================================ CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True,
                    help="scene name, e.g. pgeom or pmask. Recorded, and the grader's "
                         "anchor is per scene.")
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--seed-cloud", required=True,
                    help="the REFERENCE cloud the battery scores on-seed and thin-axis "
                         "against. Must not be the cloud the trainer seeded from; see "
                         "check_seed_cloud.")
    ap.add_argument("--out", required=True, help="this scene's artifact directory")
    ap.add_argument("--depth-dir")
    ap.add_argument("--normal-dir")
    ap.add_argument("--init-ply")
    ap.add_argument("--masks", default=None)
    ap.add_argument("--mask-polarity", choices=["drop", "keep"], default="drop",
                    help="EXPLICIT by design. `auto` is not offered: `resolved` would "
                         "record the string 'auto' rather than what it resolved to, and "
                         "then no artifact says which polarity applied.")
    ap.add_argument("--max-resolution", type=int, default=1920,
                    help="1920 for P-GEOM, 2048 for P-MASK (pre-registration section 1)")
    ap.add_argument("--budget", type=int, default=500_000)
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dn", type=float, default=0.05,
                    help="--depth-normal-weight. 0.05 is R1, the arm the escalation "
                         "grades; the gate only exists inside this term.")
    ap.add_argument("--depth-loss-space", default="disparity",
                    choices=["disparity", "metric"])
    ap.add_argument("--watchdog", type=float, default=14_400.0,
                    help="per-arm, in seconds. macOS has no timeout(1).")
    ap.add_argument("--treatment-tag", default="G0",
                    help="G0 is the pre-registered treatment. The CONDITIONAL 9th arm "
                         "(seed 43, run only if Band 1 fires or Band 2 passes) needs its "
                         "own tag so it cannot overwrite G0.")
    ap.add_argument("--treatment-seed", type=int, default=None,
                    help="defaults to --seed. A different value requires a different "
                         "--treatment-tag.")
    ap.add_argument("--floors-only", action="store_true",
                    help="run F0/F1/F2 and stop before the treatment.")
    ap.add_argument("--treatment-only", action="store_true",
                    help="run only the treatment arm. The floor arms must already be "
                         "present and rc=0.")
    ap.add_argument("--allow-low-disk", action="store_true",
                    help="skip the checkpoint-space preflight. 60 checkpoints per arm at "
                         "a 500k budget is ~28 GB per scene.")
    return ap


def main(argv=None) -> None:
    a = build_parser().parse_args(argv)
    if a.floors_only and a.treatment_only:
        raise SystemExit("--floors-only and --treatment-only are mutually exclusive")

    # Checked BEFORE the GPU is touched: a reference cloud that is the training seed
    # invalidates the whole battery, and finding that out after ten GPU-hours is the
    # expensive way to learn it.
    check_seed_cloud(a.seed_cloud, a.colmap, a.init_ply)

    from bench.runner import require_gpu_exclusive
    require_gpu_exclusive()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    disk = None if a.allow_low_disk else check_disk(out, a)

    queue = arm_queue(a)
    if a.floors_only:
        queue = [x for x in queue if x.role == FLOOR]
    elif a.treatment_only:
        queue = [x for x in queue if x.role == TREATMENT]

    (out / "harness.json").write_text(json.dumps(
        {"schema": 1, "what": "the RUNNER's own provenance. Grading is separate and is a "
                              "pure function of the artifacts.",
         "started_at": _now(), "harness_file": str(Path(__file__).resolve()),
         "harness_sha256": _sha256(Path(__file__).resolve()),
         "mg_root": str(ROOT), "scene": a.scene, "args": vars(a),
         "queue": [x._asdict() for x in queue], "disk_preflight": disk,
         "extension_lock": str(extension_lock_path())}, indent=2, default=str))

    observed = {}
    for arm in queue:
        print(f"=== {arm.tag} ({arm.role}, seed {arm.seed}) ===", flush=True)
        rep_path = run_arm(arm.tag, out, build_arm_argv(a, arm), env_overlay(arm.role),
                           a.watchdog)
        rep = json.loads(rep_path.read_text())
        # Both, immediately, so a confounded arm halts the queue rather than sitting at
        # the front of five more GPU-hours.
        lp = check_loss_path(arm.tag, arm.role, rep)
        check_resolved(arm, a, rep.get("resolved") or {})
        observed[arm.tag] = {"role": arm.role, "seed": arm.seed, "loss_path": lp}
        print(f"  {arm.tag}: OK  loss_path {json.dumps(lp)}", flush=True)

    (out / "ARMS_DONE.json").write_text(json.dumps(
        {"schema": 1, "finished_at": _now(), "scene": a.scene,
         "arms": observed,
         "note": "the arms ran and each one's observed loss path matches its role. "
                 "NOTHING here is graded; the battery and the three bands are separate."},
        indent=2))
    print(f"\nall {len(queue)} arm(s) ran and passed their role assertions", flush=True)


if __name__ == "__main__":
    main()
