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
import math
import os
import re
import shutil
import statistics
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


# ============================================================ scoring and the battery

#: `analyze/splatstats`, the tool that computes on-seed and thin-axis. Overridable,
#: because this repo is a submodule and the sibling checkout is not in the same place on
#: every machine -- and a scoring path that is wrong is only discovered AFTER the arms.
SPLATSTATS = Path(os.environ.get("MG_SPLATSTATS")
                  or ROOT.parent.parent / "analyze" / "splatstats")

ANCHOR_PATH = ROOT / "bench" / "results" / "dn_gate" / "tier3_anchor.json"

PRIMARY_TAG = "G0"


def score(out: Path, tag: str, seed_cloud: str) -> None:
    """splatstats + LPIPS + backfill. Idempotent; each step skips if its artifact exists.

    Every step asserts its ARTIFACT rather than its exit status, and then the battery
    asserts the artifact's content: a scorer that printed a number and wrote nothing is a
    state this project has been in.
    """
    if not (SPLATSTATS / "scripts" / "splat_stats.py").exists():
        raise SystemExit(f"no splatstats at {SPLATSTATS}. Set MG_SPLATSTATS. Scoring "
                         f"cannot proceed and the arms are already paid for.")
    stats = out / f"{tag}.stats.json"
    if not stats.exists():
        r = subprocess.run(
            ["caffeinate", "-i", "uv", "run", "--frozen", "python",
             "scripts/splat_stats.py", str(out / f"{tag}.ply"), "--seed", seed_cloud,
             "--json", str(stats), "--quiet"],
            cwd=str(SPLATSTATS), capture_output=True, text=True)
        if not stats.exists():
            raise SystemExit(f"{tag}: splatstats wrote no JSON\n"
                             f"{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    lp = out / f"{tag}.dump" / "lpips.json"
    if not lp.exists():
        r = subprocess.run(["caffeinate", "-i", "uv", "run", "scripts/lpips_eval.py",
                            str(out / f"{tag}.dump")], cwd=str(ROOT),
                           capture_output=True, text=True)
        if not lp.exists():
            raise SystemExit(f"{tag}: lpips_eval wrote no JSON\n"
                             f"{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    fix_openmp()
    subprocess.run([str(ROOT / ".venv/bin/python"), "scripts/backfill_lpips.py",
                    str(out), tag], cwd=str(ROOT), capture_output=True, text=True)


def peak_driver_gb(log: Path):
    """A re-grade runs against the COMMITTED artifacts, which are the reports only -- the
    multi-megabyte training logs are not in git. An absent log is `not measured here`,
    never zero: the column then falls out of the battery on both sides and is not graded.
    """
    if not log.exists():
        return None
    vals = [float(m) for m in
            re.findall(r"\[mem\] driver ([0-9.]+) GB", log.read_text(errors="replace"))]
    return max(vals) if vals else None


def report_path(out: Path, tag: str) -> Path:
    """`<tag>.json` as a run writes it, or `<tag>.report.json` as a repo commits it.

    Accepting both is what lets `--regrade` reproduce a verdict from the repository alone,
    with no scratch directory and no GPU -- the only form of reproducibility that survives
    the scratch directory being cleaned.
    """
    for name in (f"{tag}.json", f"{tag}.report.json"):
        if (out / name).exists():
            return out / name
    raise SystemExit(f"{tag}: no report at {out}/{tag}.json or {out}/{tag}.report.json")


ARM_ROLES = {"F0": FLOOR, "F1": FLOOR, "F2": FLOOR}


def role_of(tag: str) -> str:
    """Floors are named; anything else is a treatment. So the conditional 9th arm (G1) is
    role-checked as a treatment without anyone having to remember to say so."""
    return ARM_ROLES.get(tag, TREATMENT)


def battery(out: Path, tag: str) -> dict:
    """Every column the rule grades, from the artifact that produced it -- never stdout.

    THE ROLE ASSERTION IS MADE AGAIN HERE, and that is not redundancy. `--regrade` is
    meant to reproduce a verdict from committed artifacts alone, so it cannot assume the
    arms came from this harness's runner; and an arm whose loss chain is unknown must not
    become gradeable by being re-read.
    """
    rep = json.loads(report_path(out, tag).read_text())
    check_loss_path(tag, role_of(tag), rep)
    resolved = rep.get("resolved") or {}
    st = json.loads((out / f"{tag}.stats.json").read_text())
    ref = str(st.get("seed_cloud") or "")
    # Derived from THIS ARM's own inputs, not from a command line a re-grade never saw.
    check_seed_cloud(ref, resolved.get("colmap"), resolved.get("init_ply"))
    m = rep["metrics"]
    sh = m.get("shape") or {}
    vals = {f"stats.{k}": v for k, v in st["metrics"].items()
            if isinstance(v, (int, float))}
    vals.update({
        "run.psnr_masked": m.get("psnr_masked"), "run.psnr": m.get("psnr"),
        "run.coverage": m.get("coverage"), "run.lpips": m.get("lpips"),
        "run.ms_per_step": m.get("ms_per_step"), "run.n_splats": m.get("n_splats"),
        "run.aspect_p50": sh.get("aspect_p50"), "run.needle_frac": sh.get("needle_frac"),
        # REPORTED, never gated. `Dlog aspect = Dlog smid - Dlog smax`, so aspect already
        # IS that differential and a collapse test on the halves would double-count it.
        "run.hard_needle_frac": sh.get("hard_needle_frac"),
        "run.smid_p50_mm": sh.get("smid_p50_mm"),
        "run.smax_p50_mm": sh.get("smax_p50_mm"),
        "run.peak_driver_gb": peak_driver_gb(out / f"{tag}.log")})
    return {"tag": tag, "role": role_of(tag), "seed": resolved.get("seed"),
            "git": (rep.get("env") or {}).get("git"), "seed_cloud": ref,
            "resolved": resolved,
            "loss_path": rep["observed"]["loss_path"],
            "thin_axis_evaluated": st["metrics"].get("thin_axis_evaluated"),
            "values": {k: v for k, v in vals.items() if isinstance(v, (int, float))}}


# ============================================================ floors

FLOOR_TAGS = ("F0", "F1", "F2")
FLOOR_CONFIG_KEYS = ("depth_normal_weight", "depth_loss_space", "flatten_loss_weight",
                     "depth_loss_weight", "normal_loss_weight", "budget", "steps",
                     "max_resolution", "num_downscales", "masks", "mask_polarity")


def _floor_table(arms: dict) -> dict:
    keys = sorted(set.intersection(*(set(arms[t]["values"]) for t in FLOOR_TAGS)))
    fl = {}
    for k in keys:
        v = [arms[t]["values"][k] for t in FLOOR_TAGS]
        fl[k] = {"F0": v[0], "F1": v[1], "F2": v[2],
                 "mean": statistics.mean(v), "spread_n3": max(v) - min(v),
                 "repeat_pair_abs_diff": abs(v[0] - v[1])}
    return fl


def floors_from_reports(out: Path) -> dict:
    return _floor_table({t: battery(out, t) for t in FLOOR_TAGS})


def merge_extended_floors(committed: dict, rebuilt: dict) -> dict:
    """Committed floors, plus columns only the rebuild has -- and a hard refusal if the
    two DISAGREE anywhere they overlap.

    The overlap check is the point, not the merge. floors.json was written under the phase
    order that makes the protocol trustworthy; recomputing it from the same reports must
    reproduce it exactly, and if it does not, the artifacts moved and nothing graded
    against them means anything. Adding a column must never re-measure a floor.
    """
    for k, v in committed.items():
        if k not in rebuilt:
            continue
        for field in ("F0", "F1", "F2", "mean", "spread_n3"):
            a, b = v.get(field), rebuilt[k].get(field)
            if a is None or b is None or abs(a - b) > 1e-12 * max(1.0, abs(a)):
                raise SystemExit(
                    f"floors.json and the floor arms' own reports disagree on "
                    f"{k}.{field}: {a!r} vs {b!r}. The committed floors are the frozen "
                    f"record; a rebuild that does not reproduce them means the artifacts "
                    f"moved underneath, and no verdict computed from them is meaningful.")
    return {**{k: v for k, v in rebuilt.items() if k not in committed}, **committed}


def check_floors_match(configs: list, dn: float, space: str) -> None:
    """Refuse floors not measured on THIS arm's own configuration.

    Two things are checked, and the first is the one a summary field cannot do:
      1. the floor arms AGREE WITH EACH OTHER on every configuration key. Three runs that
         differ in a flag are not a repeat measurement of anything, and their spread is
         not a noise floor -- it is a treatment effect wearing one;
      2. they match the requested base.
    A missing key is a mismatch, never agreement.
    """
    if not configs:
        raise SystemExit("no floor arm reports to check")
    ref = configs[0]
    for i, c in enumerate(configs[1:], start=1):
        diff = {k: (ref.get(k, "<absent>"), c.get(k, "<absent>"))
                for k in FLOOR_CONFIG_KEYS
                if ref.get(k, "<absent>") != c.get(k, "<absent>")}
        if diff:
            raise SystemExit(
                f"floor arms 0 and {i} differ in configuration: {diff}. Three runs that "
                f"differ in a flag are not a repeat measurement, and their spread is a "
                f"treatment effect, not a noise floor.")
    for k, want in (("depth_normal_weight", dn), ("depth_loss_space", space)):
        have = ref.get(k, "<absent>")
        if have != want:
            raise SystemExit(f"floors were measured with {k}={have}; this arm's base is "
                             f"{k}={want}. A floor for another configuration is not this "
                             f"arm's floor.")


def write_floors(out: Path, scene: str, dn: float, space: str,
                 self_anchor: bool = False) -> dict:
    """PHASE 2, and its POSITION is the guarantee: floors are scored and WRITTEN before
    any treatment number exists. Enforced here rather than by operator discipline.
    """
    arms = {t: battery(out, t) for t in FLOOR_TAGS}
    refs = {arms[t]["seed_cloud"] for t in arms}
    if len(refs) > 1:
        raise SystemExit(f"the floor arms disagree about the reference cloud: {refs}. "
                         f"on-seed is only comparable with the reference held fixed, so "
                         f"their spread would not be a spread of anything.")
    if arms["F0"]["seed"] != arms["F1"]["seed"]:
        raise SystemExit("F0 and F1 must share a seed: they are the repeat pair")
    if arms["F2"]["seed"] == arms["F0"]["seed"]:
        raise SystemExit("F2 must use a different seed: it is the seed floor")
    check_floors_match([arms[t]["resolved"] for t in FLOOR_TAGS], dn, space)
    fl = _floor_table(arms)
    doc = {"schema": 1, "scene": scene, "dn": dn, "depth_loss_space": space,
           "note": "floor = spread_n3 = max-min over the three base arms. repeat_pair is "
                   "|F0-F1| and is REPORTED ONLY -- an n=2 floor was 25-45x too small "
                   "once (research/metal-gauss.md section 8.2) and is never graded "
                   "against.",
           "written_at": _now(),
           "arms": {t: {k: v for k, v in arms[t].items() if k != "values"} for t in arms},
           "floors": fl}
    (out / "floors.json").write_text(json.dumps(doc, indent=2, default=str))
    if self_anchor:
        write_self_anchor(out, scene, fl, arms["F0"]["resolved"])
    (out / "FLOORS_DONE").write_text(_now())
    return doc


def write_self_anchor(out: Path, scene: str, fl: dict, resolved: dict) -> Path:
    """A scene with no frozen Tier 3 anchor gets one from its OWN floor means, written at
    the same moment floors.json is and BEFORE the treatment is scored.

    NEVER OVERWRITTEN once written, and that is the whole point: on the second Tier 3 arm
    the floors are re-measured while the anchor stays put, which is when the cumulative
    check stops being vacuous and starts being the ratchet detector it exists to be.
    """
    p = out / "anchor.json"
    if p.exists():
        print(f"  self-anchor already frozen at {p}, keeping it", flush=True)
        return p
    missing = [k for k in COLLAPSE if k not in fl]
    if missing:
        raise SystemExit(f"cannot self-anchor {scene}: the floors have no {missing}. An "
                         f"anchor that predates a column cannot testify about it.")
    p.write_text(json.dumps({
        "schema": 1, "self_anchored": True, "scene": scene, "written_at": _now(),
        "source": {"arms": list(FLOOR_TAGS), "from": "floors.json",
                   "statistic": "mean of the three floor arms",
                   "why": "no Tier 3 arm has ever run on this scene, so there is nothing "
                          "frozen to anchor on (pre-registration e94ee45 section 5)."},
        "vacuity": "For the FIRST Tier 3 arm on this scene the cumulative half of Band 1 "
                   "is VACUOUS BY CONSTRUCTION: the anchor IS the floor mean, so the "
                   "cumulative delta equals the per-arm delta exactly. It becomes a real "
                   "check for the SECOND arm and not before. Stated here rather than "
                   "discovered later.",
        "config": {k: resolved.get(k) for k in ANCHOR_CONFIG_KEYS},
        "values": {k: fl[k]["mean"] for k in COLLAPSE}}, indent=2))
    return p


# ============================================================ the three bands
#
# THE TIER 3 KEEP/DROP RULE, `3cfd8f3`, which replaced "WORSENED anywhere = DROP".
#
# The old rule was magnitude-blind: it returned the same one-word verdict for Task 19's
# 4.5x-floor drift (needles +0.6 pp, aspect -2.5%, on-seed UP 36% relative) and for the
# Tier 2 VOID row's collapse (needles +40 pp, aspect -78%, on-seed HALVED) -- ~35x apart.
#
#   Band 1  COLLAPSE     hard DROP, any one column, per-arm AND cumulative
#   Band 2  GEOMETRY     on-seed@1cm must RISE; thin-axis must not worsen
#   Band 3  PHOTOMETRIC  hard DROP on a >0.25 dB PSNR loss, or crossing the 24 dB gate
#
# Every Band 1 threshold is `sqrt(healthy x collapse)` in the column's natural space, with
# the adopted arm chosen PER COLUMN (aspect on R1, the rest on R1p -- a uniform anchor is
# UNDEFINED for on-seed, which R1 improved). research/metal-gauss.md 13.6 re-derives all
# four from the plys. They are conventions with a stated derivation, not measurements.

DIRECTION = {
    "stats.on_seed_frac_1cm": +1,
    "stats.on_seed_frac_2cm": +1,
    "stats.thin_axis_angle_p50": -1,
    "run.aspect_p50": +1,
    "run.needle_frac": -1,
    "run.lpips": -1,
    "run.psnr_masked": 0,
}
GEOMETRY_GATE = ("stats.on_seed_frac_1cm", "stats.thin_axis_angle_p50",
                 "run.aspect_p50", "run.needle_frac")

COLLAPSE = {
    "run.needle_frac":        {"space": "abs", "worse": +1, "threshold": 0.108},
    "run.aspect_p50":         {"space": "log", "worse": -1, "threshold": 0.346},
    "stats.on_seed_frac_1cm": {"space": "log", "worse": -1, "threshold": 0.185},
    "run.lpips":              {"space": "abs", "worse": +1, "threshold": 0.017},
}
BAND2_GATE = ("stats.on_seed_frac_1cm", "stats.thin_axis_angle_p50")

PSNR_DROP_DB = 0.25
STAGE4_PSNR_DB = 24.0

DRIFT_SCOPE = tuple(dict.fromkeys(tuple(COLLAPSE) + BAND2_GATE + ("run.psnr_masked",)))

ANCHOR_CONFIG_KEYS = ("budget", "steps", "max_resolution", "num_downscales")


def transfers_between_scenes(spec: dict) -> bool:
    """A LOG threshold is a ratio and means the same relative change on any baseline; an
    ABSOLUTE one does not. DERIVED from the space rather than declared, so the two cannot
    drift apart -- and so the grade can say which of the four transfer.

    This is the machine-readable form of pre-registration section 5: `needle_frac +10.8
    pp` is 71% relative on P-GEOM's 0.1516 baseline and ~46% on P-MASK's 0.2348, so
    quoting it as a constant across scenes is quoting two different rules.
    """
    return spec["space"] == "log"


def threshold_relative(spec: dict, reference: float):
    """The threshold as a FRACTION OF THE BASELINE, in one comparable form for both
    spaces, so a reader never has to convert between pp and dlog to see how big it is."""
    if spec["space"] == "abs":
        return (spec["threshold"] / reference) if reference else None
    t = spec["threshold"]
    return (1.0 - math.exp(-t)) if spec["worse"] < 0 else (math.exp(t) - 1.0)


def collapse_delta(metric: str, value: float, reference: float) -> float:
    """How far `value` sits from `reference` TOWARD WORSE, in the column's natural space.

    POSITIVE = WORSE, always, whichever direction the column runs. A sign error inverts
    every Band 1 test -- an arm that HALVED on-seed would read as a large improvement and
    no collapse could ever fire -- so the sign is a test of its own.
    """
    spec = COLLAPSE[metric]
    if spec["space"] == "log":
        if value <= 0.0 or reference <= 0.0:
            raise SystemExit(f"{metric}: log-space column needs positive values, got "
                             f"value={value!r} reference={reference!r}")
        d = math.log(value) - math.log(reference)
    else:
        d = value - reference
    return spec["worse"] * d


def _collapse_side(values: dict, reference: dict, what: str) -> dict:
    row = {}
    for col, spec in COLLAPSE.items():
        if col not in values:
            raise SystemExit(f"Band 1 column {col} is missing from the treatment "
                             f"battery. A collapse column that was never measured must "
                             f"never read as 'did not collapse'.")
        if col not in reference:
            raise SystemExit(f"Band 1 column {col} is missing from the {what}. An anchor "
                             f"or baseline that predates a column cannot testify about "
                             f"that column.")
        d = collapse_delta(col, values[col], reference[col])
        thr = spec["threshold"]
        row[col] = {"value": values[col], "reference": reference[col], "delta": d,
                    "threshold": thr, "x_threshold": d / thr, "space": spec["space"],
                    # Pre-registration section 5: state the scene's own baseline beside
                    # the threshold rather than applying a constant from another scene.
                    "scene_baseline": reference[col],
                    "threshold_relative_to_baseline": threshold_relative(spec,
                                                                         reference[col]),
                    "threshold_transfers_between_scenes": transfers_between_scenes(spec),
                    "fired": d > thr}
    return row


def band1(t_values: dict, base_values: dict, anchor_values: dict,
          self_anchored: bool) -> dict:
    """Band 1 -- COLLAPSE. Hard DROP; any ONE column; per-arm AND cumulative.

    Per-arm is against this arm's own re-measured base. Cumulative is against the scene's
    FROZEN anchor, and it is the half that stops the rule ratcheting: four accepted 8 pp
    needle drifts are a 32 pp collapse that no single arm ever fired on.

    Comparison is STRICT: a delta exactly equal to the threshold has not fired.

    VACUITY IS MEASURED, NOT READ OFF A FLAG. On a self-anchored scene's FIRST arm the
    anchor IS the floor mean and the cumulative delta equals the per-arm delta exactly, so
    the check decides nothing -- but on the second arm the floors have moved and it starts
    deciding. A harness that reported vacuity from `self_anchored` would go on saying so
    forever, exactly when the check begins to matter.
    """
    per = _collapse_side(t_values, base_values, "baseline")
    cum = _collapse_side(t_values, anchor_values, "anchor")
    vacuous = all(abs(anchor_values[c] - base_values[c])
                  <= 1e-12 * max(1.0, abs(base_values[c])) for c in COLLAPSE)
    note = ("VACUOUS BY CONSTRUCTION on this arm: the anchor IS this arm's own floor mean, "
            "so the cumulative delta equals the per-arm delta exactly and the cumulative "
            "half decides nothing. This is NOT a cumulative check that was made and "
            "passed. It becomes a real check for the SECOND Tier 3 arm on this scene."
            if vacuous else
            "The anchor differs from this arm's floor mean, so the cumulative half is a "
            "real check.")
    decomposition = {
        c: {"everything_but_the_gate": collapse_delta(c, base_values[c],
                                                      anchor_values[c]),
            "the_gate": collapse_delta(c, t_values[c], base_values[c])}
        for c in COLLAPSE}
    pf = [k for k, v in per.items() if v["fired"]]
    cf = [k for k, v in cum.items() if v["fired"]]
    return {"per_arm": per, "cumulative": cum, "per_arm_fired": pf,
            "cumulative_fired": cf, "fired": bool(pf or cf),
            "anchor_is_self_anchored": self_anchored,
            "cumulative_check_vacuous": vacuous, "cumulative_note": note,
            # Pre-registration section 5: the frozen P-GEOM anchor differs from these arms
            # in four ways that are NOT the gate (dn, loss chain, --export-every, and the
            # MACHINE), so a cumulative firing must never be read as a treatment effect.
            "decomposition": decomposition,
            "decomposition_note":
                "(F-mean - anchor) is everything-but-the-gate; (G0 - F-mean) is the gate. "
                "They sum to the cumulative delta in the column's own space. If the "
                "cumulative check fires while the per-arm one does not, what is in "
                "question is the anchor's applicability, not the treatment."}


def band2(verdicts: dict) -> str:
    """Band 2 -- GEOMETRY GATE.

        PASS          on-seed@1cm IMPROVED beyond floor, thin-axis p50 not WORSENED
        FAIL          either column WORSENED beyond floor
        WITHIN FLOOR  neither worsened, but on-seed did not rise either

    Aspect and needles are deliberately NOT read here. Moving them to Band 1, where a 2.5%
    move and a 78% collapse get different answers, IS the amendment.
    """
    missing = [k for k in BAND2_GATE if verdicts.get(k) is None]
    if missing:
        raise SystemExit(f"Band 2 columns missing from the battery: {missing}. An absent "
                         f"gate column must never read as a pass.")
    on_seed, thin = (verdicts[k] for k in BAND2_GATE)
    if on_seed == "WORSENED" or thin == "WORSENED":
        return "FAIL"
    if on_seed == "IMPROVED":
        return "PASS"
    return "WITHIN FLOOR"


def band3(psnr_treatment: float, psnr_baseline: float) -> dict:
    """Band 3 -- PHOTOMETRIC. Hard DROP on a PSNR LOSS greater than 0.25 dB, or on falling
    below the 24 dB Stage 4 gate from at or above it.

    ONE-SIDED by construction: the rule says "falls by", and the old two-sided "must be
    WITHIN floor" is what made every Tier 3 arm unable to PASS whatever its geometry did.
    A gain is not a regression. Both comparisons are strict.
    """
    loss = psnr_baseline - psnr_treatment
    crossed = psnr_baseline >= STAGE4_PSNR_DB > psnr_treatment
    return {"baseline": psnr_baseline, "treatment": psnr_treatment, "loss_db": loss,
            "allowance_db": PSNR_DROP_DB, "exceeds_allowance": loss > PSNR_DROP_DB,
            "crossed_stage4_gate": crossed,
            "baseline_above_stage4": psnr_baseline >= STAGE4_PSNR_DB,
            "fired": bool(loss > PSNR_DROP_DB or crossed)}


def drift_columns(rows: dict, verdicts: dict, band1_detail: dict,
                  band2_verdict=None, band3_fired: bool = False) -> list:
    """Beyond floor, below Band 1, and WORSE. Reported with sign and x floor; never a DROP.

    Two exclusions carry the definition:
      * IMPROVEMENTS are not drift. Band 2 REQUIRES on-seed to improve beyond its floor,
        so counting any beyond-floor move would make KEEP AS DEFAULT unreachable by
        construction -- and a rule with an unreachable branch is a broken rule.
      * A column that FIRED Band 1 is a COLLAPSE, not a drift. Reporting it as drift would
        make a hard DROP read as adoptable-with-caveats.
    """
    fired = set(band1_detail["per_arm_fired"]) | set(band1_detail["cumulative_fired"])
    out = []
    for k in DRIFT_SCOPE:
        if k in fired or k not in rows or k not in verdicts:
            continue
        d = rows[k]["delta"]
        if DIRECTION.get(k, 0) == 0:
            worse = verdicts[k] == "MOVED" and d < 0      # two-sided: only a FALL is bad
        else:
            worse = verdicts[k] == "WORSENED"
        if not worse:
            continue
        fl = rows[k]["floor_spread_n3"]
        # A Band 2 column that WORSENED is why the scene failed, not a harmless drift; it
        # satisfies the literal definition, so it is reported rather than hidden -- but
        # flagged, because a list whose entries mean "adoptable with caveats" and "this is
        # the DROP" at once is precisely the check-shape CLAUDE.md warns about.
        caused_fail = bool(band2_verdict == "FAIL" and k in BAND2_GATE
                           and verdicts.get(k) == "WORSENED")
        caused_b3 = bool(band3_fired and k == "run.psnr_masked")
        out.append({"metric": k, "delta": d, "floor_spread_n3": fl,
                    "x_floor": abs(d) / fl if fl else None, "sign": "worse",
                    "caused_band2_fail": caused_fail, "caused_band3_fire": caused_b3})
    return out


def verdict_for(metric: str, delta: float, floor: float) -> str:
    """IMPROVED / WORSENED / WITHIN FLOOR for one metric. `abs(delta) > floor` is STRICT:
    a delta exactly equal to the floor has not cleared it."""
    sign = DIRECTION[metric]
    if abs(delta) <= floor:
        return "WITHIN FLOOR"
    if sign == 0:
        return "MOVED"
    return "IMPROVED" if sign * delta > 0 else "WORSENED"


def load_anchor(path, scene: str, out: Path) -> dict:
    """The scene's frozen anchor, from the repo file or from the scene's own self-anchor.

    A missing anchor is an ERROR and never an empty one: an empty anchor makes every
    cumulative check vacuous while still writing a well-formed grade, which is exactly the
    failure shape the other guards here exist to stop.
    """
    doc = json.loads(Path(path).read_text())
    entry = (doc.get("scenes") or {}).get(scene)
    if entry:
        return dict(entry) | {"self_anchored": bool(entry.get("self_anchored"))}
    p = out / "anchor.json"
    if p.exists():
        e = json.loads(p.read_text())
        if e.get("self_anchored") is not True:
            raise SystemExit(f"{p} does not declare self_anchored: true. An anchor whose "
                             f"provenance is unstated cannot carry a cumulative check.")
        if e.get("scene") != scene:
            raise SystemExit(f"{p} anchors scene {e.get('scene')!r}, not {scene!r}.")
        return e
    raise SystemExit(
        f"no frozen Tier 3 anchor for scene {scene!r} in {path}, and no self-anchor at "
        f"{p}. The cumulative half of Band 1 cannot be evaluated without one, and an "
        f"absent anchor must not silently become a vacuous check. Anchors present: "
        f"{sorted((doc.get('scenes') or {}))}")


def check_anchor_applies(scene: str, anchor_entry: dict, resolved: dict) -> None:
    """The anchor is re-measured only when scene, budget or resolution changes -- so a run
    that changed one of those must not be graded against the old anchor.

    `steps` and `num_downscales` are checked too: both change what a 30k arm's geometry
    columns settle at, and neither is named in the anchor's own re-measure sentence, which
    is exactly why they are the two that would slip through.
    """
    want = anchor_entry.get("config") or {}
    for k in ANCHOR_CONFIG_KEYS:
        a, b = want.get(k, "<absent>"), resolved.get(k, "<absent>")
        if a != b:
            raise SystemExit(
                f"{scene}: the frozen anchor was measured at {k}={a!r} and this arm ran "
                f"at {k}={b!r}. An anchor for another configuration is not this scene's "
                f"anchor -- re-measure it, and say so, rather than grading a ratchet "
                f"against a fiction.")


def grade(scene: str, dn: float, t: dict, fl: dict, anchor_entry: dict) -> dict:
    """The three-band rule applied to one scene. PURE, so `--regrade` reproduces it from
    the committed artifacts and so the rule is unit-tested rather than exercised once by
    an eleven-hour run."""
    anchor_values = anchor_entry["values"]
    check_anchor_applies(scene, anchor_entry, t["resolved"])
    rows, verdict = {}, {}
    for k in sorted(set(t["values"]) & set(fl)):
        base, floor = fl[k]["mean"], fl[k]["spread_n3"]
        d = t["values"][k] - base
        row = {"treatment": t["values"][k], "baseline_mean": base, "delta": d,
               "paired_vs_F0": t["values"][k] - fl[k]["F0"],
               "floor_spread_n3": floor, "moves": abs(d) > floor}
        if k in DIRECTION:
            row["direction"] = {1: "higher is better", -1: "lower is better",
                                0: "two-sided"}[DIRECTION[k]]
            row["verdict"] = verdict[k] = verdict_for(k, d, floor)
        rows[k] = row

    gate = {k: verdict.get(k) for k in GEOMETRY_GATE}
    psnr = verdict.get("run.psnr_masked")
    missing = [k for k, v in gate.items() if v is None] + \
              ([] if psnr else ["run.psnr_masked"])
    if missing:
        raise SystemExit(f"{scene}: gate columns missing from the battery: {missing}. An "
                         f"absent gate column must never read as a pass.")
    base_vals = {k: fl[k]["mean"] for k in fl}
    b1 = band1(t["values"], base_vals, anchor_values,
               bool(anchor_entry.get("self_anchored")))
    b2 = band2(verdict)
    b3 = band3(t["values"]["run.psnr_masked"], fl["run.psnr_masked"]["mean"])
    drift = drift_columns(rows, verdict, b1, b2, b3["fired"])
    drop = bool(b1["fired"] or b2 == "FAIL" or b3["fired"])
    return {"schema": 2, "rule": "tier3-three-band-2026-09-04", "scene": scene, "dn": dn,
            "arm": t["tag"], "treatment": {k: v for k, v in t.items() if k != "values"},
            "band1": b1, "band1_fired": b1["fired"],
            "band2": b2, "band3": b3, "band3_fired": b3["fired"],
            "drift": drift, "scene_drop": drop,
            "scene_pass": (b2 == "PASS" and not drop),
            "falsifier_triggered_on_this_scene":
                (verdict.get("stats.on_seed_frac_1cm") == "WITHIN FLOOR"
                 and verdict.get("stats.thin_axis_angle_p50") == "WITHIN FLOOR"),
            "geometry_gate": gate, "psnr_verdict": psnr,
            "anchor": {"values": anchor_values, "config": anchor_entry.get("config"),
                       "source": anchor_entry.get("source"),
                       "self_anchored": bool(anchor_entry.get("self_anchored"))},
            "rows": rows}


def grade_scene(out: Path, scene: str, dn: float, tag: str, anchor_path) -> dict:
    """Load the frozen floors, rebuild them from the reports, refuse if they disagree, and
    grade. REFUSES until floors.json exists, which is how the phase order stops being a
    matter of operator discipline."""
    if not (out / "FLOORS_DONE").exists() or not (out / "floors.json").exists():
        raise SystemExit(
            f"{out}: no floors.json / FLOORS_DONE. The floors must be scored and WRITTEN "
            f"before any treatment number exists -- nobody may choose a floor after "
            f"seeing a treatment number, and that is enforced here rather than trusted.")
    committed = json.loads((out / "floors.json").read_text())["floors"]
    fl = merge_extended_floors(committed, floors_from_reports(out))
    t = battery(out, tag)
    entry = load_anchor(anchor_path, scene, out)
    return grade(scene, dn, t, fl, entry)


# ============================================================ the cross-scene verdict


def grade_filename(tag: str) -> str:
    """`grade.json` for the PRIMARY arm, `grade_<tag>.json` for every other.

    The scene verdict belongs to the PRE-REGISTERED arm. `--regrade --treatment-tag M0`
    overwriting `grade.json` is a defect the plane-aux branch shipped once: both files are
    well-formed grades of real arms, differing only in which arm they grade, and
    `--summary` reads `grade.json`.
    """
    return "grade.json" if tag == PRIMARY_TAG else f"grade_{tag}.json"


def write_grade(out: Path, tag: str, doc: dict) -> None:
    (out / f"grade_{tag}.json").write_text(json.dumps(doc, indent=2, default=str))
    if tag == PRIMARY_TAG:
        (out / "grade.json").write_text(json.dumps(doc, indent=2, default=str))


def collect_scenes(root: Path, scenes_csv: str, tag: str = PRIMARY_TAG) -> dict:
    """Load exactly the NAMED scenes' grades, and refuse anything else.

    THIS IS A NEAR-MISS MADE STRUCTURAL. `--summary` originally globbed every subdirectory
    containing a grade.json, and two SYNTHETIC smoke directories -- fixtures written to
    test the grader itself, one of them a fabricated regression -- sat in that same tree.
    Because DROP is checked first and is not overridable, an invented regression would
    have produced a DROP indistinguishable from a measured one, and nothing would have
    errored. So the caller NAMES the scenes: a named scene that is missing is an error,
    and an UNNAMED grade found in the tree is ALSO an error -- the second half is the one
    that matters, because it is the half a glob gets wrong.
    """
    want = [s for s in (x.strip() for x in scenes_csv.split(",")) if s]
    if not want:
        raise SystemExit("--summary requires --scenes (e.g. --scenes pgeom,pmask). A glob "
                         "over --out would count any stray grade.json, including the "
                         "grader's own synthetic test fixtures, as a measured scene.")
    fname = grade_filename(tag)
    found = {d.name for d in root.iterdir() if d.is_dir() and (d / fname).exists()}
    missing, extra = sorted(set(want) - found), sorted(found - set(want))
    if missing:
        raise SystemExit(f"--scenes named {missing} but there is no {fname} for them "
                         f"under {root}")
    if extra:
        raise SystemExit(f"unnamed {fname} under {root}: {extra}. Name them in --scenes "
                         f"or move them out; a stray one is not silently included.")
    per = {n: json.loads((root / n / fname).read_text()) for n in want}
    for n, g in per.items():
        if g.get("scene") != n:
            raise SystemExit(f"{n}/{fname} says scene={g.get('scene')!r}: the directory "
                             f"and the report disagree about what was measured.")
    return per


def combined_verdict(per_scene: dict) -> dict:
    """The cross-scene outcome class.

        DROP                       -- Band 1 fires anywhere, or Band 2 FAILs, or Band 3
                                      fires. CHECKED FIRST AND NOT OVERRIDABLE.
        KEEP AS DEFAULT            -- Band 2 PASSes on every scene, no drift anywhere.
        OPT-IN, DEFAULT-CANDIDATE  -- PASSes everywhere, drift present. Promotable only by
                                      a blind visual A/B, which this grader cannot do.
        OPT-IN                     -- PASSes on one scene, WITHIN FLOOR on the other.
        NOT ADOPTED                -- nothing passed and nothing regressed. Reading 2 of
                                      `3cfd8f3`: the shipped default stands. For THIS arm
                                      that is the most likely outcome, since removing
                                      pixels from a loss has no prior reason to raise
                                      on-seed@1cm.

    The drop set is RECOMPUTED here from the bands rather than read from each scene's
    `scene_drop`: a pass on one scene and a collapse on another is not an opt-in, and an
    implementation that reached an opt-in branch first would turn a collapse into a
    recommendation.
    """
    scenes = sorted(per_scene)

    def _drops(s):
        g = per_scene[s]
        return bool(g.get("band1_fired") or g.get("band2") == "FAIL"
                    or g.get("band3_fired"))

    drops = [s for s in scenes if _drops(s)]
    passes = [s for s in scenes if per_scene[s].get("band2") == "PASS" and s not in drops]
    within = [s for s in scenes if per_scene[s].get("band2") == "WITHIN FLOOR"
              and s not in drops]
    drifting = {s: per_scene[s].get("drift") or [] for s in scenes}
    any_drift = any(drifting[s] for s in scenes)
    fals = [s for s in scenes if per_scene[s].get("falsifier_triggered_on_this_scene")]

    if drops:
        decision = "DROP"
    elif len(passes) == len(scenes):
        decision = "KEEP AS DEFAULT" if not any_drift else "OPT-IN, DEFAULT-CANDIDATE"
    elif passes and len(passes) + len(within) == len(scenes):
        decision = "OPT-IN"
    else:
        decision = "NOT ADOPTED (no scene passed, none regressed)"

    return {"schema": 2, "rule": "tier3-three-band-2026-09-04", "scenes": scenes,
            "decision": decision,
            "promotion_requires":
                ("a blind visual A/B on a rendered view; this grader cannot promote a "
                 "default-candidate, because nothing it measures looks at the render"
                 if decision == "OPT-IN, DEFAULT-CANDIDATE" else None),
            "reading_B_may_not_contribute":
                "Reading B (the early-divergence probe) is DIAGNOSTIC by pre-registration "
                "and contributes nothing to this line. A B-positive with an A-null is "
                "reported as 'detectable early, immaterial at 30k' and is STILL NOT "
                "ADOPTED.",
            "passed_on": passes, "within_floor_on": within, "regressed_on": drops,
            "drift_on": {s: [d["metric"] for d in drifting[s]] for s in scenes
                         if drifting[s]},
            "dn_settings_measured": sorted({per_scene[s].get("dn") for s in scenes}),
            "falsifier_scenes": fals,
            "per_scene": {s: {k: per_scene[s].get(k) for k in
                              ("band1_fired", "band2", "band3_fired", "scene_pass",
                               "scene_drop", "geometry_gate", "psnr_verdict", "dn")}
                          | {"drift": [d["metric"] for d in drifting[s]]}
                          for s in scenes}}


# ============================================================ Reading B

#: NAMED IN THE PRE-REGISTRATION AND NOT CONFIGURABLE. 500 is the first checkpoint after
#: the 21x fall and brackets the whole transient; 2000 sits inside the flat regime, so a
#: difference at 500 can be seen to persist or not. A flag here would be an invitation to
#: reselect after seeing a number, which is the one thing writing them down in advance was
#: meant to prevent. Which steps were read is recorded in the output regardless.
GRADED_EARLY_STEPS = (500, 2000)

PROBE_ARMS = ("F0", "F1", "G0")

#: probe column -> the 30k floor column it is scaled against when the early noise is zero.
PROBE_COLUMNS = {
    "aspect_p50": "run.aspect_p50", "needle_frac": "run.needle_frac",
    "hard_needle_frac": "run.hard_needle_frac", "smid_p50_mm": "run.smid_p50_mm",
    "smax_p50_mm": "run.smax_p50_mm", "splats": "run.n_splats",
    "on_seed_frac_1cm": "stats.on_seed_frac_1cm",
    "thin_axis_angle_p50": "stats.thin_axis_angle_p50",
}


def checkpoint_columns(out: Path, tag: str, step: int, seed_cloud: str) -> dict:
    """The probe's columns for one arm at one checkpoint: the five `bench/ply_shape` shape
    columns and the splat count from the ply, plus the Band 2 pair from splatstats.

    A MISSING CHECKPOINT IS REFUSED, never skipped. `G0 - F0` with F0 absent is not a
    smaller measurement, it is a different one.
    """
    ply = out / f"{tag}.step{step:06d}.ply"
    if not ply.exists():
        raise SystemExit(f"no checkpoint at {ply}. Reading B needs steps "
                         f"{list(GRADED_EARLY_STEPS)} for every arm in {list(PROBE_ARMS)};"
                         f" they exist only if the arm ran with --export-every "
                         f"{EXPORT_EVERY}.")
    sys.path.insert(0, str(ROOT))
    from bench.ply_shape import shape_of_ply
    cols = shape_of_ply(str(ply))
    js = out / f"{tag}.step{step:06d}.stats.json"
    if not js.exists():
        r = subprocess.run(
            ["caffeinate", "-i", "uv", "run", "--frozen", "python",
             "scripts/splat_stats.py", str(ply), "--seed", seed_cloud,
             "--json", str(js), "--quiet"],
            cwd=str(SPLATSTATS), capture_output=True, text=True)
        if not js.exists():
            raise SystemExit(f"{tag}@{step}: splatstats wrote no JSON\n"
                             f"{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    sm = json.loads(js.read_text())["metrics"]
    return {"aspect_p50": cols["aspect_p50"], "needle_frac": cols["needle_frac"],
            "hard_needle_frac": cols["hard_needle_frac"],
            "smid_p50_mm": cols["smid_p50_mm"], "smax_p50_mm": cols["smax_p50_mm"],
            "splats": cols["splats"],
            "on_seed_frac_1cm": sm["on_seed_frac_1cm"],
            "thin_axis_angle_p50": sm["thin_axis_angle_p50"]}


def early_divergence(out: Path, scene: str) -> dict:
    """READING B -- the early-divergence probe. DIAGNOSTIC ONLY; it returns magnitudes.

    It exists because an A-null is AMBIGUOUS between two different worlds: (i) the gate
    does nothing, and (ii) the gate does something early and the trainer's own chaos
    erased it -- same-seed atomics divergence compounding over 30k into a 0.10-0.22 dB
    spread is a mechanism for destroying exactly this evidence. Those are not the same
    finding: (i) closes the question, (ii) says only "immaterial at 30k".

    So it compares gated against ungated where the gate's exposure is MAXIMAL and the
    chaos has had LEAST time to act, and reports per column the EFFECT (G0 - F0), the
    NOISE (F1 - F0, two arms that share a seed and a configuration) and their ratio.

    NO THRESHOLD IS SET, DELIBERATELY: at 500 steps F0 and F1 may differ by almost
    nothing, so a "beyond floor" test there would fire on differences of no consequence.

    IT CANNOT RETURN AN ADOPTION OUTCOME OF ANY KIND, and that is pre-registered rather
    than a property of this implementation -- after seeing an early difference it would be
    tempting to promote it. A test asserts no such word appears anywhere in this document.
    """
    floors = json.loads((out / "floors.json").read_text())["floors"]
    seed_cloud = json.loads((out / "F0.stats.json").read_text())["seed_cloud"]
    steps = {}
    for step in GRADED_EARLY_STEPS:
        arms = {t: checkpoint_columns(out, t, step, seed_cloud) for t in PROBE_ARMS}
        row = {}
        for col in PROBE_COLUMNS:
            f0, f1, g0 = (arms[t][col] for t in PROBE_ARMS)
            effect, noise = g0 - f0, f1 - f0
            fl = (floors.get(PROBE_COLUMNS[col]) or {}).get("spread_n3")
            entry = {"F0": f0, "F1": f1, "G0": g0,
                     "effect_G0_minus_F0": effect, "noise_F1_minus_F0": noise,
                     "effect_over_noise": (effect / noise) if noise != 0 else None,
                     "floor_30k_spread_n3": fl,
                     "effect_over_30k_floor": (effect / fl) if fl else None}
            if noise == 0:
                entry["note"] = (
                    "F1 - F0 is exactly zero at this step, so the ratio is undefined and "
                    "is NOT reported as an infinite effect. F0 and F1 share a seed, and "
                    "over this many steps the trainer's atomics divergence may not have "
                    "reached the precision at which a ply prints -- i.e. the early noise "
                    "is below the ply's own float32 resolution, which is a finding about "
                    "this probe's sensitivity rather than a missing number. The effect is "
                    "given in absolute terms and against the 30k floor.")
            row[col] = entry
        steps[str(step)] = row
    return {"schema": 1, "scene": scene, "arms": list(PROBE_ARMS),
            "steps_read": list(GRADED_EARLY_STEPS),
            "steps_read_note":
                "Named in pre-registration e94ee45 section 2 and fixed in code, not "
                "configurable. Recorded here so a reader never has to take that on trust.",
            "reading": "READING B -- DIAGNOSTIC ONLY (pre-registration e94ee45 section "
                       "2). It reports magnitudes and nothing else. By design it cannot "
                       "return an adoption outcome of any kind; only Reading A's three "
                       "bands may.",
            "steps": steps}


# ============================================================ CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="",
                    help="scene name, e.g. pgeom or pmask. Recorded, and the grader's "
                         "anchor is per scene.")
    ap.add_argument("--colmap", default="")
    ap.add_argument("--images", default="")
    ap.add_argument("--seed-cloud", default="",
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
    ap.add_argument("--anchor", default=str(ANCHOR_PATH),
                    help="the FROZEN anchor Band 1's cumulative half reads. NOT the "
                         "floors: floors are re-measured per arm and would let the rule "
                         "ratchet, which is the failure the anchor exists to stop.")
    ap.add_argument("--self-anchor", action="store_true",
                    help="this scene has no frozen Tier 3 anchor (P-MASK), so write one "
                         "from its OWN floor means at the moment floors.json is written "
                         "and before the treatment is scored. The first arm's cumulative "
                         "check is then VACUOUS BY CONSTRUCTION, and the grade says so.")
    ap.add_argument("--run-only", action="store_true",
                    help="run the arms and stop. Scoring and grading are a pure function "
                         "of the artifacts and can follow at any time.")
    ap.add_argument("--score-only", action="store_true",
                    help="score and grade artifacts that already exist. No training.")
    ap.add_argument("--regrade", action="store_true",
                    help="recompute the grade from the artifacts already on disk. No "
                         "GPU, no training, no scoring: the verdict is a pure function "
                         "of the reports and stats JSONs, and this proves it.")
    ap.add_argument("--probe-early", action="store_true",
                    help="READING B, diagnostic only. Reads each arm's checkpoints at "
                         "steps 500 and 2000 and reports effect / noise / ratio per "
                         "column. It cannot return an adoption outcome of any kind.")
    ap.add_argument("--summary", action="store_true",
                    help="read the named scenes' grades under --out and emit the "
                         "cross-scene outcome. No GPU.")
    ap.add_argument("--scenes", default="",
                    help="comma-separated scene directory names --summary must find. "
                         "REQUIRED for --summary; see collect_scenes for why a glob is "
                         "not acceptable there.")
    return ap


def _print(doc) -> None:
    print(json.dumps(doc, indent=2, default=str))


def _score_and_grade(a, out: Path) -> dict:
    """PHASE 2 THEN PHASE 3, in that order, enforced by `grade_scene` refusing without
    floors.json. Nobody may choose a floor after seeing a treatment number."""
    fl_doc = None
    if not (out / "FLOORS_DONE").exists():
        print("=== scoring the floor arms, then WRITING floors.json ===", flush=True)
        for tag in FLOOR_TAGS:
            score(out, tag, a.seed_cloud or
                  json.loads((out / f"{tag}.stats.json").read_text())["seed_cloud"])
        fl_doc = write_floors(out, a.scene, a.dn, a.depth_loss_space,
                              self_anchor=a.self_anchor)
        print(f"  floors written: {len(fl_doc['floors'])} metrics", flush=True)
    tag = a.treatment_tag
    print(f"=== scoring {tag} and grading ===", flush=True)
    score(out, tag, a.seed_cloud or
          json.loads((out / "F0.stats.json").read_text())["seed_cloud"])
    doc = grade_scene(out, a.scene, a.dn, tag, a.anchor)
    write_grade(out, tag, doc)
    return doc


GRADE_HEADLINE = ("scene", "arm", "rule", "band1_fired", "band2", "band3_fired",
                  "scene_pass", "scene_drop", "falsifier_triggered_on_this_scene",
                  "geometry_gate", "psnr_verdict")


def main(argv=None) -> None:
    a = build_parser().parse_args(argv)
    out = Path(a.out)

    if a.summary:
        # `--out` is the PARENT holding one subdirectory per scene. Reads grades only, so
        # it needs no GPU, no reports and no scoring.
        if not out.is_dir():
            raise SystemExit(f"--out {out} is not a directory")
        v = combined_verdict(collect_scenes(out, a.scenes, a.treatment_tag))
        v["arm"] = a.treatment_tag
        name = ("combined_verdict.json" if a.treatment_tag == PRIMARY_TAG
                else f"combined_verdict_{a.treatment_tag}.json")
        (out / name).write_text(json.dumps(v, indent=2, default=str))
        _print(v)
        return

    if a.probe_early:
        if not a.scene:
            raise SystemExit("--probe-early needs --scene, which is recorded in the "
                             "output; the directory name is not evidence.")
        r = early_divergence(out, a.scene)
        (out / "early_divergence.json").write_text(json.dumps(r, indent=2, default=str))
        _print(r)
        return

    if a.regrade:
        if not a.scene:
            raise SystemExit("--regrade needs --scene: the frozen anchor is per scene, "
                             "and the directory name is not evidence about what was "
                             "measured.")
        doc = grade_scene(out, a.scene, a.dn, a.treatment_tag, a.anchor)
        write_grade(out, a.treatment_tag, doc)
        _print({k: doc[k] for k in GRADE_HEADLINE}
               | {"drift": [d["metric"] for d in doc["drift"]],
                  "cumulative_check_vacuous": doc["band1"]["cumulative_check_vacuous"]})
        return

    if a.score_only:
        if not a.scene:
            raise SystemExit("--score-only needs --scene")
        doc = _score_and_grade(a, out)
        _print({k: doc[k] for k in GRADE_HEADLINE}
               | {"drift": [d["metric"] for d in doc["drift"]],
                  "cumulative_check_vacuous": doc["band1"]["cumulative_check_vacuous"]})
        return

    if a.floors_only and a.treatment_only:
        raise SystemExit("--floors-only and --treatment-only are mutually exclusive")
    for req in ("scene", "colmap", "images", "seed_cloud"):
        if not getattr(a, req):
            raise SystemExit(f"--{req.replace('_', '-')} is required to run arms (it is "
                             f"optional only for --summary / --regrade / --probe-early)")

    # Checked BEFORE the GPU is touched: a reference cloud that is the training seed
    # invalidates the whole battery, and finding that out after eleven GPU-hours is the
    # expensive way to learn it.
    check_seed_cloud(a.seed_cloud, a.colmap, a.init_ply)

    from bench.runner import require_gpu_exclusive
    require_gpu_exclusive()

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

    if a.run_only or a.floors_only or a.treatment_only:
        print("--run-only / partial queue: stopping before scoring. Grading is a pure "
              "function of these artifacts and can follow at any time.", flush=True)
        return
    doc = _score_and_grade(a, out)
    (out / "ALL_DONE").write_text(_now())
    _print({k: doc[k] for k in GRADE_HEADLINE}
           | {"drift": [d["metric"] for d in doc["drift"]],
              "cumulative_check_vacuous": doc["band1"]["cumulative_check_vacuous"]})


if __name__ == "__main__":
    main()
