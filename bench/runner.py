"""One way to run the trainer from a benchmark, and one way to read its result.

Every wrong number this project has published came from a harness, never from a
kernel. The pattern repeated twice with the same shape:

  * `--steps-scaler` was overridden by the harness that was measuring it;
  * `--budget` was declared `default=300_000` in nerf_synthetic_sweep.py and
    forwarded to every child unconditionally, so `auto_budget()` never ran in a
    single 8-scene sweep and a table labelled "old defaults vs new defaults"
    actually held budget fixed in both arms.

Neither was detectable from the committed JSON, because the harness recorded
its OWN argparse namespace as "the protocol". The trainer knew the truth and
was never asked.

So: harnesses state what they want, the trainer states what it did, and this
module refuses to let those two disagree silently.

    from bench.runner import run
    rep = run({"blender": scene, "steps": 7000})       # budget unspecified
    rep["resolved"]["budget"]                          # -> 100000, recorded

A knob the caller does not pass is left to the trainer AND still recorded, so
"unspecified" stops meaning "unknown". A knob the caller does pass is verified
against what ran, and a mismatch raises rather than quietly producing a number
attributed to the wrong settings.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RunFailed(RuntimeError):
    """The trainer did not produce a usable report."""


class RunDiverged(RuntimeError):
    """The trainer ran with settings other than the ones requested.

    This is the exception that the two historical bugs above would have raised
    on their first run instead of after a week of published numbers.
    """


def _transform_allowed(key: str, want, got, resolved: dict) -> str | None:
    """Is this divergence a documented rewrite by main(), or a harness bug?

    An unconditional allow-list is not good enough here. Listing `steps` as
    "the steps-scaler may rewrite it" would wave through a run that used 3500
    steps when 7000 were requested and no scaler was set -- which is precisely
    the class of bug this module exists to catch. Each allowance therefore has
    to check that its triggering condition actually held.
    """
    scaler = resolved.get("steps_scaler", 1.0)
    if key in ("steps", "relocate_every", "eval_every", "sh_warmup"):
        if scaler != 1.0:
            return f"--steps-scaler {scaler} rewrites it"
        return None
    if key == "start_active":
        # main() clamps start_active to budget//2 when it exceeds the budget,
        # because the parameter tensors are preallocated at `budget`.
        if want > resolved.get("budget", 0) and got == max(1000, resolved["budget"] // 2):
            return f"clamped to budget//2 ({got:,}); requested {want:,} exceeds budget"
        return None
    return None


def _flag(key: str) -> str:
    return "--" + key.replace("_", "-")


def build_cmd(spec: dict, report: Path) -> list[str]:
    """Turn a spec into argv. Only what the caller asked for is passed.

    This is the whole fix in one line: there is no default here. A knob absent
    from `spec` produces no flag, so the trainer's own default applies and gets
    recorded. Harness defaults are what caused the 300k mixup, so this module
    has none.
    """
    cmd = [sys.executable, "-m", "metal_gauss.train", "--report", str(report)]
    for k, v in spec.items():
        if v is None:
            continue
        if isinstance(v, bool):
            cmd.append(_flag(k) if v else _flag("no_" + k))
        else:
            cmd += [_flag(k), str(v)]
    return cmd


def check(spec: dict, report: dict) -> list[str]:
    """Compare requested against resolved. Returns the unexpected divergences."""
    resolved = report.get("resolved")
    if resolved is None:
        raise RunFailed("report has no 'resolved' block -- trainer too old? "
                        "Every benchmarked number needs one.")
    bad = []
    for k, want in spec.items():
        if want is None or k not in resolved:
            continue
        got = resolved[k]
        if got == want:
            continue
        # str() both sides: paths and numbers arrive as strings on argv.
        if str(got) == str(want):
            continue
        msg = f"{k}: requested {want!r}, ran with {got!r}"
        why = _transform_allowed(k, want, got, resolved)
        if why:
            print(f"    [transform] {msg}  ({why})", flush=True)
        else:
            bad.append(msg)
    return bad


def run(spec: dict, *, report: Path | None = None, cwd: Path = ROOT,
        timeout: float | None = None, strict: bool = True) -> dict:
    """Train once and return the trainer's own report.

    Never parses stdout. Scraping is how "278,571 splats" became 571, and how a
    zero exit code from splat-apple counted as a successful run.
    """
    tmp = None
    if report is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".report.json", delete=False)
        tmp.close()
        report = Path(tmp.name)
    report = Path(report)
    if report.exists():
        report.unlink()          # never read a stale report from a failed run

    cmd = build_cmd(spec, report)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd),
                       timeout=timeout)
    wall = time.perf_counter() - t0

    if not report.exists():
        tail = ((p.stderr or p.stdout).strip().splitlines() or ["no output"])[-4:]
        raise RunFailed(f"no report written (exit {p.returncode})\n  cmd: "
                        f"{' '.join(cmd)}\n  " + "\n  ".join(tail))

    rep = json.loads(report.read_text())
    rep["harness_wall_s"] = round(wall, 1)
    rep["cmd"] = cmd

    bad = check(spec, rep)
    if bad:
        detail = "\n  ".join(bad)
        rep["divergence"] = bad
        if strict:
            raise RunDiverged(
                "the trainer ran with settings other than the ones requested. "
                "Any number from this run would be attributed to the wrong "
                f"protocol.\n  {detail}\n  cmd: {' '.join(cmd)}")
        print(f"    [DIVERGENCE, recorded not raised]\n  {detail}", flush=True)

    if tmp is not None:
        Path(tmp.name).unlink(missing_ok=True)
    return rep


# Argv markers that identify a metal-gauss GPU process. Deliberately narrow: a
# machine runs many pythons, and firing on all of them would recreate the
# always-nonzero guard this function was rewritten to fix, just by another route.
_ARGV_MARKERS = ("metal_gauss", "bench/", "bench.", "-m pytest", "pytest ")
_COMPETITOR_NAMES = ("spirula", "brush_app", "brush", "msplat-train")


def _parse_competitors(ps_text: str, self_pid: int, self_pgid: int) -> list[str]:
    """Pure half of `gpu_competitors`, so the discriminating cases are testable.

    A test cannot spawn a process outside its own tree, which is exactly the
    distinction that matters here, so the parsing takes `ps` output as text.

    Two rules, and both are needed:
      * a NON-python executable whose basename is a known competitor trainer --
        the original rule, kept unchanged;
      * a PYTHON process whose argv carries a metal-gauss marker. This is the half
        that was missing: every metal-gauss process is `python`, so matching the
        executable name alone made our own trainer and our own test suite
        invisible, and `require_gpu_exclusive()` returned clean while another
        agent's GPU suite saturated the machine.

    Our own process TREE is excluded -- self, ancestors, descendants and anything
    sharing our process group. That is what keeps the old false positive dead: a
    wrapper shell whose command line merely CONTAINS "brush" is not python and
    does not match a competitor NAME, and our own children are in our tree.
    """
    rows = []
    for line in ps_text.splitlines():
        parts = line.split(None, 4)
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        pid, ppid, pgid = int(parts[0]), int(parts[1]), int(parts[2])
        comm = parts[3]
        args = parts[4] if len(parts) > 4 else ""
        rows.append((pid, ppid, pgid, comm, args))
    by_pid = {r[0]: r for r in rows}

    # DESCENDANTS ARE EXPANDED FROM SELF ONLY, NEVER FROM THE ANCESTOR SET. The
    # first version of this seeded the descendant walk with self AND its
    # ancestors, which pulls in every SIBLING of this process and everything they
    # spawned -- so a competing run launched from a neighbouring shell under the
    # same parent was silently excluded. That is the very case the guard exists
    # for, and a live out-of-tree decoy went undetected until it was fixed.
    kin = {self_pid}
    changed = True
    while changed:
        changed = False
        for pid, ppid, _, _, _ in rows:
            if ppid in kin and pid not in kin:
                kin.add(pid); changed = True
    # Ancestors are excluded too, but they contribute no descendants: a login
    # shell or an agent harness is not a GPU competitor, and its OTHER children
    # may well be.
    mine = set(kin)
    cur, seen = by_pid.get(self_pid), 0
    while cur and cur[1] and cur[1] in by_pid and seen < 64:
        mine.add(cur[1]); cur = by_pid[cur[1]]; seen += 1

    hits = []
    for pid, _ppid, pgid, comm, args in rows:
        if pid in mine or pgid == self_pgid:
            continue
        # `comm` IS TRUNCATED TO 16 CHARACTERS BY macOS ps, so every venv python
        # reads as something like "/private/tmp/cla" and no basename test on it can
        # recognise an interpreter. The executable is therefore taken from the FIRST
        # TOKEN OF argv, which is not truncated. This cost a full round of green
        # tests: the synthetic fixtures were hand-written with untruncated paths, so
        # they could not reproduce the failure, and a live out-of-tree decoy went
        # undetected while the suite passed.
        exe = (args.split(" ", 1)[0] if args else comm).rsplit("/", 1)[-1]
        base = comm.rsplit("/", 1)[-1]
        if base in _COMPETITOR_NAMES or exe in _COMPETITOR_NAMES:
            hits.append(f"{exe or base} (pid {pid})")
        elif exe.startswith("python") and any(m in args for m in _ARGV_MARKERS):
            hits.append(f"python (pid {pid}): {args[:110]}")
    return hits


def gpu_competitors(exclude_pids=()) -> list[str]:
    """Other GPU processes currently running, for timing hygiene.

    Written after a naive guard reported "4 competing GPU procs" during a run
    that had exactly one: `pgrep -f "spirula|msplat|brush"` also matched the
    wrapper shell whose command line CONTAINED that pattern, and the python
    process running the benchmark. A guard that can never read zero is worse
    than no guard, because it trains you to ignore it.

    That fix over-corrected to matching the executable name only, which made
    every metal-gauss process invisible -- see `_parse_competitors`. Both
    failures are now covered, and both have regression tests.
    """
    import os
    import subprocess as sp
    out = sp.run(["ps", "-eo", "pid=,ppid=,pgid=,comm=,args="],
                 capture_output=True, text=True).stdout
    hits = _parse_competitors(out, os.getpid(), os.getpgrp())
    if exclude_pids:
        drop = {str(p) for p in exclude_pids}
        hits = [h for h in hits if not any(f"pid {p})" in h or f"pid {p}):" in h for p in drop)]
    return hits


def require_gpu_exclusive() -> None:
    """Raise unless this process has the GPU to itself.

    Every wall-clock number in this repo assumes exclusive use; a contended run
    is not slightly wrong, it is meaningless.
    """
    busy = gpu_competitors()
    if busy:
        raise RunFailed("GPU is not exclusive -- timings would be invalid. "
                        "Running: " + ", ".join(busy))


def psnr(rep: dict):
    return (rep.get("metrics") or {}).get("psnr")


def wall_s(rep: dict):
    return (rep.get("metrics") or {}).get("wall_s")
