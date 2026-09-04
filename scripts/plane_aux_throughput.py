#!/usr/bin/env python3
"""Fixed-capacity step time for `--depth-source center` vs `plane-aux`, interleaved, n>=3.

Answers two pre-registered questions at once (Task 19, first commit on `feat/plane-aux`):

  * the COST CEILING -- plane-aux must be <= 1.05x the `center` path;
  * the ROUTE (ii) TRIGGER -- route (i) forms the plane depth in ~10 per-pixel torch
    dispatches, and route (ii) (folding the ray-plane division into the kernel with
    closed-form Jacobians) is built ONLY IF that costs more than 5% of the step time.
    Both are the same measurement: the marginal ms/step of plane-aux over center.

PROTOCOL, from research/metal-gauss.md section 11.6a, which found the Tier 2 headline was
mislabelled:

  * `--no-grow`. With the default `--grow`, `active` climbs from --start-active toward the
    budget until 70% of the schedule, so a "500k splats" figure is a GROWING-capacity
    average over ~400k. The two regimes differ by 12% and are not interchangeable.
  * ms/step is read from the report's own `wall_s` between two eval steps, so dataset load
    and startup are excluded. NEVER from `ms_per_step`, which divides total wall by steps.
  * INTERLEAVED A/B/A/B/..., so within-session thermal drift cannot align with the arm.
  * GPU exclusive via bench.runner.require_gpu_exclusive() -- not a hand-rolled pgrep,
    which only knows the competitors its author thought of (section 11.6).

Every guard here exists because its absence cost this project time:
  * macOS HAS NO `timeout`(1). The watchdog is hand-rolled.
  * A 0-byte log after 90 s is an impossible healthy state -- usually a stale `FileBaton`
    lock. Cleared per arm, but ONLY when `lsof` shows it unheld.
  * `uv run --frozen` silently reverts scripts/fix_openmp.py, so it is re-applied after
    every `uv` call.
  * The report artifact is ASSERTED to exist. A harness once printed six `done` lines in
    three seconds for six crashed runs.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

# MG_ROOT, never `__file__`. This script is COPIED to an immutable snapshot inside the
# output directory before it runs (section 11.3c: freeze everything a long job reads, and
# bash/python both re-read source), so deriving the repo root from `__file__` points at the
# OUTPUT TREE and every path below it vanishes. run_tier1_arms.sh carries the same warning
# and this script reproduced the bug anyway, on its first launch, in three seconds --
# caught by the report-artifact assertion, which is what that assertion is for.
ROOT = Path(os.environ.get("MG_ROOT") or Path(__file__).resolve().parents[1])
if not (ROOT / "metal_gauss" / "train.py").exists():
    raise SystemExit(f"MG_ROOT={ROOT} is not a metal-gauss checkout (no metal_gauss/train.py). "
                     f"Set MG_ROOT when running from a snapshot.")
sys.path.insert(0, str(ROOT))
from bench.runner import require_gpu_exclusive  # noqa: E402

LOCK = Path.home() / ("Library/Caches/torch_extensions/py312_cpu/"
                      "metal_gauss_metal/lock")


def clear_stale_lock() -> None:
    """Remove the extension build lock, but only if nothing holds it.

    A held lock means a real concurrent build and removing it would corrupt that build.
    `lsof` returning nothing is the proof that it is stale.
    """
    if not LOCK.exists():
        return
    held = subprocess.run(["lsof", str(LOCK)], capture_output=True, text=True)
    if held.stdout.strip():
        raise SystemExit(f"the extension lock is HELD by another build:\n{held.stdout}")
    LOCK.unlink()
    print(f"  cleared stale lock {LOCK}", flush=True)


def fix_openmp() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts/fix_openmp.py")],
                   capture_output=True, text=True)


def run_arm(tag: str, out_dir: Path, common: list[str], extra: list[str],
            watchdog_s: float, empty_log_s: float = 90.0) -> Path:
    report = out_dir / f"{tag}.json"
    log = out_dir / f"{tag}.log"
    if report.exists():
        print(f"  {tag}: report exists, skipping", flush=True)
        return report
    clear_stale_lock()
    cmd = ["caffeinate", "-i", str(ROOT / ".venv/bin/python"), "-m", "metal_gauss.train",
           *common, *extra, "--report", str(report)]
    env = dict(os.environ, PYTHONUNBUFFERED="1")   # a killed process loses buffered stdout
    t0 = time.perf_counter()
    with log.open("wb") as fh:
        p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
                             env=env, start_new_session=True)
        while p.poll() is None:
            time.sleep(2.0)
            el = time.perf_counter() - t0
            # LIVENESS, not just correctness: asserting the report exists at the end is
            # structurally blind to a run that never finishes.
            if el > empty_log_s and log.stat().st_size == 0:
                p.kill()
                raise SystemExit(f"{tag}: 0-byte log after {el:.0f}s -- almost certainly "
                                 f"the FileBaton lock. Check `lsof {LOCK}`.")
            if el > watchdog_s:
                p.kill()
                raise SystemExit(f"{tag}: watchdog at {el:.0f}s")
    fix_openmp()
    if not report.exists():
        tail = log.read_text(errors="replace").strip().splitlines()[-6:]
        raise SystemExit(f"{tag}: NO REPORT written (rc={p.returncode})\n" + "\n".join(tail))
    return report


def ms_per_step(report: Path, lo: int, hi: int) -> float:
    """(wall[hi] - wall[lo]) / (hi - lo), in ms. Never `ms_per_step`, which includes load."""
    d = json.loads(report.read_text())
    by = {e["step"]: e["wall_s"] for e in d["log"] if "wall_s" in e}
    if lo not in by or hi not in by:
        raise SystemExit(f"{report.name}: need eval steps {lo} and {hi}, have "
                         f"{sorted(by)[:12]}")
    return 1000.0 * (by[hi] - by[lo]) / (hi - lo)


def summarise(name: str, vals: list[float], unit: str = "ms") -> dict:
    return {"arm": name, "n": len(vals), "runs": [round(v, 2) for v in vals],
            "mean": round(statistics.mean(vals), 3),
            "spread": round(max(vals) - min(vals), 3),
            "stdev": round(statistics.stdev(vals), 3) if len(vals) > 1 else None,
            "unit": unit}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--depth-dir")
    ap.add_argument("--normal-dir")
    ap.add_argument("--init-ply")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-resolution", type=int, default=1920)
    ap.add_argument("--budget", type=int, default=500_000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1400)
    ap.add_argument("--lo", type=int, default=400)
    ap.add_argument("--hi", type=int, default=1200)
    ap.add_argument("--watchdog", type=float, default=1800.0)
    a = ap.parse_args()

    require_gpu_exclusive()
    if a.hi % a.lo or a.hi <= a.lo or a.steps < a.hi:
        raise SystemExit(f"--hi ({a.hi}) must be a multiple of --lo ({a.lo}), greater "
                         f"than it, and <= --steps ({a.steps})")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    common = ["--colmap", a.colmap, "--images", a.images,
              "--max-resolution", str(a.max_resolution), "--steps", str(a.steps),
              "--budget", str(a.budget), "--no-grow", "--num-downscales", "0",
              # eval_every == lo so the two sampling points BOTH exist: the trainer evals
              # at `step % eval_every == 0` and at the final step only, so a mismatched
              # eval_every silently leaves the report with no usable timing pair.
              "--eval-split-every", "8", "--eval-every", str(a.lo), "--seed", "42",
              # R1p: the recipe with depth-normal consistency OFF, so the depth loss is
              # the only consumer of the depth map and the arms differ in ONE variable.
              "--flatten-loss-weight", "1.0", "--depth-loss-weight", "1.0",
              "--normal-loss-weight", "0.2"]
    for flag, val in (("--depth-dir", a.depth_dir), ("--normal-dir", a.normal_dir),
                      ("--init-ply", a.init_ply)):
        if val:
            common += [flag, val]

    rows: dict[str, list[float]] = {"center": [], "plane-aux": []}
    for i in range(a.repeats):
        for src in ("center", "plane-aux"):          # interleaved, never blocked
            tag = f"{src}_{i}"
            rep = run_arm(tag, out, common, ["--depth-source", src], a.watchdog)
            v = ms_per_step(rep, a.lo, a.hi)
            rows[src].append(v)
            print(f"  [{i+1}/{a.repeats}] {src:10s} {v:7.2f} ms/step", flush=True)

    res = {k: summarise(k, v) for k, v in rows.items()}
    c, p = res["center"]["mean"], res["plane-aux"]["mean"]
    res["ratio"] = round(p / c, 4)
    res["marginal_ms"] = round(p - c, 3)
    res["marginal_frac_of_center"] = round((p - c) / c, 4)
    # Pre-registered, before any number existed. Both are reported even when they pass.
    res["cost_ceiling_1.05x"] = "PASS" if p <= 1.05 * c else "MISS"
    res["route_ii_trigger_5pct"] = "BUILD ROUTE (ii)" if (p - c) > 0.05 * c else "DO NOT BUILD"
    (out / "throughput.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
