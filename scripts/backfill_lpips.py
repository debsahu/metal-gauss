#!/usr/bin/env python3
"""Merge each arm's LPIPS into its training report.

    scripts/backfill_lpips.py OUT_DIR [ARM ...]

`lpips_eval.py` computes LPIPS into `<arm>.dump/lpips.json`, and nothing ever merged it
into `<arm>.json`. So `metrics.lpips` read absent on every arm of every scene, and the
Stage 4 gate -- masked PSNR >= 24 dB AND LPIPS <= 0.25 -- was only ever half-checked, on
runs that had already been paid for in full.

Idempotent, and it will not clobber an existing value: a report scored more recently than
its dump must win. Nothing but `metrics.lpips` is touched, because other tooling reads
these files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def backfill(out: Path, arms: list[str] | None = None) -> int:
    """Returns the number of reports updated."""
    if arms is None:
        arms = [p.stem for p in sorted(out.glob("*.json"))
                if not p.name.endswith(".stats.json")]
    n = 0
    for arm in arms:
        rp = out / f"{arm}.json"
        lp = out / f"{arm}.dump" / "lpips.json"
        if not rp.exists() or not lp.exists():
            continue
        try:
            rep = json.loads(rp.read_text())
        except Exception:
            continue
        if not isinstance(rep.get("metrics"), dict):
            continue                                  # not a training report
        if isinstance(rep["metrics"].get("lpips"), (int, float)):
            continue                                  # already present; do not clobber
        rep["metrics"]["lpips"] = json.loads(lp.read_text())["mean"]
        rp.write_text(json.dumps(rep, indent=2, default=str))
        print(f"  {arm}: lpips {rep['metrics']['lpips']:.4f}")
        n += 1
    return n


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    out = Path(sys.argv[1])
    n = backfill(out, sys.argv[2:] or None)
    print(f"backfilled {n} report(s) in {out}")


if __name__ == "__main__":
    main()
