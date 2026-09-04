#!/usr/bin/env python
"""Shape columns of the Tier 3 battery, read straight from a ply. No GPU, no scene.

`smid_p50_mm`, `smax_p50_mm` and `hard_needle_frac` are in `train.shape_metrics` and in
every training report, but a report only exists for a run someone launched with `--report`.
This reads them back from the plys themselves, which is what makes the frozen Band-1 anchor
(the scene's Tier 3 F-arm means) computable for columns that were added after those arms ran.

It calls `train.shape_metrics` rather than reimplementing it: a second implementation of a
gate's own statistic is how two numbers with the same name come to mean different things.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch


def log_scales_from_ply(path: str) -> torch.Tensor:
    """(N,3) LOG scales, exactly as the trainer parameterises them.

    The ply's `scale_i` fields are already log-space (`train.export_ply` writes
    `p["log_scales"]` unchanged), so there is no exp/log round trip here and none is
    wanted -- `io.load_ply` exponentiates and would have to be undone.
    """
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"]
    return torch.from_numpy(
        np.stack([np.asarray(v[f"scale_{i}"], dtype=np.float32) for i in range(3)], 1).copy())


def shape_of_ply(path: str) -> dict:
    from metal_gauss.train import shape_metrics
    ls = log_scales_from_ply(path)
    out = shape_metrics(ls)
    out["splats"] = int(ls.shape[0])
    return out


def anchor(rows: list[dict], keys: list[str]) -> dict:
    """Mean and spread of each column over the arms given -- the frozen Band-1 anchor.

    Refuses fewer than three arms: an n<3 spread is what research/metal-gauss.md 8.2
    retracted a whole batch of claims over, and a Band-1 cumulative check anchored on a
    two-run mean would inherit that.
    """
    if len(rows) < 3:
        raise RuntimeError(f"an anchor needs n >= 3 arms, got {len(rows)}")
    return {k: {"mean": statistics.fmean(r[k] for r in rows),
                "spread": max(r[k] for r in rows) - min(r[k] for r in rows)}
            for k in keys}


COLUMNS = ["aspect_p50", "needle_frac", "hard_needle_frac", "smid_p50_mm", "smax_p50_mm"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plys", nargs="+")
    ap.add_argument("--anchor", action="store_true",
                    help="also emit the mean/spread over the plys given (needs n >= 3)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rows = []
    for p in a.plys:
        r = shape_of_ply(p)
        r["ply"] = p
        rows.append(r)
        print(f'{Path(p).name:28s} splats {r["splats"]:>8,}  aspect_p50 {r["aspect_p50"]:.6f}  '
              f'needle {r["needle_frac"]:.6f}  hard_needle {r["hard_needle_frac"]:.6f}  '
              f'smid {r["smid_p50_mm"]:.4f}mm  smax {r["smax_p50_mm"]:.4f}mm')
    out = {"kind": "ply_shape_columns", "rows": rows}
    if a.anchor:
        out["anchor"] = anchor(rows, COLUMNS)
        print(json.dumps(out["anchor"], indent=2, sort_keys=True))
    if a.out:                      # written LAST and only on success
        Path(a.out).write_text(json.dumps(out, indent=2, sort_keys=True))
    return out


if __name__ == "__main__":
    main()
