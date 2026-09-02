#!/usr/bin/env python3
"""Collect every arm's pre-registered metrics into one table (plan Task 10, step 3).

Reads report / splatstats / LPIPS JSONs ONLY -- never stdout. A number that cannot be
traced to a file on disk does not go in the results section.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

KEYS = [("run.psnr_masked", "masked PSNR", "{:.4f}"), ("run.psnr", "unmasked PSNR", "{:.4f}"),
        ("run.lpips", "LPIPS", "{:.4f}"), ("run.coverage", "coverage", "{:.4f}"),
        ("stats.thin_axis_angle_median_deg", "thin-axis p50 deg", "{:.2f}"),
        ("stats.thin_axis_frac_under_15deg", "thin-axis <15deg", "{:.4f}"),
        ("stats.on_seed_frac_1cm", "on-seed @1cm", "{:.4f}"),
        ("stats.on_seed_frac_2cm", "on-seed @2cm", "{:.4f}"),
        ("stats.opacity_p50", "opacity p50", "{:.4f}"),
        ("stats.scale_p50", "scale p50 (max axis)", "{:.5f}"),
        ("stats.dark_splat_frac", "dark frac", "{:.4f}"),
        ("run.ms_per_step", "ms/step", "{:.2f}"), ("run.n_splats", "splats", "{:.0f}")]


def load(out: Path, arm: str) -> dict:
    rep = json.loads((out / f"{arm}.json").read_text())
    st = json.loads((out / f"{arm}.stats.json").read_text())
    v = {f"stats.{k}": x for k, x in st["metrics"].items() if isinstance(x, (int, float))}
    v.update({f"run.{k}": rep["metrics"][k] for k in
              ("psnr_masked", "psnr", "coverage", "ms_per_step", "n_splats")
              if isinstance(rep["metrics"].get(k), (int, float))})
    lp = out / f"{arm}.dump" / "lpips.json"
    if lp.exists():
        v["run.lpips"] = json.loads(lp.read_text())["mean"]
    v["_seed"] = rep["resolved"]["seed"]
    v["_git"] = rep["env"]["git"]
    v["_seed_cloud"] = st.get("seed_cloud")
    return v


def main() -> None:
    out = Path(sys.argv[1])
    arms = [a for a in sys.argv[2:]] or ["B0a", "B0b", "B0c", "F1", "R1"]
    data = {a: load(out, a) for a in arms if (out / f"{a}.stats.json").exists()}
    w = max(len(lbl) for _, lbl, _ in KEYS) + 2
    print("| metric".ljust(w) + "| " + " | ".join(f"{a:>12}" for a in data) + " |")
    print("|" + "-" * (w - 1) + "|" + "|".join("-" * 14 for _ in data) + "|")
    for key, lbl, fmt in KEYS:
        if not any(key in v for v in data.values()):
            continue
        cells = [fmt.format(v[key]) if key in v else "--" for v in data.values()]
        print(f"| {lbl}".ljust(w) + "| " + " | ".join(f"{c:>12}" for c in cells) + " |")
    print()
    for a, v in data.items():
        print(f"{a}: seed={v['_seed']} git={v['_git']} seed_cloud={v['_seed_cloud']}")
    json.dump(data, open(out / "collected.json", "w"), indent=1)
    print(f"\nwrote {out/'collected.json'}")


if __name__ == "__main__":
    main()
