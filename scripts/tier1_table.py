#!/usr/bin/env python3
"""Collect every arm's pre-registered metrics into one table (plan Task 10, step 3).

    scripts/tier1_table.py OUT_DIR [ARM ...]

Reads report / splatstats / LPIPS JSONs and the exported ply. NEVER stdout: a number that
cannot be traced to a file on disk does not go in the results section.

Two things this file got wrong before, both worth keeping in mind:

  * It gated inclusion on `<arm>.stats.json` existing. A scene with no reference cloud --
    lego has none, so splatstats is skipped as UNDEFINED rather than failing -- therefore
    produced an EMPTY TABLE with no error. A collector that silently returns nothing when
    handed valid arms is the same disease as a check that cannot fire. It now gates on the
    report, treats stats and LPIPS as optional, and says out loud what it could not find.

  * It had no in-plane aspect column. On P-GEOM's R1, opacity p50, dark fraction and
    thin-axis all reported the arm as HEALTHIER than baseline while its renders were
    visibly worse; what had actually happened was a mid-axis collapse into needles
    (smid 6.62 -> 1.35 mm with smax held at ~23 mm, needle fraction 16.6% -> 56.8%). No
    metric in the battery could see it. These columns are that missing row.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

KEYS = [
    ("run.psnr_masked", "masked PSNR", "{:.4f}"),
    ("run.psnr", "unmasked PSNR", "{:.4f}"),
    ("run.lpips", "LPIPS", "{:.4f}"),
    ("run.coverage", "coverage", "{:.4f}"),
    ("ply.aspect_p50", "in-plane aspect p50", "{:.4f}"),
    ("ply.needle_frac", "needle frac (<0.1)", "{:.4f}"),
    ("ply.smid_p50_mm", "smid p50 (mm)", "{:.3f}"),
    ("ply.smax_p50_mm", "smax p50 (mm)", "{:.3f}"),
    ("stats.thin_axis_angle_median_deg", "thin-axis p50 deg", "{:.2f}"),
    ("stats.thin_axis_frac_under_15deg", "thin-axis <15deg", "{:.4f}"),
    ("stats.on_seed_frac_1cm", "on-seed @1cm", "{:.4f}"),
    ("stats.on_seed_frac_2cm", "on-seed @2cm", "{:.4f}"),
    ("stats.opacity_p50", "opacity p50", "{:.4f}"),
    ("stats.scale_p50", "scale p50 (max axis)", "{:.5f}"),
    ("stats.dark_splat_frac", "dark frac", "{:.4f}"),
    ("run.ms_per_step", "ms/step", "{:.2f}"),
    ("run.n_splats", "splats", "{:.0f}"),
]


def ply_shape_metrics(path: Path) -> dict:
    """In-plane aspect and needle fraction from the exported ply.

    A gaussian's three activated scales sorted smin <= smid <= smax. `smax` is the surface
    extent, `smid` the other in-plane axis, `smin` the thickness. Flatten legitimately
    drives smin down -- that is a disc. smid collapsing while smax holds is a NEEDLE, and
    that is the failure the geometry terms produced with the blending-weight path open.
    """
    import plyfile
    v = plyfile.PlyData.read(str(path))["vertex"]
    s = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float64))
    s = np.sort(s, axis=1)                       # smin, smid, smax
    smid, smax = s[:, 1], s[:, 2]
    aspect = smid / np.maximum(smax, 1e-12)
    return {"ply.aspect_p50": float(np.median(aspect)),
            "ply.needle_frac": float((aspect < 0.1).mean()),
            "ply.smid_p50_mm": float(np.median(smid) * 1000.0),
            "ply.smax_p50_mm": float(np.median(smax) * 1000.0),
            "ply.smin_p50_mm": float(np.median(s[:, 0]) * 1000.0)}


def load(out: Path, arm: str) -> dict:
    v: dict = {}
    rep = json.loads((out / f"{arm}.json").read_text())
    v.update({f"run.{k}": rep["metrics"][k] for k in
              ("psnr_masked", "psnr", "coverage", "ms_per_step", "n_splats")
              if isinstance(rep["metrics"].get(k), (int, float))})
    st_path = out / f"{arm}.stats.json"
    if st_path.exists():
        st = json.loads(st_path.read_text())
        v.update({f"stats.{k}": x for k, x in st["metrics"].items()
                  if isinstance(x, (int, float))})
        v["_seed_cloud"] = st.get("seed_cloud")
    lp = out / f"{arm}.dump" / "lpips.json"
    if lp.exists():
        v["run.lpips"] = json.loads(lp.read_text())["mean"]
    ply = out / f"{arm}.ply"
    if ply.exists():
        v.update(ply_shape_metrics(ply))
    env = rep.get("env", {})
    v["_seed"] = rep["resolved"]["seed"]
    v["_git"] = env.get("git")
    v["_started"] = env.get("started_at")
    v["_terms"] = rep["metrics"].get("terms")
    return v


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    out = Path(sys.argv[1])
    arms = sys.argv[2:] or [p.stem for p in sorted(out.glob("*.json"))
                            if not p.name.endswith((".stats.json", "floors.json"))]
    missing = [a for a in arms if not (out / f"{a}.json").exists()]
    if missing:
        print(f"!! no report JSON for: {', '.join(missing)} (looked in {out})",
              file=sys.stderr)
    data = {a: load(out, a) for a in arms if (out / f"{a}.json").exists()}
    if not data:
        sys.exit(f"no arms found in {out}; asked for {arms or '<auto>'}")

    w = max(len(lbl) for _, lbl, _ in KEYS) + 3
    print("| metric".ljust(w) + "| " + " | ".join(f"{a:>12}" for a in data) + " |")
    print("|" + "-" * (w - 1) + "|" + "|".join("-" * 14 for _ in data) + "|")
    for key, lbl, fmt in KEYS:
        if not any(key in v for v in data.values()):
            continue
        cells = [fmt.format(v[key]) if key in v else "--" for v in data.values()]
        print(f"| {lbl}".ljust(w) + "| " + " | ".join(f"{c:>12}" for c in cells) + " |")
    print()
    for a, v in data.items():
        print(f"{a}: seed={v['_seed']} git={v['_git']} started={v['_started']} "
              f"ref={v.get('_seed_cloud', '(none -- geometry metrics UNDEFINED)')}")
        if v.get("_terms"):
            print("     terms " + "  ".join(f"{k} {x:.5f}" for k, x in v["_terms"].items()))
    json.dump(data, open(out / "collected.json", "w"), indent=1)
    print(f"\nwrote {out/'collected.json'}")


if __name__ == "__main__":
    main()
