#!/usr/bin/env python3
"""Grade a needle batch against its PRE-REGISTERED bars, from the artifacts only.

Every number is read from a `--report` JSON or a `splatstats` JSON on disk. Nothing is
taken from stdout, a log line or a summary, which is the rule this repo keeps paying to
relearn.

THE BARS, fixed before any arm ran (playroom_0821, 500k splats, 30k steps):

    needle_frac        <= 0.050     the Brush/LFS band on this scene
    aspect_p50         >= 0.40
    hard_needle_frac   <= 0.001
    masked PSNR        >= baseline - 0.10 dB
    thin-axis p50      within this batch's own floor of the arm's recipe value
    on-seed@1cm        within this batch's own floor of the arm's recipe value

TWO REPORTING RULES, also fixed in advance.

  * THIN-AXIS IS REPORTED UNGATED BESIDE GATED, with `thin_axis_evaluated` for both.
    splatstats' default gate admits only splats within 5 cm of the seed, so two arms can
    be scored over different POPULATIONS; a delta between them is then a composition
    statistic wearing an orientation statistic's name. That mistake has produced two
    confident wrong results in this project. `--ungated-suffix` names the second
    splatstats JSON (`splat_stats.py --thin-axis-gate -1`).

  * An arm that changes `--num-downscales` has a different resolution schedule, so its
    `ms_per_step` is not comparable to any other arm's and is printed with a `!` and
    excluded from any comparison.

WHAT THIS FILE IS NOT. It grades; it does not decide. A PASS here means the six columns
cleared six numbers. The falsifier for the whole line of work -- an arm that clears
`needle_frac` while the operator still sees needles -- is not a column and cannot be.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BARS = {"needle_frac": ("<=", 0.050), "aspect_p50": (">=", 0.40),
        "hard_needle_frac": ("<=", 0.001)}
PSNR_DROP_ALLOWED = 0.10

# Arms whose ms/step is not comparable to the rest of the batch, by flag.
SCHEDULE_FLAGS = ("--num-downscales",)


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def read_arm(out: Path, arm: str, ungated_suffix: str = ".ungated.json") -> dict:
    """Everything about one arm, from its files. Absent inputs stay None -- never 0.0,
    which would grade as a spectacular pass on `needle_frac` and a failure on `aspect`.

    THE PLY WINS OVER THE REPORT, and on one arm in this batch they genuinely differ.
    `--filter-3d` bakes Mip-Splatting's widened scales into the export (`export_ply`),
    while the per-eval `shape_metrics` reads `p["log_scales"]`, the raw parameter the
    filter never touches. The ply is what a client receives, so `<arm>.shape.json` from
    `bench/ply_shape.py` is preferred when present and the disagreement is reported."""
    rep = _load(out / f"{arm}.json")
    st = _load(out / f"{arm}.stats.json")
    un = _load(out / f"{arm}{ungated_suffix}")
    plyshape = _load(out / f"{arm}.shape.json")
    m = (rep or {}).get("metrics") or {}
    report_shape = m.get("shape") or {}
    shape = plyshape if plyshape else report_shape
    sm = (st or {}).get("metrics") or {}
    um = (un or {}).get("metrics") or {}
    resolved = (rep or {}).get("resolved") or {}
    row = {
        "arm": arm,
        "present": rep is not None,
        "git": ((rep or {}).get("env") or {}).get("git"),
        "dirty": ((rep or {}).get("env") or {}).get("dirty"),
        "psnr_masked": m.get("psnr_masked"),
        "lpips": m.get("lpips"),
        "coverage": m.get("coverage"),
        "ms_per_step": m.get("ms_per_step"),
        "n_splats": m.get("n_splats"),
        "aspect_p50": shape.get("aspect_p50"),
        "needle_frac": shape.get("needle_frac"),
        "hard_needle_frac": shape.get("hard_needle_frac"),
        "smid_p50_mm": shape.get("smid_p50_mm"),
        "smax_p50_mm": shape.get("smax_p50_mm"),
        "thin_gated_p50": sm.get("thin_axis_angle_median_deg"),
        "thin_gated_n": sm.get("thin_axis_evaluated"),
        "thin_ungated_p50": um.get("thin_axis_angle_median_deg"),
        "thin_ungated_n": um.get("thin_axis_evaluated"),
        "on_seed_1cm": sm.get("on_seed_frac_1cm"),
        "num_downscales": resolved.get("num_downscales"),
        "schedule_arm": False,
    }
    # `.get`, not `[]`: reports written before per-eval shape metrics existed carry no
    # "shape" key at all, and an arm scored by an older binary carries a partial one.
    # A time course that CRASHES on those is worse than one that is short.
    row["shape_source"] = "ply" if plyshape else ("report" if report_shape else None)
    row["report_shape"] = report_shape or None
    # A disagreement is information, not an error: it is how --filter-3d announces itself.
    row["shape_disagreement"] = None
    if plyshape and report_shape:
        d = {k: plyshape[k] - report_shape[k] for k in ("aspect_p50", "needle_frac")
             if k in plyshape and k in report_shape}
        if any(abs(v) > 1e-6 for v in d.values()):
            row["shape_disagreement"] = d
    row["shape_course"] = [(e["step"], (e.get("shape") or {}).get("needle_frac"),
                            (e.get("shape") or {}).get("aspect_p50"))
                           for e in ((rep or {}).get("log") or []) if e.get("shape")
                           is not None]
    return row


def grade(rows: list[dict], baseline: str, floors: dict | None) -> list[dict]:
    """Attach a verdict per arm. `floors` is the harness `floors.json` for this batch --
    NOT a floor set carried over from another batch, another binary or another machine."""
    base = next((r for r in rows if r["arm"] == baseline), None)
    if base is None or not base["present"]:
        raise SystemExit(f"baseline arm {baseline} has no report; refusing to grade")
    ds0 = base["num_downscales"]
    fl = (floors or {}).get("floors") or {}

    def floor_of(key):
        return (fl.get(key) or {}).get("repeat_floor")

    thin_floor = floor_of("stats.thin_axis_angle_median_deg")
    seed_floor = floor_of("stats.on_seed_frac_1cm")
    for r in rows:
        r["schedule_arm"] = (r["num_downscales"] is not None
                             and r["num_downscales"] != ds0)
        fails, notes = [], []
        if not r["present"]:
            r["verdict"], r["fails"] = "NO REPORT", ["absent"]
            continue
        for k, (op, bar) in BARS.items():
            v = r.get(k)
            if v is None:
                fails.append(f"{k}: MISSING")
            elif (op == "<=" and v > bar) or (op == ">=" and v < bar):
                fails.append(f"{k}={v:.6g} {'>' if op == '<=' else '<'} {bar:g}")
        if r["psnr_masked"] is None or base["psnr_masked"] is None:
            fails.append("psnr_masked: MISSING")
        else:
            d = r["psnr_masked"] - base["psnr_masked"]
            r["psnr_delta"] = d
            if d < -PSNR_DROP_ALLOWED:
                fails.append(f"psnr {d:+.4f} dB < -{PSNR_DROP_ALLOWED}")
        for name, key, base_key, floor in (
                ("thin-axis p50", "thin_gated_p50", "thin_gated_p50", thin_floor),
                ("on-seed@1cm", "on_seed_1cm", "on_seed_1cm", seed_floor)):
            if r[key] is None or base[base_key] is None:
                notes.append(f"{name}: not scored")
            elif floor is None:
                notes.append(f"{name}: no floor in floors.json, ungraded")
            else:
                r[f"{key}_delta"] = r[key] - base[base_key]
        # A gated thin-axis delta is only an ORIENTATION statement if the two arms were
        # scored over comparable populations. Say so rather than assume it.
        if r["thin_gated_n"] and base["thin_gated_n"]:
            ratio = r["thin_gated_n"] / base["thin_gated_n"]
            r["thin_gated_n_ratio"] = ratio
            if abs(ratio - 1.0) > 0.02:
                notes.append(f"gated thin-axis population differs {100*(ratio-1):+.1f}% "
                             f"from the baseline's: the gated delta is a COMPOSITION "
                             f"statistic here, read the ungated column")
        if r["shape_source"] is None:
            notes.append("no shape measured from either the report or the ply")
        elif r["shape_source"] == "report":
            notes.append("shape read from the REPORT, not the ply -- run "
                         "bench/ply_shape.py; on a --filter-3d arm these differ and only "
                         "the ply is what gets delivered")
        if r["shape_disagreement"]:
            d = r["shape_disagreement"]
            notes.append("ply shape differs from the in-memory shape by "
                         + ", ".join(f"{k} {v:+.6f}" for k, v in d.items())
                         + " -- graded on the PLY. Expected on --filter-3d, which bakes "
                           "the widened scales into the export.")
        if r["thin_ungated_p50"] is None:
            notes.append("UNGATED THIN-AXIS NOT MEASURED -- rerun splat_stats.py with "
                         "--thin-axis-gate -1; the gated number alone is not reportable")
        r["fails"], r["notes"] = fails, notes
        r["verdict"] = "PASS" if not fails else "FAIL"
    return rows


def render(rows: list[dict], baseline: str) -> str:
    hdr = ("arm", "needle%", "hard%", "aspect", "PSNR", "dPSNR", "LPIPS",
           "thin gated", "n", "thin ungated", "n", "on-seed1cm", "ms/step", "verdict")
    out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        f = lambda v, s="{:.4f}": "--" if v is None else s.format(v)  # noqa: E731
        out.append("| " + " | ".join([
            r["arm"],
            f(None if r["needle_frac"] is None else 100 * r["needle_frac"], "{:.3f}"),
            f(None if r["hard_needle_frac"] is None else 100 * r["hard_needle_frac"],
              "{:.4f}"),
            f(r["aspect_p50"]), f(r["psnr_masked"]),
            "base" if r["arm"] == baseline else f(r.get("psnr_delta"), "{:+.4f}"),
            f(r["lpips"]),
            f(r["thin_gated_p50"], "{:.3f}"), f(r["thin_gated_n"], "{:,.0f}"),
            f(r["thin_ungated_p50"], "{:.3f}"), f(r["thin_ungated_n"], "{:,.0f}"),
            f(r["on_seed_1cm"], "{:.5f}"),
            (f(r["ms_per_step"], "{:.2f}") + ("!" if r["schedule_arm"] else "")),
            r["verdict"]]) + " |")
    for r in rows:
        for x in r.get("fails", []):
            out.append(f"* {r['arm']} FAIL: {x}")
        for x in r.get("notes", []):
            out.append(f"* {r['arm']} note: {x}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--baseline", default="B0a")
    ap.add_argument("--ungated-suffix", default=".ungated.json")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    arms = [x for x in a.arms.split(",") if x]
    rows = grade([read_arm(a.out, arm, a.ungated_suffix) for arm in arms],
                 a.baseline, _load(a.out / "floors.json"))
    print(render(rows, a.baseline))
    if a.json:
        a.json.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
