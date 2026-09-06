"""The verdict tool, exercised END TO END on synthetic artifacts.

WHY THIS EXISTS. bench/bilagrid_verdict.py is the last thing that runs after
about nine GPU-hours, and until this file it had never been executed on anything
-- only imported. This task has already lost a block to a script whose `--help`
worked and whose body did not, and Task 21 lost six renders the same way. A
verdict tool that crashes on the shape of a real report is the most expensive
possible place for that to happen.

The artifacts here are SYNTHETIC and are built to the schema the trainer actually
writes (checked against a real report). Nothing here validates a scientific
claim; it validates that the tool runs, reads the right fields, and returns the
verdict the rules say it should for a known input.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _write_arm(out: Path, tag: str, *, appearance: str, seed: int, lpips: float,
               psnr: float, on_seed: float, thin: float, aspect: float,
               needle: float, ms: float = 100.0):
    (out / f"{tag}.json").write_text(json.dumps({
        "schema": 1,
        "resolved": {"seed": seed, "appearance": appearance, "budget": 500000,
                     "steps": 30000, "colmap": "/x/sparse/0"},
        "env": {"git": "deadbee"},
        "metrics": {"psnr": psnr - 0.2, "psnr_masked": psnr, "coverage": 1.0,
                    "wall_s": 2800.0, "n_splats": 500000, "ms_per_step": ms,
                    "shape": {"aspect_p50": aspect, "needle_frac": needle,
                              "hard_needle_frac": 0.015, "smid_p50_mm": 7.1,
                              "smax_p50_mm": 25.8},
                    "appearance": None if appearance == "off" else
                                  {"mode": "bilagrid", "params": 4202496,
                                   "dims": [16, 16, 8], "max_abs_dev": 0.31,
                                   "tv": 0.004, "reg_weight": 10.0,
                                   "lr_final": 2.3e-05}}}))
    (out / f"{tag}.stats.json").write_text(json.dumps({
        "seed_cloud": "/x/points3D.tsdf.txt",
        "metrics": {"on_seed_frac_1cm": on_seed, "on_seed_frac_2cm": on_seed * 3,
                    "thin_axis_angle_p50": thin, "opacity_p50": 0.14,
                    "dark_frac": 0.02}}))
    d = out / f"{tag}.dump"; d.mkdir(exist_ok=True)
    (d / "lpips.json").write_text(json.dumps({"net": "vgg", "mean": lpips, "n": 25,
                                              "per_view": {"v": lpips}}))


def _scene(out: Path, *, treat_lpips, treat_psnr=22.60, treat_on_seed=0.0890,
           treat_thin=29.45, treat_aspect=0.2730, treat_needle=0.1930):
    """Base floors from the real P-GEOM R1 figures; treatment supplied per case."""
    for tag, s, lp, ps in (("pg_base_a", 42, 0.39680, 22.6037),
                           ("pg_base_b", 42, 0.39705, 22.5900),
                           ("pg_base_c", 43, 0.39640, 22.6200)):
        _write_arm(out, tag, appearance="off", seed=s, lpips=lp, psnr=ps,
                   on_seed=0.0890, thin=29.45, aspect=0.2730, needle=0.1930)
    for i, (tag, s) in enumerate((("pg_bila_a", 42), ("pg_bila_b", 42),
                                  ("pg_bila_c", 43))):
        _write_arm(out, tag, appearance="bilagrid", seed=s,
                   lpips=treat_lpips + i * 1e-5, psnr=treat_psnr,
                   on_seed=treat_on_seed, thin=treat_thin,
                   aspect=treat_aspect, needle=treat_needle)


def _run(out: Path, tmp: Path):
    r = subprocess.run([sys.executable, "bench/bilagrid_verdict.py",
                        "--out-dir", str(out), "--scene", "pg",
                        "--seed-cloud", "/x/points3D.tsdf.txt",
                        "--json", str(tmp / "v.json")],
                       cwd=str(ROOT), capture_output=True, text=True)
    assert r.returncode == 0, f"verdict tool crashed:\n{r.stdout}\n{r.stderr}"
    return json.loads((tmp / "v.json").read_text()), r.stdout


def test_the_tool_RUNS_and_a_material_gain_with_clean_geometry_is_KEEP(tmp_path):
    """The happy path, which is also the "does it run at all" test. A 0.020
    improvement clears both the floor (~0.00065) and the 0.015 materiality."""
    out = tmp_path / "out"; out.mkdir()
    _scene(out, treat_lpips=0.37680)                      # -0.020 = improvement
    doc, txt = _run(out, tmp_path)
    assert doc["plan_rule"]["verdict"] == "KEEP", txt
    assert doc["plan_rule"]["beats_materiality"] and doc["plan_rule"]["beats_floor"]
    assert doc["three_band"]["band2"] == "PASS"           # photometric form
    assert doc["three_band"]["band2_geometry_form"] == "WITHIN FLOOR"
    assert doc["three_band"]["band2_forms_disagree"] is True
    assert doc["three_band"]["verdict"] == "KEEP"
    assert doc["rules_disagree"] is False


def test_a_REAL_BUT_IMMATERIAL_gain_is_DROP_and_the_rules_DISAGREE(tmp_path):
    """AMENDMENT 1's middle case, and the one the evidence predicts: an
    improvement well above the n=3 floor but under the 0.015 materiality bar.

    The plan's rule says DROP (0.015 was the bar Task 21's own build decision
    used). The amended three-band rule says KEEP (nothing collapsed, geometry did
    no harm, PSNR held). THE TWO RULES DISAGREE, and the tool must SAY SO rather
    than quietly returning one of them."""
    out = tmp_path / "out"; out.mkdir()
    _scene(out, treat_lpips=0.39380)                      # -0.003: ~5x floor, < 0.015
    doc, txt = _run(out, tmp_path)
    assert doc["plan_rule"]["beats_floor"] is True
    assert doc["plan_rule"]["beats_materiality"] is False
    assert doc["plan_rule"]["verdict"] == "DROP"
    assert doc["three_band"]["verdict"] == "KEEP"
    assert doc["rules_disagree"] is True
    assert "THE TWO RULES DISAGREE" in txt


def test_a_gain_INSIDE_the_floor_is_not_read_as_a_gain(tmp_path):
    """"Not resolvable at n=3", never "zero". The tool must not report a
    sub-floor move as beating the floor."""
    out = tmp_path / "out"; out.mkdir()
    _scene(out, treat_lpips=0.39675)                      # -0.00005, floor ~0.00065
    doc, _ = _run(out, tmp_path)
    assert doc["plan_rule"]["beats_floor"] is False
    assert doc["plan_rule"]["verdict"] == "DROP"


def test_a_grid_that_buys_LPIPS_by_DAMAGING_GEOMETRY_is_caught(tmp_path):
    """The failure mode the amended Band 2 exists to catch, and the reason the
    inversion is acceptable: a large LPIPS win bought by on-seed collapsing.
    Fails if the photometric form degenerated into "skip Band 2"."""
    out = tmp_path / "out"; out.mkdir()
    _scene(out, treat_lpips=0.35000, treat_on_seed=0.0500)   # huge win, on-seed -44%
    doc, txt = _run(out, tmp_path)
    assert doc["three_band"]["band2"] == "FAIL", txt
    assert doc["three_band"]["verdict"] == "DROP"
    assert doc["plan_rule"]["verdict"] == "DROP", "geometry worsened, so the plan drops too"
    assert "stats.on_seed_frac_1cm" in doc["plan_rule"]["geometry_columns_worsened"]


def test_a_PSNR_loss_beyond_the_allowance_fires_band3(tmp_path):
    """Pre-registered expectation: masked PSNR FALLS when the gaussians are freed
    from per-frame photometry. Band 3 allows 0.25 dB; beyond that is a hard DROP
    however good LPIPS is."""
    out = tmp_path / "out"; out.mkdir()
    _scene(out, treat_lpips=0.35000, treat_psnr=22.6037 - 0.60)
    doc, _ = _run(out, tmp_path)
    assert doc["three_band"]["band3"]["fired"] is True
    assert doc["three_band"]["verdict"] == "DROP"
    # ...and a loss INSIDE the allowance must not fire, or the test above is
    # satisfied by a band that fires on any loss at all.
    out2 = tmp_path / "out2"; out2.mkdir()
    _scene(out2, treat_lpips=0.35000, treat_psnr=22.6037 - 0.10)
    doc2, _ = _run(out2, tmp_path / "b")
    assert doc2["three_band"]["band3"]["fired"] is False


def test_the_tool_REFUSES_arms_scored_against_different_reference_clouds(tmp_path):
    """CLAUDE.md records an 11.6 deg thin-axis error from exactly this -- larger
    than every recipe gain it was used to judge. Mixed references must abort, not
    average."""
    out = tmp_path / "out"; out.mkdir()
    _scene(out, treat_lpips=0.37680)
    st = json.loads((out / "pg_bila_a.stats.json").read_text())
    st["seed_cloud"] = "/x/SOMETHING_ELSE.txt"
    (out / "pg_bila_a.stats.json").write_text(json.dumps(st))
    r = subprocess.run([sys.executable, "bench/bilagrid_verdict.py",
                        "--out-dir", str(out), "--scene", "pg",
                        "--seed-cloud", "/x/points3D.tsdf.txt",
                        "--json", str(tmp_path / "v.json")],
                       cwd=str(ROOT), capture_output=True, text=True)
    assert r.returncode != 0
    assert "reference cloud" in (r.stdout + r.stderr)


def test_the_tool_REFUSES_arms_that_are_not_the_configuration_they_claim(tmp_path):
    """A base arm that actually ran WITH the appearance model, or a treatment arm
    that ran without, would produce a verdict about nothing. The tool reads the
    mode out of each report rather than trusting the filename."""
    out = tmp_path / "out"; out.mkdir()
    _scene(out, treat_lpips=0.37680)
    rep = json.loads((out / "pg_base_a.json").read_text())
    rep["resolved"]["appearance"] = "bilagrid"
    (out / "pg_base_a.json").write_text(json.dumps(rep))
    r = subprocess.run([sys.executable, "bench/bilagrid_verdict.py",
                        "--out-dir", str(out), "--scene", "pg",
                        "--seed-cloud", "/x/points3D.tsdf.txt",
                        "--json", str(tmp_path / "v.json")],
                       cwd=str(ROOT), capture_output=True, text=True)
    assert r.returncode != 0
    assert "not the configurations" in (r.stdout + r.stderr)
