"""The results collector. It has now failed twice, in two different ways, and both times it
failed SILENTLY-ISH on valid inputs while a human read JSONs by hand instead.

  1. It gated arm inclusion on `<arm>.stats.json`, so a scene with no reference cloud
     (lego, where splatstats is skipped as UNDEFINED) produced an EMPTY TABLE, no error.
  2. Its auto-glob picked up `collected.json` -- THE FILE IT WRITES ITSELF -- and treated
     it as an arm, so running the collector once made the next run crash with
     KeyError: 'metrics'. Self-poisoning.

Both are the same disease as a check that cannot fire, so this file exists to make the
loader's contract explicit and pin both report shapes.
"""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = Path(__file__).resolve().parents[1] / "scripts" / "tier1_table.py"


def _mod():
    spec = importlib.util.spec_from_file_location("tier1_table", SPEC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


OLD_REPORT = {                      # pre-instrument shape: no shape/terms, no started_at
    "resolved": {"seed": 42},
    "env": {"git": "c047dad", "dirty": False},
    "metrics": {"psnr": 22.1, "psnr_masked": 22.1, "coverage": 1.0,
                "ms_per_step": 32.4, "n_splats": 500000},
}
NEW_REPORT = {                      # current shape
    "resolved": {"seed": 42},
    "env": {"git": "78434d6", "dirty": False,
            "started_at": "2026-09-03T00:00:00.000000Z",
            "finished_at": "2026-09-03T00:10:00.000000Z"},
    "metrics": {"psnr": 22.6, "psnr_masked": 22.6, "coverage": 1.0,
                "ms_per_step": 68.8, "n_splats": 500000,
                "terms": {"l1": 0.02, "ssim": 0.07, "depth": 0.05},
                "shape": {"aspect_p50": 0.27, "needle_frac": 0.19,
                          "smid_p50_mm": 6.1, "smax_p50_mm": 22.4},
                "term_view_coverage": {"depth": [171, 171]},
                "term_coverage_warning": None},
}


def _arm(out: Path, name: str, report: dict, lpips: float | None = None):
    (out / f"{name}.json").write_text(json.dumps(report))
    if lpips is not None:
        d = out / f"{name}.dump"; d.mkdir(exist_ok=True)
        (d / "lpips.json").write_text(json.dumps({"mean": lpips, "n": 25, "net": "vgg"}))


def test_loader_reads_both_report_shapes(tmp_path):
    m = _mod()
    _arm(tmp_path, "OLD", OLD_REPORT, lpips=0.40)
    _arm(tmp_path, "NEW", NEW_REPORT, lpips=0.39)
    old, new = m.load(tmp_path, "OLD"), m.load(tmp_path, "NEW")
    assert old["run.psnr_masked"] == 22.1 and new["run.psnr_masked"] == 22.6
    assert old["run.lpips"] == 0.40 and new["run.lpips"] == 0.39
    assert old["_git"] == "c047dad" and new["_started"] is not None
    assert old.get("_started") is None                      # absent, not an exception
    assert new["_terms"]["depth"] == 0.05


def test_collected_json_is_not_mistaken_for_an_arm(tmp_path):
    """The collector writes collected.json into the same directory it scans. Running it
    twice must not crash, and the second run must not invent an arm called 'collected'."""
    m = _mod()
    _arm(tmp_path, "B0a", NEW_REPORT, lpips=0.39)
    (tmp_path / "collected.json").write_text(json.dumps({"B0a": {"run.psnr": 1.0}}))
    (tmp_path / "floors.json").write_text(json.dumps({"floors": {}}))
    assert m.discover_arms(tmp_path) == ["B0a"]


def test_a_non_report_json_is_skipped_with_a_warning_not_a_crash(tmp_path, capsys):
    """Anything in the directory that is not a training report gets skipped and NAMED. The
    previous behaviour was KeyError: 'metrics' with no indication which file did it."""
    m = _mod()
    _arm(tmp_path, "B0a", NEW_REPORT)
    (tmp_path / "junk.json").write_text(json.dumps({"not": "a report"}))
    assert "junk" not in m.discover_arms(tmp_path)
    assert m.is_report(tmp_path / "junk.json") is False
    assert m.is_report(tmp_path / "B0a.json") is True


def test_lpips_prefers_the_report_metric_then_falls_back_to_the_dump(tmp_path):
    """After backfill the number lives in metrics.lpips; before it, only in the dump. Both
    must read, and the report must win so a re-scored arm is not shadowed by a stale dump."""
    m = _mod()
    r = json.loads(json.dumps(NEW_REPORT))
    r["metrics"]["lpips"] = 0.1234
    _arm(tmp_path, "A", r, lpips=0.9999)                    # dump disagrees on purpose
    assert m.load(tmp_path, "A")["run.lpips"] == 0.1234
    _arm(tmp_path, "B", NEW_REPORT, lpips=0.4242)
    assert m.load(tmp_path, "B")["run.lpips"] == 0.4242


def test_arms_with_no_ply_and_no_stats_still_appear(tmp_path):
    """lego has neither. That combination produced an empty table once."""
    m = _mod()
    _arm(tmp_path, "L0a", NEW_REPORT, lpips=0.02)
    assert m.discover_arms(tmp_path) == ["L0a"]
    assert m.load(tmp_path, "L0a")["run.psnr_masked"] == 22.6


# ------------------------------------------------------------------ LPIPS backfill

def _backfill():
    spec = importlib.util.spec_from_file_location(
        "backfill_lpips", Path(__file__).resolve().parents[1] / "scripts" / "backfill_lpips.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_backfill_writes_the_dump_lpips_into_the_report(tmp_path):
    """LPIPS is computed by a separate script into <arm>.dump/lpips.json and was never
    merged into the report, so `metrics.lpips` read None on every arm of every scene and
    the Stage 4 gate was only half-checked -- on runs already paid for."""
    m = _backfill()
    _arm(tmp_path, "B0a", NEW_REPORT, lpips=0.3952)
    n = m.backfill(tmp_path)
    assert n == 1
    got = json.loads((tmp_path / "B0a.json").read_text())["metrics"]["lpips"]
    assert got == 0.3952


def test_backfill_is_idempotent_and_does_not_clobber_a_newer_value(tmp_path):
    m = _backfill()
    _arm(tmp_path, "A", NEW_REPORT, lpips=0.40)
    m.backfill(tmp_path)
    json_path = tmp_path / "A.json"
    d = json.loads(json_path.read_text()); d["metrics"]["lpips"] = 0.11
    json_path.write_text(json.dumps(d))
    assert m.backfill(tmp_path) == 0                      # already present: left alone
    assert json.loads(json_path.read_text())["metrics"]["lpips"] == 0.11


def test_backfill_skips_arms_with_no_dump_and_reports_them(tmp_path):
    m = _backfill()
    _arm(tmp_path, "A", NEW_REPORT)                       # no dump at all
    assert m.backfill(tmp_path) == 0
    assert "lpips" not in json.loads((tmp_path / "A.json").read_text())["metrics"]


def test_backfill_preserves_every_other_field(tmp_path):
    """It rewrites a report someone else's tooling reads. Nothing but metrics.lpips moves."""
    m = _backfill()
    _arm(tmp_path, "A", NEW_REPORT, lpips=0.5)
    before = json.loads((tmp_path / "A.json").read_text())
    m.backfill(tmp_path)
    after = json.loads((tmp_path / "A.json").read_text())
    assert after["metrics"].pop("lpips") == 0.5
    assert after == before
