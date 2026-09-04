"""Task 21 step 1's rescoring tool: the refusals, not the arithmetic.

Each test names a way the step-1 table could be quietly wrong: a self-check that
never ran, an arm counted twice, a fabricated file swept in with the real ones,
a requested arm silently absent.
"""
import json

import numpy as np
import pytest
from PIL import Image

from bench import lpips_attr as LA
from bench import lpips_rescore as LR


def _dump(tmp_path, n=3, with_gt=True, with_lpips=None):
    d = tmp_path / "arm.dump"
    d.mkdir()
    rng = np.random.default_rng(0)
    for i in range(n):
        a = rng.integers(0, 255, (4, 5, 3), dtype=np.uint8)
        Image.fromarray(a).save(d / f"v{i}_render.png")
        if with_gt:
            Image.fromarray(a).save(d / f"v{i}_gt.png")
    if with_lpips is not None:
        (d / "lpips.json").write_text(json.dumps(with_lpips))
    return d


def _result(scene, arm, passed=True, stage="step1_rescore"):
    return {"stage": stage, "scene": scene, "arm": arm, "dump": "/x",
            "n_views": 2, "has_mask_files": False, "image_hw": [4, 5],
            "self_check": {"passed": passed},
            "per_view": {"vgg": {"a": 0.3, "b": 0.4}},
            "dist": {"vgg": LR.dist([0.3, 0.4])},
            "black_render_pixel_frac": {"per_view": {}, "mean": 0.0}}


def test_pairs_refuses_a_render_with_no_ground_truth(tmp_path):
    """A silently skipped pair changes the mean and nothing says so."""
    d = _dump(tmp_path, with_gt=False)
    with pytest.raises(FileNotFoundError, match="ground truth"):
        LR.pairs(d)


def test_pairs_refuses_an_empty_directory(tmp_path):
    d = tmp_path / "e.dump"; d.mkdir()
    with pytest.raises(FileNotFoundError, match="no \\*_render.png"):
        LR.pairs(d)


def test_summary_refuses_a_failed_self_check(tmp_path):
    """A failed self-check means this scoring chain is not the one that produced
    the published numbers. Reporting a table anyway is the worst outcome."""
    LA.write_json(tmp_path / "step1" / "a__b.json", _result("a", "b", passed=False))
    with pytest.raises(ValueError, match="self-check FAILED"):
        LR.summary(_args(tmp_path))


def test_summary_refuses_a_duplicate_scene_arm(tmp_path):
    LA.write_json(tmp_path / "step1" / "a__b.json", _result("a", "b"))
    LA.write_json(tmp_path / "step1" / "copy.json", _result("a", "b"))
    with pytest.raises(ValueError, match="duplicate"):
        LR.summary(_args(tmp_path))


def test_summary_refuses_a_foreign_file_in_its_own_tree(tmp_path):
    """Guard your own fixtures out of the data: a summary tool in this project
    once globbed its own fabricated smoke fixtures."""
    (tmp_path / "step1").mkdir(parents=True)
    (tmp_path / "step1" / "smoke.json").write_text(
        json.dumps({"kind": "smoke_fixture", "schema": 1}))
    with pytest.raises(ValueError, match="kind"):
        LR.summary(_args(tmp_path))


def test_summary_refuses_a_right_kind_wrong_stage_file(tmp_path):
    LA.write_json(tmp_path / "step1" / "x.json", _result("a", "b", stage="step3_probe"))
    with pytest.raises(ValueError, match="stage"):
        LR.summary(_args(tmp_path))


def test_summary_refuses_a_missing_required_arm(tmp_path):
    LA.write_json(tmp_path / "step1" / "a__b.json", _result("a", "b"))
    with pytest.raises(ValueError, match="missing required"):
        LR.summary(_args(tmp_path, require="a/b,a/c"))


def test_summary_accepts_a_complete_grid(tmp_path, capsys):
    LA.write_json(tmp_path / "step1" / "a__b.json", _result("a", "b"))
    LA.write_json(tmp_path / "step1" / "a__c.json", _result("a", "c"))
    LR.summary(_args(tmp_path, require="a/b,a/c"))
    assert "2 arm(s)" in capsys.readouterr().out


def test_dist_reports_the_tail_not_only_the_mean():
    """A bimodal per-view LPIPS is a coverage problem and a flat one is a
    capacity problem; a mean cannot tell them apart, which is why step 1 is
    specified on the distribution."""
    flat = LR.dist([0.30] * 9 + [0.31])
    bimodal = LR.dist([0.25] * 9 + [0.75])
    assert flat["mean"] == pytest.approx(bimodal["mean"], abs=0.02)
    assert flat["frac_above_mean_plus_0p1"] == 0.0
    assert bimodal["frac_above_mean_plus_0p1"] == pytest.approx(0.1)
    assert bimodal["spread"] > 10 * flat["spread"]


def _args(root, require="", scenes=""):
    import argparse
    return argparse.Namespace(out_root=str(root), scenes=scenes, require=require)


def test_composite_replaces_the_DROPPED_region_and_not_the_kept_one(tmp_path):
    """MASK POLARITY, and it is the expensive one to get wrong. `View.mask` is
    uint8 255 = KEEP, so compositing must paste ground truth where the mask is
    0. Inverted, it would neutralise the region the trainer was actually scored
    on and leave the ignored region charged -- and it would still return a
    plausible, smaller LPIPS, which is exactly the shape of answer step 1b is
    looking for."""
    import torch
    from PIL import Image
    render = torch.zeros(4, 4, 3)
    gt = torch.ones(4, 4, 3)
    m = np.zeros((4, 4), np.uint8)
    m[:, :2] = 255                       # left half KEPT, right half dropped
    p = tmp_path / "m.png"
    Image.fromarray(m).save(p)
    out = LR.composite(render, gt, p)
    assert torch.equal(out[:, :2], render[:, :2]), "kept region must survive"
    assert torch.equal(out[:, 2:], gt[:, 2:]), "dropped region must become GT"


def test_score_refuses_composite_on_a_dump_with_no_masks(tmp_path):
    import argparse
    d = _dump(tmp_path)
    a = argparse.Namespace(dump=str(d), scene="s", arm="a", out_root=str(tmp_path / "o"),
                           nets="vgg", mask_mode="composite")
    with pytest.raises(ValueError, match="no \\*_mask.png"):
        LR.score(a)


def test_summary_keys_on_mask_mode_so_the_two_readings_do_not_collide(tmp_path, capsys):
    """The full-frame and composite numbers for one arm are two measurements of
    the same arm, not a duplicate."""
    full = _result("a", "b"); full["mask_mode"] = "full"
    comp = _result("a", "b"); comp["mask_mode"] = "composite"
    comp["self_check"] = {"passed": None}
    LA.write_json(tmp_path / "step1" / "a__b.json", full)
    LA.write_json(tmp_path / "step1" / "a__b__composite.json", comp)
    LR.summary(_args(tmp_path))
    assert "2 arm(s)" in capsys.readouterr().out
